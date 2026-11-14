"""Landscape coupling — the world model beyond the column.

Fields run as one batched Field (B = parcels). The graph object routes
each day's runoff downhill along a row-stochastic weight matrix and pools
deep drainage into a shared aquifer store, so neighboring parcels exchange
water and the landscape water budget stays exactly conservative: routed
fluxes are internal transfers, losses only through ET and basin outflow.

Two graph implementations share the `weights(field, fluxes)` protocol:

- `FieldGraph`: static row-stochastic matrix (topography prior).
- `RouteGraph`: a `RouteNet` graph-attention router producing a dynamic
  W_t conditioned on daily fluxes — learned routing that still conserves
  mass by construction (masked softmax over the directed adjacency).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from nasa_farm.constants import DTYPE
from nasa_farm.state.field import Field


@dataclass(frozen=True, slots=True)
class FieldGraph:
    """Row-stochastic downstream routing: w[i, j] = fraction of field i's
    surface runoff reaching field j next day. Row sums ≤ 1; the remainder
    leaves the basin."""

    w: torch.Tensor          # (B, B)
    share_aquifer: bool = True

    @classmethod
    def chain(cls, n: int, frac: float = 0.5) -> "FieldGraph":
        """1-D hillside: each field passes `frac` of runoff to i+1."""
        w = torch.zeros(n, n, dtype=DTYPE)
        idx = torch.arange(n - 1)
        w[idx, idx + 1] = frac
        return cls(w)

    def weights(self, field: Field, fluxes: dict) -> torch.Tensor:
        return self.w

    def to(self, device) -> "FieldGraph":
        return FieldGraph(self.w.to(device), self.share_aquifer)


class RouteNet(nn.Module):
    """Graph-attention edge scorer: attention logits on adjacent directed
    edges (i → j, i upstream of j) softmaxed over each source's outbound
    edges, so every field's water is conserved by construction.

    Node features: [sw_col, runoff, tws] of that day — the driver sees
    both how much water is moving and the local wetness memory."""

    def __init__(self, n_feats: int = 3, hidden: int = 32,
                 seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.embed = nn.Sequential(
            nn.Linear(n_feats, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden))
        self.score = nn.Linear(2 * hidden, 1)
        self.to(DTYPE)

    def forward(self, feats: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """feats (B, F) → W (B, B) row-stochastic on `adj` (incl self)."""
        h = self.embed(feats)
        b = h.shape[0]
        hi = h.unsqueeze(1).expand(b, b, h.shape[-1])
        hj = h.unsqueeze(0).expand(b, b, h.shape[-1])
        e = self.score(torch.cat([hi, hj], dim=-1)).squeeze(-1)
        e = e.masked_fill(~adj, -1e30)
        return torch.softmax(e, dim=-1)


@dataclass(frozen=True, slots=True)
class RouteGraph:
    """Learned routing: `net` produces a dynamic row-stochastic W_t on a
    fixed directed adjacency built from topography (`from_topo`)."""

    adj: torch.Tensor        # (B, B) bool — directed downstream edges
    net: RouteNet
    share_aquifer: bool = True

    @classmethod
    def from_topo(cls, lat: torch.Tensor, lon: torch.Tensor,
                  elev: torch.Tensor, k: int = 3,
                  hidden: int = 32,
                  seed: int | None = None) -> "RouteGraph":
        """Adjacency: each field links to its k nearest downhill neighbors
        plus itself; fields at local minima only keep their self-loop, so
        their runoff ponds locally instead of vanishing."""
        xy = torch.stack([lat, lon], dim=-1).to(DTYPE)
        b = xy.shape[0]
        d = torch.cdist(xy, xy)
        _, nbr = d.topk(min(k + 1, b), dim=-1, largest=False)
        adj = torch.zeros(b, b, dtype=torch.bool)
        elev = elev.to(DTYPE)
        for i in range(b):
            lo = [j for j in nbr[i].tolist()
                  if j != i and elev[j] <= elev[i]]
            adj[i, i] = True
            adj[i, torch.tensor(lo or [i], dtype=torch.long)] = True
        return cls(adj=adj, net=RouteNet(3, hidden, seed))

    def weights(self, field: Field, fluxes: dict) -> torch.Tensor:
        s = field.state
        feats = torch.stack(
            [s.sw.sum(-1) / s.sw.sum(-1).max().clamp(min=1e-9),
             fluxes["runoff"] / fluxes["runoff"].max().clamp(min=1e-9),
             s.tws / s.tws.max().clamp(min=1e-9)], dim=-1)
        return self.net(feats, self.adj)


def fit_route(graph: RouteGraph, feats: torch.Tensor,
              w_ref: torch.Tensor, epochs: int = 100,
              lr: float = 1e-3) -> dict:
    """Imitation-fit the router on (features → reference matrix) pairs:
    daily flux snapshots against a routing prior or observed downstream
    arrival proportions. Returns the loss history."""
    opt = torch.optim.Adam(graph.net.parameters(), lr=lr)
    best, hist = float("inf"), []
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        w = graph.net(feats.to(DTYPE), graph.adj)
        loss = (w - w_ref).pow(2).mean()
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
        best = min(best, hist[-1])
    return {"best_loss": round(best, 8), "history": hist}


class Landscape:
    """Wraps a WorldModel.step call with inter-field routing.

    State between days: `pending` (B,) surface water arriving at the next
    step; the aquifer `tws` channels are already per-field — sharing means
    all fields' deep drainage mixes into one pool, redistributed uniformly
    (fast-aquifer approximation) or kept per-field when share_aquifer=False.
    """

    def __init__(self, model, graph: FieldGraph | RouteGraph) -> None:
        self.model = model
        self.graph = graph
        self.pending: torch.Tensor | None = None

    def step(self, field: Field, action: dict, forcing_d: dict,
             t: int = 0) -> tuple[Field, dict]:
        if self.pending is None:
            self.pending = torch.zeros(field.batch, dtype=DTYPE,
                                       device=field.state.sw.device)
        f2, fluxes = self.model.step(
            field, action,
            {**forcing_d, "precip": forcing_d["precip"] + self.pending}, t)
        s = f2.state
        routed = self.graph.weights(f2, fluxes).t() @ fluxes["runoff"]
        self.pending = routed
        if self.graph.share_aquifer:
            mean_tws = s.tws.mean().expand_as(s.tws)
            s = s.replace(tws=s.tws + 0.5 * (mean_tws - s.tws))
        f2 = f2.__class__(state=s, soil=f2.soil, lat=f2.lat,
                          elev=f2.elev, ctx=f2.ctx)
        return f2, fluxes

    def rollout(self, field: Field, forcing, actions,
                observer=None) -> Field:
        for t in range(forcing.days):
            field, fx = self.step(field, actions.day(t), forcing.day(t), t)
            if observer is not None:
                observer(t, field.state, fx)
        return field
