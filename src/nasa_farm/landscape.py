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
    def chain(cls, n: int, fraction: float = 0.5) -> "FieldGraph":
        """1-D hillside: each field passes `frac` of runoff to i+1."""
        w = torch.zeros(n, n, dtype=DTYPE)
        index = torch.arange(n - 1)
        w[index, index + 1] = fraction
        return cls(w)

    def weights(self, field: Field, fluxes: dict) -> torch.Tensor:
        return self.w

    def to(self, device) -> "FieldGraph":
        return FieldGraph(self.w.to(device), self.share_aquifer)


class Route(nn.Module):
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
                adjacency: torch.Tensor) -> torch.Tensor:
        """feats (B, F) → W (B, B) row-stochastic on `adj` (incl self)."""
        h = self.embed(feats)
        b = h.shape[0]
        high = h.unsqueeze(1).expand(b, b, h.shape[-1])
        hj = h.unsqueeze(0).expand(b, b, h.shape[-1])
        e = self.score(torch.cat([high, hj], dim=-1)).squeeze(-1)
        e = e.masked_fill(~adjacency, -1e30)
        return torch.softmax(e, dim=-1)


@dataclass(frozen=True, slots=True)
class RouteGraph:
    """Learned routing: `net` produces a dynamic row-stochastic W_t on a
    fixed directed adjacency built from topography (`from_topo`)."""

    adjacency: torch.Tensor        # (B, B) bool — directed downstream edges
    net: Route
    share_aquifer: bool = True

    @classmethod
    def from_topo(cls, latitude: torch.Tensor, longitude: torch.Tensor,
                  elevation: torch.Tensor, k: int = 3,
                  hidden: int = 32,
                  seed: int | None = None) -> "RouteGraph":
        """Adjacency: each field links to its k nearest downhill neighbors
        plus itself; fields at local minima only keep their self-loop, so
        their runoff ponds locally instead of vanishing."""
        xy = torch.stack([latitude, longitude], dim=-1).to(DTYPE)
        b = xy.shape[0]
        d = torch.cdist(xy, xy)
        _, nbr = d.topk(min(k + 1, b), dim=-1, largest=False)
        adjacency = torch.zeros(b, b, dtype=torch.bool)
        elevation = elevation.to(DTYPE)
        for i in range(b):
            low = [j for j in nbr[i].tolist()
                  if j != i and elevation[j] <= elevation[i]]
            adjacency[i, i] = True
            adjacency[i, torch.tensor(low or [i], dtype=torch.long)] = True
        return cls(adjacency=adjacency, net=Route(3, hidden, seed))

    def weights(self, field: Field, fluxes: dict) -> torch.Tensor:
        s = field.state
        feats = torch.stack(
            [s.sw.sum(-1) / s.sw.sum(-1).max().clamp(min=1e-9),
             fluxes["runoff"] / fluxes["runoff"].max().clamp(min=1e-9),
             s.tws / s.tws.max().clamp(min=1e-9)], dim=-1)
        return self.net(feats, self.adjacency)


def fit_route(graph: RouteGraph, feats: torch.Tensor,
              w_ref: torch.Tensor, epochs: int = 100,
              rate: float = 1e-3) -> dict:
    """Imitation-fit the router on (features → reference matrix) pairs:
    daily flux snapshots against a routing prior or observed downstream
    arrival proportions. Returns the loss history."""
    opt = torch.optim.Adam(graph.net.parameters(), lr=rate)
    best, history = float("inf"), []
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        w = graph.net(feats.to(DTYPE), graph.adjacency)
        loss = (w - w_ref).pow(2).mean()
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        best = min(best, history[-1])
    return {"best_loss": round(best, 8), "history": history}


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

    def step(self, field: Field, action: dict, forcing_day: dict,
             t: int = 0) -> tuple[Field, dict]:
        if self.pending is None:
            self.pending = torch.zeros(field.batch, dtype=DTYPE,
                                       device=field.state.sw.device)
        f2, fluxes = self.model.step(
            field, action,
            {**forcing_day, "precip": forcing_day["precip"] + self.pending}, t)
        s = f2.state
        routed = self.graph.weights(f2, fluxes).t() @ fluxes["runoff"]
        self.pending = routed
        if self.graph.share_aquifer:
            mean_tws = s.tws.mean().expand_as(s.tws)
            s = s.replace(tws=s.tws + 0.5 * (mean_tws - s.tws))
        f2 = f2.__class__(state=s, soil=f2.soil, latitude=f2.latitude,
                          elevation=f2.elevation, context=f2.context)
        return f2, fluxes

    def rollout(self, field: Field, forcing, actions,
                observer=None) -> Field:
        for t in range(forcing.days):
            field, fx = self.step(field, actions.day(t), forcing.day(t), t)
            if observer is not None:
                observer(t, field.state, fx)
        return field
