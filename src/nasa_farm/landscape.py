"""Landscape coupling — the world model beyond the column.

Fields run as one batched Field (B = parcels). `FieldGraph` routes each
day's runoff downhill along a row-stochastic weight matrix and pools deep
drainage into a shared aquifer store, so neighboring parcels exchange water
and the landscape water budget stays exactly conservative: routed fluxes
are internal transfers, losses only through ET and basin outflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.state.field import Field, FieldState


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

    def to(self, device) -> "FieldGraph":
        return FieldGraph(self.w.to(device), self.share_aquifer)


class Landscape:
    """Wraps a WorldModel.step call with inter-field routing.

    State between days: `pending` (B,) surface water arriving at the next
    step; the aquifer `tws` channels are already per-field — sharing means
    all fields' deep drainage mixes into one pool, redistributed uniformly
    (fast-aquifer approximation) or kept per-field when share_aquifer=False.
    """

    def __init__(self, model, graph: FieldGraph) -> None:
        self.model = model
        self.graph = graph
        self.pending: torch.Tensor | None = None

    def step(self, field: Field, action: dict, forcing_d: dict,
             ) -> tuple[Field, dict]:
        if self.pending is None:
            self.pending = torch.zeros(field.batch, dtype=DTYPE,
                                       device=field.state.sw.device)
        f2, fluxes = self.model.step(
            field, action,
            {**forcing_d, "precip": forcing_d["precip"] + self.pending})
        s = f2.state
        routed = self.graph.w.t() @ fluxes["runoff"]      # (B,) arrival
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
            field, fx = self.step(field, actions.day(t), forcing.day(t))
            if observer is not None:
                observer(t, field.state, fx)
        return field
