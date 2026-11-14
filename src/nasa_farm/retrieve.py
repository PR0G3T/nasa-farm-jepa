"""Realizable-subgoal retrieval — hierarchical planning over experience.

Generated latent goals are often physically unrealizable: the world model
cannot reach them, so any plan chasing them wastes its horizon. Keeping a
library of *realized* rollout states and answering every goal query with
the nearest realized state guarantees the subgoal is reachable by
construction — the retrieval fix that makes latent-space hierarchies
sound. Rows live in the model's z-space (latent enabled) or feature space
(otherwise), matching `mppi(goal=...)` conditioning.
"""

from __future__ import annotations

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.policy import _day_feats
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


class StateLibrary:
    """Realized states from past rollouts, rows (N, K) in goal space."""

    def __init__(self) -> None:
        self.rows: torch.Tensor | None = None

    def __len__(self) -> int:
        return 0 if self.rows is None else self.rows.shape[0]

    def append(self, z: torch.Tensor) -> None:
        z = z.detach().to(DTYPE).reshape(-1, z.shape[-1])
        self.rows = z if self.rows is None else torch.cat([self.rows, z])

    def nearest(self, goal: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Closest realized state(s) to `goal` — always a state the model
        actually visited, so the returned subgoal is realizable."""
        if self.rows is None:
            raise ValueError("StateLibrary is empty — record rollouts first")
        g = goal.reshape(-1, goal.shape[-1]).to(DTYPE)
        d = torch.cdist(g, self.rows)
        index = (d.argmin(-1) if k == 1
               else d.topk(k, largest=False).indices)
        return self.rows[index]


def record(lib: StateLibrary, model, field: Field, forcing: Forcing,
           actions: Actions, every: int = 1) -> None:
    """Roll once, append realized states. `every` subsamples days."""
    with torch.inference_mode():
        for t in range(forcing.days):
            fd, action = forcing.day(t), actions.day(t)
            field, _ = model.step(field, action, fd, t)
            if (t + 1) % every == 0 or t == forcing.days - 1:
                v = (field.state.z if field.state.z is not None
                     else _day_feats(model, field.state, fd, action))
                lib.append(v)
