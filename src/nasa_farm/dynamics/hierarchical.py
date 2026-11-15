"""Hierarchical multi-stride latent dynamics (H-JEPA).

Level l commits a latent update every `strides[l]` days and conditions
the level below it (top-down context), so slow modes — rotation-scale
structure, cultivar identity, soil memory — steer fast daily dynamics
without re-running their recurrent state. All levels share the obs
embedding space: `hjepa_loss` holds each level's committed latent against
the EMA-encoded observation `strides[l]` days ahead.

State stays a flat (B, L·K) / (B, L·H) concat so `FieldState.z`/`zh` and
`initial_state(z_dim=latent.dim)` work unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from nasa_farm.dynamics.latent import Dynamics


class Hierarchy(nn.Module):
    """Stack of `LatentDyn` levels sharing one embedding space.

    Zero-init heads → every level is the identity at construction, so a
    hierarchical model is exactly the physics-anchored baseline until
    trained. The flat z packs levels contiguously: level i occupies
    z[:, i·K:(i+1)·K] and zh[:, i·H:(i+1)·H].
    """

    def __init__(
        self,
        dim: int,
        hidden: int = 64,
        strides: tuple[int, ...] = (1, 4, 16),
        stochastic: bool = False,
        context_dimension: int = 0,
    ) -> None:
        super().__init__()
        self.k_dim = dim
        self.strides = tuple(int(s) for s in strides)
        # +dim ctx slots carry the level above as top-down context
        self.levels = nn.ModuleList(
            Dynamics(dim, hidden, stochastic, context_dimension + dim) for _ in self.strides
        )
        self.dim = dim * len(self.strides)
        self.hidden_dimension = hidden * len(self.strides)

    @property
    def stochastic(self) -> bool:
        return self.levels[0].stochastic

    @property
    def kl(self) -> torch.Tensor:
        return sum(lvl.kl for lvl in self.levels)

    def seed(self, z: torch.Tensor) -> torch.Tensor:
        """Obs embedding (B, K) → full latent (B, L·K): level 0 seeded,
        upper levels start at zero and commit on their own cadence."""
        return torch.cat([z] + [torch.zeros_like(z)] * (len(self.strides) - 1), dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        z: torch.Tensor,
        zh: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        generation: torch.Generator | None = None,
        t: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, k, h = len(self.strides), self.k_dim, self.levels[0].hidden_dimension
        z_parts = list(z.chunk(n, dim=-1))
        zh_parts = list((zh if zh is not None else z.new_zeros(z.shape[0], n * h)).chunk(n, dim=-1))
        context = context if context is not None else features[:, :0]
        z_new, zh_new = list(z_parts), list(zh_parts)
        for i in range(n - 1, -1, -1):  # top first: lower levels
            if t % self.strides[i] != 0:  # see the fresh upper latent
                continue
            up = z_new[i + 1].detach() if i + 1 < n else torch.zeros_like(z)
            lvl = self.levels[i]
            lvl_ctx = torch.cat([context.to(features.dtype), up[:, :k]], dim=-1)
            z_new[i], zh_new[i] = lvl(features, z_parts[i], zh_parts[i], lvl_ctx, generation)
        return torch.cat(z_new, dim=-1), torch.cat(zh_new, dim=-1)


def hierarchical_loss(
    z_traj: torch.Tensor, z_target: torch.Tensor, hlatent: Hierarchy, every: int = 1
) -> torch.Tensor:
    """Per-level prediction loss: level l's committed latent predicts the
    target embedding `strides[l]` days ahead. z_traj (B, N, L·K) online
    predictions, z_tgt (B, N, K) EMA-encoded obs at the trace cadence;
    index 0 is the seed and is excluded."""
    k, total = hlatent.k_dim, z_traj.new_zeros(())
    for i, s in enumerate(hlatent.strides):
        sh = max(round(s / every), 1)
        z_l = z_traj[:, 1:, i * k : (i + 1) * k]
        target = torch.cat([z_target[:, 1:], z_target[:, -1:].expand(-1, sh, -1)], dim=1)
        total = total + (z_l - target[:, : z_l.shape[1]]).pow(2).mean()
    return total / len(hlatent.strides)
