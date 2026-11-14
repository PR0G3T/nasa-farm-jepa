"""Site context conditioning — EO foundation embeddings → ctx → traits/z0.

A static per-field vector `ctx` is the world model's site memory. This
module owns its whole lifecycle:

- `CtxNorm`: streaming mean/std over embedding batches (AlphaEarth 64-d,
  Prithvi, TerraMind, or any provider — the contract is a plain tensor),
  optionally stacked with soil/climate scalars.
- `build_ctx`: normalized embedding + optional side features → (B, C)
  attachable to `Field.ctx`.
- `TraitHyper`: ctx → bounded per-crop trait multipliers — regionalization:
  site traits are predicted from context instead of refit per site.
- `LatentInit`: ctx → (z0, zh0) — the recurrent dynamics starts informed
  rather than at zero.
- `condition`: one call wires ctx, trait table and latent init together.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nasa_farm.constants import DTYPE
from nasa_farm.state.crops import CropLibrary
from nasa_farm.state.field import Field


class Normalizer(nn.Module):
    """Standardizer for raw site embeddings; unfit = pass-through."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("mean", torch.zeros(dim, dtype=DTYPE))
        self.register_buffer("deviation", torch.ones(dim, dtype=DTYPE))
        self.fitted = False

    @torch.no_grad()
    def fit(self, embedding: torch.Tensor) -> "Normalizer":
        flat = embedding.reshape(-1, embedding.shape[-1]).to(DTYPE)
        if flat.shape[-1] != self.dim:
            raise ValueError(f"embedding dim {flat.shape[-1]} != ctx dim {self.dim}")
        self.mean.copy_(flat.mean(0))
        self.deviation.copy_(flat.std(0).clamp(min=1e-6))
        self.fitted = True
        return self

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return (embedding.to(DTYPE) - self.mean) / self.deviation


def build_context(embedding: torch.Tensor, norm: Normalizer | None = None,
              side: torch.Tensor | None = None) -> torch.Tensor:
    """(B, D) raw embedding (+ optional (B, S) scaled side features) → ctx.

    Side features are the caller's responsibility to scale (soil scalars,
    lat/90, climatology summaries); embeddings are standardized by `norm`.
    """
    z = norm(embedding) if norm is not None else embedding.to(DTYPE)
    if side is not None:
        z = torch.cat([z, side.to(DTYPE)], dim=-1)
    return torch.nan_to_num(z)


class Traits(nn.Module):
    """ctx → log-multipliers on the crop trait table (regionalization).

    Output is tanh-bounded to ±log(bound), so the predicted table stays in
    [base/bound, base·bound] — the same plausibility band variational
    calibration uses. Bias zero-init ⇒ multiplier 1 at init (prior table).
    """

    def __init__(self, context_dimension: int, n_crops: int, n_traits: int,
                 hidden: int = 64, bound: float = 2.0) -> None:
        super().__init__()
        self.bound = math.log(bound)
        self.shape = (n_crops, n_traits)
        self.net = nn.Sequential(
            nn.Linear(context_dimension, hidden), nn.SiLU(),
            nn.Linear(hidden, n_crops * n_traits))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.to(DTYPE)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """(B, C_ctx) → (B, C, P) trait multipliers in [1/bound, bound]."""
        d = torch.tanh(self.net(context.to(DTYPE))) * self.bound
        return torch.exp(d.view(context.shape[0], *self.shape))

    def table(self, context: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Contextualized tables, one (C, P) per batch row."""
        return base.unsqueeze(0) * self(context)

    def apply(self, model, context: torch.Tensor) -> torch.Tensor:
        """Single-field path: live-binds the predicted table to the model."""
        table = self.table(context[:1], model.crop_table.detach())[0]
        model.crop_table = table
        return table


class Initialization(nn.Module):
    """ctx → (z0, zh0) — informed recurrent start instead of zeros."""

    def __init__(self, context_dimension: int, z_dim: int, hidden_dimension: int,
                 hidden: int = 64) -> None:
        super().__init__()
        self.z_dim, self.hidden_dimension = z_dim, hidden_dimension
        self.z = nn.Linear(context_dimension, z_dim)
        self.h = nn.Sequential(nn.Linear(context_dimension, hidden), nn.SiLU(),
                               nn.Linear(hidden, hidden_dimension))
        self.to(DTYPE)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.z(context.to(DTYPE)), self.h(context.to(DTYPE))


def condition(model, field: Field, context: torch.Tensor,
              hyper: Traits | None = None,
              initialization: Initialization | None = None) -> Field:
    """Wire a context vector into field + model in one call.

    - field.ctx := ctx (consumed by LatentDyn when built with ctx_dim)
    - hyper: model.crop_table := predicted table
    - init: state.z/zh := predicted initial latents
    """
    if field.batch != 1 and context.shape[0] == 1:
        context = context.expand(field.batch, -1)
    if hyper is not None:
        table = hyper.table(context, model.crop_table.detach())
        model.crop_table = (table[0] if field.batch == 1
                            else table.mean(0))
    if initialization is not None:
        z0, h0 = initialization(context)
        field = Field(state=field.state.replace(z=z0, zh=h0),
                      soil=field.soil, latitude=field.latitude, elevation=field.elevation,
                      context=context)
        return field
    return Field(state=field.state, soil=field.soil, latitude=field.latitude,
                 elevation=field.elevation, context=context)


def lib_table(crops: CropLibrary) -> torch.Tensor:
    """Detachable base table — the hypernet's reference."""
    return crops.table.detach().clone()
