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


class CtxNorm(nn.Module):
    """Standardizer for raw site embeddings; unfit = pass-through."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("mu", torch.zeros(dim, dtype=DTYPE))
        self.register_buffer("sd", torch.ones(dim, dtype=DTYPE))
        self.fitted = False

    @torch.no_grad()
    def fit(self, emb: torch.Tensor) -> "CtxNorm":
        e = emb.reshape(-1, emb.shape[-1]).to(DTYPE)
        if e.shape[-1] != self.dim:
            raise ValueError(f"embedding dim {e.shape[-1]} != ctx dim {self.dim}")
        self.mu.copy_(e.mean(0))
        self.sd.copy_(e.std(0).clamp(min=1e-6))
        self.fitted = True
        return self

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return (emb.to(DTYPE) - self.mu) / self.sd


def build_ctx(emb: torch.Tensor, norm: CtxNorm | None = None,
              side: torch.Tensor | None = None) -> torch.Tensor:
    """(B, D) raw embedding (+ optional (B, S) scaled side features) → ctx.

    Side features are the caller's responsibility to scale (soil scalars,
    lat/90, climatology summaries); embeddings are standardized by `norm`.
    """
    z = norm(emb) if norm is not None else emb.to(DTYPE)
    if side is not None:
        z = torch.cat([z, side.to(DTYPE)], dim=-1)
    return torch.nan_to_num(z)


class TraitHyper(nn.Module):
    """ctx → log-multipliers on the crop trait table (regionalization).

    Output is tanh-bounded to ±log(bound), so the predicted table stays in
    [base/bound, base·bound] — the same plausibility band variational
    calibration uses. Bias zero-init ⇒ multiplier 1 at init (prior table).
    """

    def __init__(self, ctx_dim: int, n_crops: int, n_traits: int,
                 hidden: int = 64, bound: float = 2.0) -> None:
        super().__init__()
        self.bound = math.log(bound)
        self.shape = (n_crops, n_traits)
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, n_crops * n_traits))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.to(DTYPE)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """(B, C_ctx) → (B, C, P) trait multipliers in [1/bound, bound]."""
        d = torch.tanh(self.net(ctx.to(DTYPE))) * self.bound
        return torch.exp(d.view(ctx.shape[0], *self.shape))

    def table(self, ctx: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Contextualized tables, one (C, P) per batch row."""
        return base.unsqueeze(0) * self(ctx)

    def apply(self, model, ctx: torch.Tensor) -> torch.Tensor:
        """Single-field path: live-binds the predicted table to the model."""
        tbl = self.table(ctx[:1], model.crop_table.detach())[0]
        model.crop_table = tbl
        return tbl


class LatentInit(nn.Module):
    """ctx → (z0, zh0) — informed recurrent start instead of zeros."""

    def __init__(self, ctx_dim: int, z_dim: int, h_dim: int,
                 hidden: int = 64) -> None:
        super().__init__()
        self.z_dim, self.h_dim = z_dim, h_dim
        self.z = nn.Linear(ctx_dim, z_dim)
        self.h = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.SiLU(),
                               nn.Linear(hidden, h_dim))
        self.to(DTYPE)

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.z(ctx.to(DTYPE)), self.h(ctx.to(DTYPE))


def condition(model, field: Field, ctx: torch.Tensor,
              hyper: TraitHyper | None = None,
              init: LatentInit | None = None) -> Field:
    """Wire a context vector into field + model in one call.

    - field.ctx := ctx (consumed by LatentDyn when built with ctx_dim)
    - hyper: model.crop_table := predicted table
    - init: state.z/zh := predicted initial latents
    """
    if field.batch != 1 and ctx.shape[0] == 1:
        ctx = ctx.expand(field.batch, -1)
    if hyper is not None:
        tbl = hyper.table(ctx, model.crop_table.detach())
        model.crop_table = (tbl[0] if field.batch == 1
                            else tbl.mean(0))
    if init is not None:
        z0, h0 = init(ctx)
        field = Field(state=field.state.replace(z=z0, zh=h0),
                      soil=field.soil, lat=field.lat, elev=field.elev,
                      ctx=ctx)
        return field
    return Field(state=field.state, soil=field.soil, lat=field.lat,
                 elev=field.elev, ctx=ctx)


def lib_table(crops: CropLibrary) -> torch.Tensor:
    """Detachable base table — the hypernet's reference."""
    return crops.table.detach().clone()
