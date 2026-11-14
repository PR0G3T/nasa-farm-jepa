"""Field state — the world model's state vector.

Batch semantics: scalar channels are (B,) tensors, soil water is (B, L) over
the profile's layers, latents are (B, K) / (B, H). B carries fields × plans ×
scenarios simultaneously, so rollouts never need a Python-level loop over
realizations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.state.soil import SoilProfile


@dataclass(frozen=True, slots=True)
class FieldState:
    """Mutable-in-time state. float64 except `crop`/`growing`/`harvested`."""

    sw: torch.Tensor           # (B, L) mm — soil water per layer
    n_min: torch.Tensor        # kg N/ha — mineral N pool
    n_org: torch.Tensor        # kg N/ha — slow organic N pool
    litter: torch.Tensor       # kg DM/ha — surface residue pool
    soc_f: torch.Tensor        # kg C/ha — active/fast SOC pool (~yr turnover)
    soc_s: torch.Tensor        # kg C/ha — stable/slow SOC pool (~decadal)
    et: torch.Tensor           # mm — actual evapotranspiration today
    n2o: torch.Tensor          # kg N-N2O/ha — cumulative emitted
    tws: torch.Tensor          # mm — aquifer/total-water-storage anomaly store
    gdd: torch.Tensor          # °C·d — accumulated since sowing
    lai: torch.Tensor          # m²/m² — leaf area index
    biomass: torch.Tensor      # kg DM/ha — above-ground
    season_yield: torch.Tensor  # kg/ha — accumulated harvested yield this season
    cum_yield: torch.Tensor    # kg/ha — all harvested yield since sim start
    crop: torch.Tensor         # long — index into CropLibrary, −1 = bare
    growing: torch.Tensor      # bool — crop in the ground
    harvested: torch.Tensor    # bool — current crop already terminated
    vern: torch.Tensor         # d — chilling days accumulated this crop cycle
    heat_acc: torch.Tensor     # °C·d — reproductive heat load this crop cycle
    z: torch.Tensor | None = None   # (B, K) free latent dims; None = off
    zh: torch.Tensor | None = None  # (B, H) recurrent latent hidden state
    depth: tuple[float, ...] = ()   # m — layer thicknesses (static metadata)
    tair: torch.Tensor | None = None  # °C — last day's mean air temperature

    @property
    def batch(self) -> int:
        return self.sw.shape[0]

    @property
    def n_layers(self) -> int:
        return self.sw.shape[1]

    def _map(self, fn) -> "FieldState":
        """Apply fn to every tensor field, preserving static metadata."""
        return replace(self, **{
            f: (fn(v) if isinstance(v, torch.Tensor) else v)
            for f in self.__dataclass_fields__
            if (v := getattr(self, f)) is not None})

    def replace(self, **kw) -> "FieldState":
        return replace(self, **kw)

    def clone(self) -> "FieldState":
        return self._map(lambda v: v.clone())

    def repeat(self, n: int) -> "FieldState":
        return self._map(
            lambda v: v.repeat(n, *([1] * (v.dim() - 1))))

    def to(self, device: torch.device | str) -> "FieldState":
        return self._map(lambda v: v.to(device))


@dataclass(frozen=True, slots=True)
class Field:
    """A field = state + static soil + location + static context vector."""

    state: FieldState
    soil: SoilProfile
    lat: torch.Tensor               # (B,) degrees
    elev: torch.Tensor | None = None  # (B,) meters; None → 200 m assumed
    ctx: torch.Tensor | None = None   # (B, C) static site embedding

    @property
    def batch(self) -> int:
        return self.state.batch

    def elev_or_default(self) -> torch.Tensor:
        if self.elev is not None:
            return self.elev
        return torch.full((self.batch,), 200.0, dtype=self.lat.dtype,
                          device=self.lat.device)

    def repeat(self, n: int) -> "Field":
        return Field(
            state=self.state.repeat(n),
            soil=self.soil.repeat(n),
            lat=self.lat.repeat(n),
            elev=self.elev.repeat(n) if self.elev is not None else None,
            ctx=self.ctx.repeat(n, 1) if self.ctx is not None else None,
        )

    def to(self, device: torch.device | str) -> "Field":
        return Field(
            state=self.state.to(device),
            soil=self.soil.to(device),
            lat=self.lat.to(device),
            elev=self.elev.to(device) if self.elev is not None else None,
            ctx=self.ctx.to(device) if self.ctx is not None else None,
        )


def initial_state(
    batch: int,
    soil: SoilProfile,
    n_min0: float = 40.0,
    n_org0: float = 4500.0,
    soc0: float | None = None,
    litter0: float = 500.0,
    fill_frac: float = 0.85,
    z_dim: int = 0,
    h_dim: int = 0,
) -> FieldState:
    """FieldState at simulation start, soils filled to `fill_frac` of
    available water. SOC splits 15 % active / 85 % stable (typical
    temperate-cropland pool ratio)."""
    def t(v): return torch.full((batch,), float(v), dtype=DTYPE)

    sw = (soil.wp + fill_frac * (soil.fc - soil.wp)).expand(batch, -1).clone()
    if soc0 is None:
        soc_tot = (14_000.0 * soil.om[:, 0]).expand(batch)
    else:
        soc_tot = torch.full((batch,), soc0, dtype=DTYPE)
    return FieldState(
        sw=sw,
        n_min=t(n_min0), n_org=t(n_org0), litter=t(litter0),
        soc_f=soc_tot * 0.15, soc_s=soc_tot * 0.85,
        et=t(0.0), n2o=t(0.0), tws=t(0.0),
        gdd=t(0.0), lai=t(0.0), biomass=t(0.0),
        season_yield=t(0.0), cum_yield=t(0.0),
        crop=torch.full((batch,), -1, dtype=torch.long),
        growing=torch.zeros(batch, dtype=torch.bool),
        harvested=torch.zeros(batch, dtype=torch.bool),
        vern=t(0.0), heat_acc=t(0.0),
        z=torch.zeros(batch, z_dim, dtype=DTYPE) if z_dim else None,
        zh=torch.zeros(batch, h_dim, dtype=DTYPE) if h_dim else None,
        depth=tuple(float(d) for d in soil.depth),
        tair=t(12.0),
    )
