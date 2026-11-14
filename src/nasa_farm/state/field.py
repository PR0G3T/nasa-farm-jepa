"""Field state — the world model's state vector.

Every field of `FieldState` is a (B,) tensor: B is the batch dimension that
carries fields × plans × scenarios simultaneously, so rollouts never need a
Python-level loop over realizations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.state.soil import SoilProfile


@dataclass(frozen=True, slots=True)
class FieldState:
    """Mutable-in-time state. (B,) tensors, float64 except `crop` (long)."""

    sw1: torch.Tensor        # mm — soil water, layer 1
    sw2: torch.Tensor        # mm — soil water, layer 2
    n_min: torch.Tensor      # kg N/ha — mineral N pool
    n_org: torch.Tensor      # kg N/ha — slow organic N pool
    soc: torch.Tensor        # kg C/ha — soil organic carbon pool
    litter: torch.Tensor     # kg DM/ha — surface residue pool
    gdd: torch.Tensor        # °C·d — accumulated since sowing
    lai: torch.Tensor        # m²/m² — leaf area index
    biomass: torch.Tensor    # kg DM/ha — above-ground
    season_yield: torch.Tensor  # kg/ha — accumulated harvested yield this season
    cum_yield: torch.Tensor  # kg/ha — all harvested yield since sim start
    crop: torch.Tensor       # long — index into CropLibrary, −1 = bare
    growing: torch.Tensor    # bool — crop in the ground
    harvested: torch.Tensor  # bool — current crop already terminated

    @property
    def batch(self) -> int:
        return self.sw1.shape[0]

    def replace(self, **kw) -> "FieldState":
        return replace(self, **kw)

    def clone(self) -> "FieldState":
        return replace(self, **{f: getattr(self, f).clone() for f in self.__dataclass_fields__})

    def repeat(self, n: int) -> "FieldState":
        return replace(
            self,
            **{f: getattr(self, f).repeat(n) for f in self.__dataclass_fields__},
        )

    def to(self, device: torch.device | str) -> "FieldState":
        return replace(
            self,
            **{f: getattr(self, f).to(device) for f in self.__dataclass_fields__},
        )


@dataclass(frozen=True, slots=True)
class Field:
    """A field = state + static soil + location. The world model's full input."""

    state: FieldState
    soil: SoilProfile
    lat: torch.Tensor               # (B,) degrees
    elev: torch.Tensor | None = None  # (B,) meters; None → 200 m assumed

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
            soil=_repeat_soil(self.soil, n),
            lat=self.lat.repeat(n),
            elev=self.elev.repeat(n) if self.elev is not None else None,
        )

    def to(self, device: torch.device | str) -> "Field":
        return Field(
            state=self.state.to(device),
            soil=self.soil.to(device),
            lat=self.lat.to(device),
            elev=self.elev.to(device) if self.elev is not None else None,
        )


def _repeat_soil(soil: SoilProfile, n: int) -> SoilProfile:
    from nasa_farm.state.soil import SoilLayer

    def r(t): return t.repeat(n)
    l1 = SoilLayer(r(soil.layer1.theta_sat), r(soil.layer1.theta_fc),
                   r(soil.layer1.theta_wp), r(soil.layer1.theta_air))
    l2 = SoilLayer(r(soil.layer2.theta_sat), r(soil.layer2.theta_fc),
                   r(soil.layer2.theta_wp), r(soil.layer2.theta_air))
    return SoilProfile(l1, l2, r(soil.sand1), r(soil.clay1), r(soil.om1),
                       r(soil.sand2), r(soil.clay2), r(soil.om2))


def initial_state(
    batch: int,
    soil: SoilProfile,
    n_min0: float = 40.0,
    n_org0: float = 4500.0,
    soc0: float | None = None,
    litter0: float = 500.0,
    fill_frac: float = 0.85,
) -> FieldState:
    """FieldState at simulation start, soils filled to `fill_frac` of available water."""
    def t(v): return torch.full((batch,), float(v), dtype=DTYPE)

    avail1 = soil.layer1.theta_wp + fill_frac * (soil.layer1.theta_fc - soil.layer1.theta_wp)
    avail2 = soil.layer2.theta_wp + fill_frac * (soil.layer2.theta_fc - soil.layer2.theta_wp)
    if soc0 is None:
        soc0_v = 14_000.0 * soil.om1  # ≈ OM × C fraction, topsoil-scaled
    else:
        soc0_v = torch.full((batch,), soc0, dtype=DTYPE)
    return FieldState(
        sw1=avail1, sw2=avail2,
        n_min=t(n_min0), n_org=t(n_org0), soc=soc0_v, litter=t(litter0),
        gdd=t(0.0), lai=t(0.0), biomass=t(0.0),
        season_yield=t(0.0), cum_yield=t(0.0),
        crop=torch.full((batch,), -1, dtype=torch.long),
        growing=torch.zeros(batch, dtype=torch.bool),
        harvested=torch.zeros(batch, dtype=torch.bool),
    )
