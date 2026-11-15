"""Time-indexed containers shared by dynamics, scenarios and planning.

Both are (B, T) — batch × day. `.day(t)` returns the per-day dict consumed by
`WorldModel.step`, keeping the model decoupled from storage layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nasa_farm.constants import DTYPE

FORCING_VARS: tuple[str, ...] = ("tmin", "tmax", "srad", "precip", "rh", "u2", "doy", "co2")

ACTION_VARS: tuple[str, ...] = ("sow", "irrigation", "n_fert", "till", "res_remove")

CO2_REF_PPM = 420.0
"""Reference atmospheric CO2 at which crop `co2_resp` traits are calibrated."""

type DayDict = dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class Forcing:
    """Daily exogenous drivers, (B, T).

    rh: mean relative humidity (%); u2: wind speed at 2 m (m/s) — both feed
    FAO-56 Penman–Monteith ET0. co2: atmospheric CO2 (ppm); None → CO2_REF_PPM,
    so climate-scenario forcings stay a one-knob override.
    """

    tmin: torch.Tensor
    tmax: torch.Tensor
    srad: torch.Tensor  # MJ m^-2 d^-1
    precip: torch.Tensor  # mm
    rh: torch.Tensor  # %
    u2: torch.Tensor  # m/s
    doy: torch.Tensor  # 1..365
    co2: torch.Tensor | None = None  # ppm

    def __post_init__(self) -> None:
        if self.co2 is None:
            object.__setattr__(self, "co2", torch.full_like(self.tmin, CO2_REF_PPM))
        elif not isinstance(self.co2, torch.Tensor):
            object.__setattr__(self, "co2", torch.full_like(self.tmin, float(self.co2)))

    @property
    def batch(self) -> int:
        return self.tmin.shape[0]

    @property
    def days(self) -> int:
        return self.tmin.shape[1]

    def day(self, t: int) -> DayDict:
        return {k: getattr(self, k)[:, t] for k in FORCING_VARS}

    def perturb(
        self, delta_t: float = 0.0, precip_scale: float = 1.0, srad_scale: float = 1.0
    ) -> Forcing:
        """Counterfactual climate shift — the 'promptable world event'."""
        return Forcing(
            tmin=self.tmin + delta_t,
            tmax=self.tmax + delta_t,
            srad=self.srad * srad_scale,
            precip=self.precip * precip_scale,
            rh=self.rh,
            u2=self.u2,
            doy=self.doy,
            co2=self.co2,
        )

    def repeat(self, n: int) -> Forcing:
        return Forcing(*(getattr(self, k).repeat(n, 1) for k in FORCING_VARS))

    def to(self, device: torch.device | str) -> Forcing:
        return Forcing(*(getattr(self, k).to(device) for k in FORCING_VARS))

    @classmethod
    def concat(cls, parts: list[Forcing]) -> Forcing:
        return cls(*(torch.cat([getattr(p, k) for p in parts], dim=0) for k in FORCING_VARS))


@dataclass(frozen=True, slots=True)
class Actions:
    """Daily management drivers, (B, T).

    sow: long, −1 = no sowing. till: tillage fraction of the surface layer
    mixed into the soil that day ∈ [0,1]. res_remove: fraction of surface
    residue exported (baling, grazing) ∈ [0,1].
    """

    sow: torch.Tensor  # (B, T) long
    irrigation: torch.Tensor  # (B, T) mm
    n_fert: torch.Tensor  # (B, T) kg N/ha
    till: torch.Tensor | None = None  # (B, T) ∈ [0,1]
    res_remove: torch.Tensor | None = None  # (B, T) ∈ [0,1]

    def __post_init__(self) -> None:
        for k in ("till", "res_remove"):
            if getattr(self, k) is None:
                object.__setattr__(self, k, torch.zeros_like(self.irrigation))

    @property
    def batch(self) -> int:
        return self.sow.shape[0]

    @property
    def days(self) -> int:
        return self.sow.shape[1]

    def day(self, t: int) -> DayDict:
        return {k: getattr(self, k)[:, t] for k in ACTION_VARS}

    @classmethod
    def idle(cls, batch: int, days: int) -> Actions:
        return cls(
            sow=torch.full((batch, days), -1, dtype=torch.long),
            irrigation=torch.zeros(batch, days, dtype=DTYPE),
            n_fert=torch.zeros(batch, days, dtype=DTYPE),
        )

    def repeat(self, n: int) -> Actions:
        return Actions(*(getattr(self, k).repeat(n, 1) for k in ACTION_VARS))

    def to(self, device: torch.device | str) -> Actions:
        return Actions(*(getattr(self, k).to(device) for k in ACTION_VARS))
