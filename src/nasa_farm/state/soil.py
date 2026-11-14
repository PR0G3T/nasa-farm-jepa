"""Soil profile and pedotransfer functions (Saxton & Rawls 2006 style).

N-layer representation: every hydraulic property is a (B, L) tensor of
storages in mm over each layer depth, so the water balance runs at any
vertical resolution with no per-layer Python state.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from nasa_farm.constants import DTYPE

DEPTHS: tuple[float, ...] = (0.30, 1.50)
"""Default layer thicknesses (m): evaporative/shallow-root zone then the
WOFOST/DSSAT ~1.8 m rooting depth sustaining rainfed summer dry spells."""


def theta_wp(sand_pct: torch.Tensor, clay_pct: torch.Tensor, om_pct: torch.Tensor) -> torch.Tensor:
    """Wilting-point volumetric water content, Saxton & Rawls regression."""
    s, c, organic = sand_pct / 100.0, clay_pct / 100.0, om_pct
    return torch.clamp(
        -0.024 * s + 0.487 * c + 0.006 * organic + 0.005 * s * organic
        - 0.013 * c * organic + 0.068 * s * c + 0.031,
        min=0.01,
    )


def theta_fc(sand_pct: torch.Tensor, clay_pct: torch.Tensor, om_pct: torch.Tensor) -> torch.Tensor:
    """Field-capacity volumetric water content."""
    s, c, organic = sand_pct / 100.0, clay_pct / 100.0, om_pct
    return torch.clamp(
        -0.251 * s + 0.195 * c + 0.011 * organic + 0.006 * s * organic
        - 0.027 * c * organic + 0.452 * s * c + 0.299,
        min=0.05,
    )


def theta_sat(sand_pct: torch.Tensor, clay_pct: torch.Tensor) -> torch.Tensor:
    """Saturation (porosity proxy), Saxton et al. 1986 texture-only form."""
    s_pct, c_pct = sand_pct.clamp(1.0, 99.0), clay_pct.clamp(1.0, 99.0)
    return torch.clamp(0.332 - 7.251e-4 * s_pct + 0.1276 * torch.log10(c_pct), min=0.25)


@dataclass(frozen=True, slots=True)
class SoilLayer:
    """Hydraulic parameters of one layer; storages in mm, (B,) tensors."""

    theta_sat: torch.Tensor
    theta_fc: torch.Tensor
    theta_wp: torch.Tensor
    theta_air: torch.Tensor  # air-dry lower bound for evaporation

    @property
    def batch(self) -> int:
        return self.theta_sat.shape[0]


def _stack(sand: torch.Tensor, clay: torch.Tensor, organic: torch.Tensor,
           depths: tuple[float, ...]) -> tuple[torch.Tensor, ...]:
    """Texture (B,) or (B,L) → hydraulic storages (B,L) mm per layer."""
    if sand.dim() == 1:
        sand = sand.unsqueeze(1).expand(-1, len(depths))
        clay = clay.unsqueeze(1).expand(-1, len(depths))
        organic = organic.unsqueeze(1).expand(-1, len(depths))
    d = torch.tensor(depths, dtype=sand.dtype, device=sand.device) * 1000.0
    sat = theta_sat(sand, clay) * d
    fc = theta_fc(sand, clay, organic) * d
    wp = theta_wp(sand, clay, organic) * d
    air = wp * 0.35  # air-dry ≈ 0.3–0.4 × wilting point
    sat = torch.maximum(sat, fc + 5.0)
    fc = torch.maximum(fc, wp + 2.0)
    return sand, clay, organic, sat, fc, wp, air


@dataclass(frozen=True, slots=True)
class SoilProfile:
    """N-layer soil. Textures and hydraulic storages are (B, L) tensors."""

    depth: torch.Tensor     # (L,) m
    sand: torch.Tensor      # (B, L) %
    clay: torch.Tensor      # (B, L) %
    organic: torch.Tensor        # (B, L) % organic matter
    sat: torch.Tensor       # (B, L) mm
    fc: torch.Tensor        # (B, L) mm
    wp: torch.Tensor        # (B, L) mm
    air: torch.Tensor       # (B, L) mm

    @property
    def batch(self) -> int:
        return self.sat.shape[0]

    @property
    def n_layers(self) -> int:
        return self.sat.shape[1]

    def layer(self, i: int) -> SoilLayer:
        return SoilLayer(self.sat[:, i], self.fc[:, i],
                         self.wp[:, i], self.air[:, i])

    @classmethod
    def from_texture(
        cls,
        sand_pct: torch.Tensor,
        clay_pct: torch.Tensor,
        om_pct: torch.Tensor | None = None,
        depths: tuple[float, ...] = DEPTHS,
    ) -> "SoilProfile":
        """Texture percents (B,) or (B,L); om defaults to 1.5 %."""
        if om_pct is None:
            om_pct = torch.full_like(sand_pct, 1.5)
        sand, clay, organic, sat, fc, wp, air = _stack(
            sand_pct, clay_pct, om_pct, depths)
        return cls(torch.tensor(depths, dtype=DTYPE),
                   sand.to(DTYPE), clay.to(DTYPE), organic.to(DTYPE),
                   sat.to(DTYPE), fc.to(DTYPE), wp.to(DTYPE), air.to(DTYPE))

    @classmethod
    def sample(cls, low: "SoilProfile", high: "SoilProfile", n: int,
               seed: int = 0) -> "SoilProfile":
        """Parameter-uncertainty ensemble: n profiles whose textures are
        uniform draws inside [lo, hi] per layer (e.g. SoilGrids Q05/Q95),
        hydraulics recomputed so every member stays self-consistent."""
        g = torch.Generator().manual_seed(seed)

        def u(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            r = torch.rand(n, *a.shape, generator=g, dtype=DTYPE)
            x = a.unsqueeze(0) + (b - a).unsqueeze(0) * r   # (n, B, L)
            return x.reshape(-1, a.shape[-1])

        depths = tuple(float(d) for d in low.depth)
        return cls.from_texture(
            u(low.sand, high.sand), u(low.clay, high.clay), u(low.organic, high.organic), depths)

    @classmethod
    def preset(cls, texture: str, batch: int = 1,
               depths: tuple[float, ...] = DEPTHS) -> "SoilProfile":
        table = {
            "sand": (92.0, 3.0, 1.0),
            "loamy_sand": (82.0, 8.0, 1.2),
            "sandy_loam": (65.0, 15.0, 1.5),
            "loam": (42.0, 22.0, 2.0),
            "silt_loam": (20.0, 20.0, 2.5),
            "clay_loam": (32.0, 35.0, 2.2),
            "clay": (22.0, 50.0, 2.0),
        }
        s, c, organic = (torch.tensor([v] * batch, dtype=DTYPE) for v in table[texture])
        return cls.from_texture(s, c, organic, depths)

    def to(self, device: torch.device | str) -> "SoilProfile":
        mv = lambda t: t.to(device)
        return SoilProfile(mv(self.depth), mv(self.sand), mv(self.clay),
                           mv(self.organic), mv(self.sat), mv(self.fc),
                           mv(self.wp), mv(self.air))

    def repeat(self, n: int) -> "SoilProfile":
        r = lambda t: t.repeat(n, *([1] * (t.dim() - 1)))
        return SoilProfile(self.depth, r(self.sand), r(self.clay), r(self.organic),
                           r(self.sat), r(self.fc), r(self.wp), r(self.air))


def root_weights(root_frac2: torch.Tensor, n_layers: int) -> torch.Tensor:
    """Per-layer uptake weight (B, L) — exponential root density profile.

    Layer 0 weight is 1−r; deeper layers take r·(1−r)^(i−1), which reduces
    exactly to the two-layer (1−r, r) split at L=2 and decays with depth
    the way root mass does.
    """
    i = torch.arange(n_layers, device=root_frac2.device,
                     dtype=root_frac2.dtype)
    w = torch.where(i == 0, 1.0 - root_frac2.unsqueeze(-1),
                    root_frac2.unsqueeze(-1)
                    * (1.0 - root_frac2.unsqueeze(-1)).clamp(min=0.0) ** (i - 1))
    return w
