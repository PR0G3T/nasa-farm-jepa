"""Soil profile and pedotransfer functions (Saxton & Rawls 2006 style).

All hydraulic properties are volumetric water contents (m^3/m^3) converted
to storage in mm over the layer depth where needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nasa_farm.constants import DTYPE, LAYER1_DEPTH_M, LAYER2_DEPTH_M


def theta_wp(sand_pct: torch.Tensor, clay_pct: torch.Tensor, om_pct: torch.Tensor) -> torch.Tensor:
    """Wilting-point volumetric water content, Saxton & Rawls regression."""
    s, c, om = sand_pct / 100.0, clay_pct / 100.0, om_pct
    return torch.clamp(
        -0.024 * s + 0.487 * c + 0.006 * om + 0.005 * s * om
        - 0.013 * c * om + 0.068 * s * c + 0.031,
        min=0.01,
    )


def theta_fc(sand_pct: torch.Tensor, clay_pct: torch.Tensor, om_pct: torch.Tensor) -> torch.Tensor:
    """Field-capacity volumetric water content."""
    s, c, om = sand_pct / 100.0, clay_pct / 100.0, om_pct
    return torch.clamp(
        -0.251 * s + 0.195 * c + 0.011 * om + 0.006 * s * om
        - 0.027 * c * om + 0.452 * s * c + 0.299,
        min=0.05,
    )


def theta_sat(sand_pct: torch.Tensor, clay_pct: torch.Tensor) -> torch.Tensor:
    """Saturation (porosity proxy), Saxton et al. 1986 texture-only form."""
    s_pct, c_pct = sand_pct.clamp(1.0, 99.0), clay_pct.clamp(1.0, 99.0)
    return torch.clamp(0.332 - 7.251e-4 * s_pct + 0.1276 * torch.log10(c_pct), min=0.25)


@dataclass(frozen=True, slots=True)
class SoilLayer:
    """Hydraulic parameters of one layer; storages in mm."""

    theta_sat: torch.Tensor  # (B,) mm
    theta_fc: torch.Tensor   # (B,) mm
    theta_wp: torch.Tensor   # (B,) mm
    theta_air: torch.Tensor  # (B,) mm — air-dry lower bound for evaporation

    @property
    def batch(self) -> int:
        return self.theta_sat.shape[0]


@dataclass(frozen=True, slots=True)
class SoilProfile:
    """Two-layer soil. All tensors batched (B,)."""

    layer1: SoilLayer
    layer2: SoilLayer
    sand1: torch.Tensor
    clay1: torch.Tensor
    om1: torch.Tensor
    sand2: torch.Tensor
    clay2: torch.Tensor
    om2: torch.Tensor

    @property
    def batch(self) -> int:
        return self.layer1.batch

    @classmethod
    def from_texture(
        cls,
        sand_pct: torch.Tensor,
        clay_pct: torch.Tensor,
        om_pct: torch.Tensor | None = None,
        depth1_m: float = LAYER1_DEPTH_M,
        depth2_m: float = LAYER2_DEPTH_M,
    ) -> "SoilProfile":
        """Same texture both layers. Percents; om defaults to 1.5 %."""
        if om_pct is None:
            om_pct = torch.full_like(sand_pct, 1.5)

        def layer(s, c, om, depth_m):
            sat = theta_sat(s, c) * depth_m * 1000.0
            fc = theta_fc(s, c, om) * depth_m * 1000.0
            wp = theta_wp(s, c, om) * depth_m * 1000.0
            air = wp * 0.35  # air-dry ≈ 0.3–0.4 × wilting point
            sat = torch.clamp(sat, min=fc + 5.0)
            fc = torch.clamp(fc, min=wp + 2.0)
            return SoilLayer(sat, fc, wp, air)

        return cls(
            layer1=layer(sand_pct, clay_pct, om_pct, depth1_m),
            layer2=layer(sand_pct, clay_pct, om_pct, depth2_m),
            sand1=sand_pct, clay1=clay_pct, om1=om_pct,
            sand2=sand_pct, clay2=clay_pct, om2=om_pct,
        )

    @classmethod
    def preset(cls, texture: str, batch: int = 1) -> "SoilProfile":
        table = {
            "sand": (92.0, 3.0, 1.0),
            "loamy_sand": (82.0, 8.0, 1.2),
            "sandy_loam": (65.0, 15.0, 1.5),
            "loam": (42.0, 22.0, 2.0),
            "silt_loam": (20.0, 20.0, 2.5),
            "clay_loam": (32.0, 35.0, 2.2),
            "clay": (22.0, 50.0, 2.0),
        }
        s, c, om = (torch.tensor([v] * batch, dtype=DTYPE) for v in table[texture])
        return cls.from_texture(s, c, om)

    def to(self, device: torch.device) -> "SoilProfile":
        def mv(t): return t.to(device)
        l1 = SoilLayer(mv(self.layer1.theta_sat), mv(self.layer1.theta_fc),
                       mv(self.layer1.theta_wp), mv(self.layer1.theta_air))
        l2 = SoilLayer(mv(self.layer2.theta_sat), mv(self.layer2.theta_fc),
                       mv(self.layer2.theta_wp), mv(self.layer2.theta_air))
        return SoilProfile(l1, l2, mv(self.sand1), mv(self.clay1), mv(self.om1),
                           mv(self.sand2), mv(self.clay2), mv(self.om2))
