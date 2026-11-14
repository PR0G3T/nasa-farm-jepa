"""L-band radiometer operator (SMAP): zero-order τ-ω model

TB = Ts·[(1−ω)(1−γ) + γ(1−r) + r·γ·(1−ω)(1−γ)]
γ = exp(−b·VWC/cosθ), r from Fresnel reflectivities on a Wang–Schmugge-style
dielectric mv → ε.
"""

from __future__ import annotations

import math

import torch


def _eps(mv: torch.Tensor) -> torch.Tensor:
    """Soil dielectric constant — linearized Wang–Schmugge (valid 0–0.5)."""
    return 2.9 + 55.0 * mv


def tb(mv: torch.Tensor, vwc: torch.Tensor, ts: torch.Tensor | float = 298.15,
       theta_deg: float = 40.0, omega: float = 0.06,
       b: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """(TB_h, TB_v) in kelvin. mv: m3/m3, vwc: kg/m2, ts: soil temp K."""
    th = math.radians(theta_deg)
    cos, sin2 = math.cos(th), math.sin(th) ** 2
    eps = _eps(mv)
    sqr = torch.sqrt(eps - sin2)
    rh = ((cos - sqr) / (cos + sqr)) ** 2
    rv = ((eps * cos - sqr) / (eps * cos + sqr)) ** 2
    gam = torch.exp(-b * vwc / cos)
    out = []
    for r in (rh, rv):
        out.append(ts * ((1.0 - omega) * (1.0 - gam)
                         + gam * (1.0 - r)
                         + r * gam * (1.0 - omega) * (1.0 - gam)))
    return tuple(out)
