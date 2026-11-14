"""SAR operator (NISAR L-band / Sentinel-1 C-band): water-cloud model

σ⁰ = σ⁰_veg + γ² σ⁰_soil   with   σ⁰_veg = a·VWC·cosθ·(1 − γ²),
γ² = exp(−2b·VWC/cosθ),     σ⁰_soil = 10^((c + d·mv)/10).
"""

from __future__ import annotations

import math

import torch

BANDS: dict[str, dict[str, float]] = {
    # a, b: vegetation params; c + d·mv: soil linear model in dB
    "l": {"a": 1.2e-3, "b": 0.091, "c": -15.0, "d": 35.0},
    "c": {"a": 4.0e-3, "b": 0.100, "c": -12.0, "d": 40.0},
}


def sigma0(mv: torch.Tensor, vwc: torch.Tensor, theta_deg: float = 40.0,
           band: str = "l") -> torch.Tensor:
    """Backscatter in dB. mv: m3/m3 topsoil moisture, vwc: kg/m2."""
    p = BANDS[band]
    cos = math.cos(math.radians(theta_deg))
    g2 = torch.exp(-2.0 * p["b"] * vwc / cos)
    s_veg = p["a"] * vwc * cos * (1.0 - g2)
    s_soil = 10.0 ** ((p["c"] + p["d"] * mv) / 10.0)
    return 10.0 * torch.log10(s_veg + g2 * s_soil + 1e-12)
