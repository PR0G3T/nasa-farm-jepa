"""Optical operators (Sentinel-2 / HLS): LAI → fAPAR / NDVI via a Beer's-law
two-stream approximation."""

from __future__ import annotations

import torch


def fapar(lai: torch.Tensor, k: float = 0.5) -> torch.Tensor:
    """Fraction of absorbed PAR — Beer extinction through the canopy."""
    return 1.0 - torch.exp(-k * lai)


def ndvi(lai: torch.Tensor, soil_bg: float = 0.15, ndvi_inf: float = 0.92,
         k: float = 0.5) -> torch.Tensor:
    """Soil-adjusted NDVI: linear mix between bare-soil and closed-canopy
    reflectance weighted by canopy cover."""
    return soil_bg + (ndvi_inf - soil_bg) * fapar(lai, k)
