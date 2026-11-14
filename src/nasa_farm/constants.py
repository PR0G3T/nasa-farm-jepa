"""Physical constants — single source of truth for all modules."""

from __future__ import annotations

import torch

DTYPE = torch.float64

SOLAR_CONSTANT_MJ_M2_MIN = 0.0820
"""FAO-56 solar constant G_sc (MJ m^-2 min^-1)."""

LATENT_HEAT_VAPORIZATION = 2.45
"""Latent heat of vaporization λ (MJ kg^-1), FAO-56."""

DAYS_PER_YEAR = 365
"""Simulation year length. Feb 29 is dropped when preparing forcing."""

# Soil layer geometry (m). Layer 1: evaporation + shallow roots.
# Layer 2 spans to 1.8 m — the WOFOST/DSSAT max rooting depth for field crops,
# which is what lets rainfed Midwest soils sustain summer dry spells.
LAYER1_DEPTH_M = 0.30
LAYER2_DEPTH_M = 1.50
