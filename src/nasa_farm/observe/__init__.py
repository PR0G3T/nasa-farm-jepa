"""Differentiable observation operators: model state → satellite-observable
quantities (HLS/ Sentinel-2 optical, NISAR/Sentinel-1 SAR, SMAP radiometer).

Each channel maps FieldState tensors to the physical observable, so
predicted-observation residuals enter variational assimilation and the JEPA
encoder target with full gradient flow.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from nasa_farm.constants import LAYER1_DEPTH_M
from nasa_farm.observe import optical, radiometer, sar
from nasa_farm.state.field import FieldState


def sm_vol(state: FieldState) -> torch.Tensor:
    """Topsoil volumetric water content m3/m3."""
    return state.sw1 / (LAYER1_DEPTH_M * 1000.0)


def vwc(state: FieldState) -> torch.Tensor:
    """Vegetation water content kg/m2 (≈ above-ground biomass kg/ha × 1e-4
    with ~60 % water fraction)."""
    return state.biomass * 6.0e-5


CHANNELS: dict[str, Callable[[FieldState], torch.Tensor]] = {
    "lai": lambda s: s.lai,
    "ndvi": lambda s: optical.ndvi(s.lai),
    "fapar": lambda s: optical.fapar(s.lai),
    "sm1": sm_vol,
    "vwc": vwc,
    "s0v": lambda s: sar.sigma0(sm_vol(s), vwc(s)),
    "tb_v": lambda s: radiometer.tb(sm_vol(s), vwc(s))[1],
    "tb_h": lambda s: radiometer.tb(sm_vol(s), vwc(s))[0],
}


def observables(state: FieldState,
                names: tuple[str, ...] | None = None) -> torch.Tensor:
    """Stack requested channels → (B, C). Unknown name → KeyError early."""
    names = names or tuple(CHANNELS)
    return torch.stack([CHANNELS[n](state) for n in names], dim=-1)
