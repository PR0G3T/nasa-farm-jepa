"""Differentiable observation operators: model state → satellite-observable
quantities (HLS/Sentinel-2 optical, NISAR/Sentinel-1 SAR, SMAP radiometer,
OpenET-style ET, GRACE-style total water storage, land-surface temperature).

Each channel maps FieldState tensors to the physical observable, so
predicted-observation residuals enter variational assimilation and the JEPA
encoder target with full gradient flow.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from nasa_farm.observe import optical, radiometer, sar
from nasa_farm.observe.detect import infer_actions
from nasa_farm.state.field import FieldState


def sm_vol(state: FieldState, i: int = 0) -> torch.Tensor:
    """Layer-i volumetric water content m3/m3."""
    return state.sw[:, i] / (state.depth[i] * 1000.0)


def sm_col(state: FieldState) -> torch.Tensor:
    """Total column soil water storage (mm)."""
    return state.sw.sum(-1)


def vwc(state: FieldState) -> torch.Tensor:
    """Vegetation water content kg/m2 (≈ above-ground biomass kg/ha × 1e-4
    with ~60 % water fraction)."""
    return state.biomass * 6.0e-5


def lst(state: FieldState) -> torch.Tensor:
    """Land-surface temperature proxy (°C): air temperature plus a
    moisture-stress heating increment — dry surface heats toward ~+8 °C."""
    ta = state.tair if state.tair is not None else torch.full_like(state.et, 20.0)
    stress = 1.0 - torch.clamp(state.et / 6.0, 0.0, 1.0)
    return ta + 8.0 * stress


def tws(state: FieldState) -> torch.Tensor:
    """GRACE-style total water storage anomaly: column water + aquifer (mm)."""
    return state.sw.sum(-1) + state.tws


CHANNELS: dict[str, Callable[[FieldState], torch.Tensor]] = {
    "lai": lambda s: s.lai,
    "ndvi": lambda s: optical.ndvi(s.lai),
    "fapar": lambda s: optical.fapar(s.lai),
    "sm": sm_col,
    "sm1": lambda s: sm_vol(s, 0),
    "sm2": lambda s: sm_vol(s, 1),
    "vwc": vwc,
    "s0v": lambda s: sar.sigma0(sm_vol(s), vwc(s)),
    "s0c": lambda s: sar.sigma0(sm_vol(s), vwc(s), band="c"),
    "tb_v": lambda s: radiometer.tb(
        sm_vol(s), vwc(s), ts=s.tair + 273.15)[1],
    "tb_h": lambda s: radiometer.tb(
        sm_vol(s), vwc(s), ts=s.tair + 273.15)[0],
    "et": lambda s: s.et,
    "tws": tws,
    "lst": lst,
    "n2o": lambda s: s.n2o,
    "soc": lambda s: s.soc_f + s.soc_s,
    "soc_f": lambda s: s.soc_f,
    "yield": lambda s: s.cum_yield,
    "biomass": lambda s: s.biomass,
    "n_min": lambda s: s.n_min,
}


def observables(state: FieldState,
                names: tuple[str, ...] | None = None) -> torch.Tensor:
    """Stack requested channels → (B, C). Unknown name → KeyError early."""
    names = names or tuple(CHANNELS)
    return torch.stack([CHANNELS[n](state) for n in names], dim=-1)
