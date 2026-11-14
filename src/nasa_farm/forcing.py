"""Forcing ensembles — scenario construction for counterfactual rollouts.

A scenario is one (T,) weather realization. Ensembles batch scenarios along B
so a single rollout covers climatology, counterfactual shifts, and seeded
noise simultaneously.
"""

from __future__ import annotations

import math

import torch

from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.types import Forcing


def synthetic_year(
    lat_deg: float,
    tmean_annual: float = 11.0,
    tamp: float = 12.0,
    srad_mean: float = 15.0,
    precip_mean: float = 2.2,
    wet_prob: float = 0.32,
    co2_ppm: float = 420.0,
    seed: int = 0,
) -> Forcing:
    """Deterministic seasonal climatology + seeded daily noise. (B=1, T=365.)

    Used for tests and fully-offline operation — generated, never hard-coded
    per location.
    """
    g = torch.Generator().manual_seed(seed)
    lat = math.radians(lat_deg)
    doy = torch.arange(1, DAYS_PER_YEAR + 1, dtype=DTYPE)
    phase = 2.0 * math.pi * (doy - 200.0) / DAYS_PER_YEAR  # peak ~mid-July NH
    if lat < 0:
        phase = phase + math.pi

    tmean = tmean_annual + tamp * torch.cos(phase)
    diurnal = 6.0 + 2.0 * torch.cos(phase)
    tmin = tmean - diurnal / 2.0 + torch.randn(DAYS_PER_YEAR, generator=g) * 1.5
    tmax = tmean + diurnal / 2.0 + torch.randn(DAYS_PER_YEAR, generator=g) * 1.5
    tmax = torch.maximum(tmax, tmin)

    lat_amp = 1.0 + abs(lat_deg) / 45.0  # seasonal swing grows with latitude
    srad = srad_mean + (8.0 * lat_amp) * torch.cos(phase)
    srad = torch.clamp(srad + torch.randn(DAYS_PER_YEAR, generator=g) * 3.0, min=0.5)

    wet = torch.rand(DAYS_PER_YEAR, generator=g) < wet_prob
    precip = wet * torch.rand(DAYS_PER_YEAR, generator=g) * (precip_mean / max(wet_prob, 1e-3)) * 3.0

    # Humidity anticorrelates with temperature; wind has weak seasonal cycle.
    rh = torch.clamp(80.0 - 0.8 * (tmean - tmean_annual)
                     + torch.randn(DAYS_PER_YEAR, generator=g) * 4.0, 25.0, 98.0)
    u2 = torch.clamp(2.5 + 0.8 * torch.cos(phase)
                     + torch.randn(DAYS_PER_YEAR, generator=g) * 0.4, 0.3, 8.0)
    return Forcing(
        tmin.unsqueeze(0), tmax.unsqueeze(0),
        srad.unsqueeze(0), precip.unsqueeze(0),
        rh.unsqueeze(0), u2.unsqueeze(0), doy.unsqueeze(0),
        torch.full((1, DAYS_PER_YEAR), co2_ppm, dtype=DTYPE),
    )


def augment(f: Forcing, n: int, rho: float = 0.6,
            seed: int = 0) -> Forcing:
    """AR(1)-correlated weather perturbations → n extra members per base row.

    Returns the original batch followed by n perturbed members each, so a
    single deterministic forcing becomes an (B·(n+1), T) scenario ensemble —
    cheap probabilistic forcing for risk metrics and robust control.
    """
    from nasa_farm.control import ou_noise
    B, T = f.batch, f.days
    g = torch.Generator().manual_seed(seed)
    e = [ou_noise(B * n, T, rho, g) for _ in range(5)]

    def rep(v: torch.Tensor) -> torch.Tensor:
        return v.repeat_interleave(n, dim=0)

    tmin = rep(f.tmin) + 1.5 * e[0]
    tmax = torch.maximum(rep(f.tmax) + 1.5 * e[0], tmin + 0.5)
    return Forcing.concat([f, Forcing(
        tmin=tmin, tmax=tmax,
        srad=(rep(f.srad) * torch.exp(0.08 * e[1])).clamp(min=0.5),
        precip=rep(f.precip) * torch.exp(0.35 * e[2]),
        rh=(rep(f.rh) + 4.0 * e[3]).clamp(5.0, 100.0),
        u2=(rep(f.u2) * torch.exp(0.12 * e[4])).clamp(min=0.1),
        doy=rep(f.doy), co2=rep(f.co2),
    )])


def ensemble_of(years: list[Forcing]) -> Forcing:
    """Stack single-scenario forcings into one batch."""
    return Forcing.concat(years)


def tile_to_horizon(f: Forcing, years: int) -> Forcing:
    """Repeat a 365-day forcing across `years` simulation years."""
    from nasa_farm.types import FORCING_VARS
    return Forcing(*(getattr(f, k).repeat(1, years) for k in FORCING_VARS))


def climatology_ensemble(
    years: list[Forcing],
    horizon_years: int,
    delta_t: float = 0.0,
    precip_scale: float = 1.0,
) -> Forcing:
    """Each historical year becomes one scenario, tiled across the horizon and
    optionally shifted — 'drier/hotter season' counterfactuals for free."""
    return Forcing.concat([
        tile_to_horizon(y, horizon_years).perturb(delta_t=delta_t,
                                                  precip_scale=precip_scale)
        for y in years
    ])
