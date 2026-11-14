"""Atmospheric drivers: extraterrestrial radiation, ET0, thermal time.

Pure functions — no state, no mutation. All accept broadcastable tensors.
"""

from __future__ import annotations

import math

import torch

from nasa_farm.constants import DAYS_PER_YEAR, SOLAR_CONSTANT_MJ_M2_MIN


def extraterrestrial_radiation(lat_deg: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
    """Ra (MJ m^-2 d^-1), FAO-56 eqs 21–24. lat/doy broadcastable."""
    latitude = torch.deg2rad(lat_deg)
    dr = 1.0 + 0.033 * torch.cos(2.0 * math.pi * doy / DAYS_PER_YEAR)
    decl = 0.409 * torch.sin(2.0 * math.pi * doy / DAYS_PER_YEAR - 1.39)
    ws = torch.acos(torch.clamp(-torch.tan(latitude) * torch.tan(decl), -1.0, 1.0))
    return (
        (24.0 * 60.0 / math.pi)
        * SOLAR_CONSTANT_MJ_M2_MIN
        * dr
        * (ws * torch.sin(latitude) * torch.sin(decl)
           + torch.cos(latitude) * torch.cos(decl) * torch.sin(ws))
    )


STEFAN_BOLTZMANN = 4.903e-9
"""σ (MJ K^-4 m^-2 d^-1), FAO-56 eq 39."""


def saturation_vapor_pressure(tc: torch.Tensor) -> torch.Tensor:
    """e° (kPa), FAO-56 eq 11."""
    return 0.6108 * torch.exp(17.27 * tc / (tc + 237.3))


def svp_slope(tc: torch.Tensor) -> torch.Tensor:
    """Δ (kPa °C^-1), FAO-56 eq 13."""
    e = saturation_vapor_pressure(tc)
    return 4098.0 * e / (tc + 237.3) ** 2


def psychrometric_constant(elev_m: torch.Tensor) -> torch.Tensor:
    """γ (kPa °C^-1), FAO-56 eqs 7–8."""
    p = 101.3 * ((293.0 - 0.0065 * elev_m) / 293.0) ** 5.26
    return 0.665e-3 * p


def et0_penman_monteith(
    tmin: torch.Tensor,
    tmax: torch.Tensor,
    rh_mean: torch.Tensor,
    u2: torch.Tensor,
    srad: torch.Tensor,
    ra: torch.Tensor,
    elev_m: torch.Tensor,
) -> torch.Tensor:
    """Reference ET (mm/d), FAO-56 Penman–Monteith eq 6 (daily, G≈0)."""
    tmean = 0.5 * (tmin + tmax)
    es = 0.5 * (saturation_vapor_pressure(tmin) + saturation_vapor_pressure(tmax))
    ea = es * torch.clamp(rh_mean, 0.0, 100.0) / 100.0
    delta = svp_slope(tmean)
    gamma = psychrometric_constant(elev_m)

    rns = 0.77 * srad
    rso = (0.75 + 2e-5 * elev_m) * ra
    cloud = torch.clamp(1.35 * srad / torch.clamp(rso, min=1.0) - 0.35, 0.0, 1.0)
    rnl = STEFAN_BOLTZMANN * 0.5 * ((tmin + 273.16) ** 4 + (tmax + 273.16) ** 4) \
        * (0.34 - 0.14 * torch.sqrt(torch.clamp(ea, min=0.0))) * cloud
    rn = rns - rnl

    number = 0.408 * delta * rn + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)
    den = delta + gamma * (1.0 + 0.34 * u2)
    return torch.clamp(number / den, min=0.0)


def et0_hargreaves(tmin: torch.Tensor, tmax: torch.Tensor, ra: torch.Tensor) -> torch.Tensor:
    """Reference ET (mm/d), FAO-56 Hargreaves eq 52 — fallback when
    humidity/wind are unavailable (overestimates at high radiation)."""
    tmean = 0.5 * (tmin + tmax)
    trange = torch.clamp(tmax - tmin, min=0.0)
    return torch.clamp(0.0023 * (tmean + 17.8) * torch.sqrt(trange) * ra, min=0.0)


def day_length(lat_deg: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
    """Photoperiod N (h), FAO-56 eq 34. Clamped to [0, 24] at the poles."""
    latitude = torch.deg2rad(lat_deg)
    decl = 0.409 * torch.sin(2.0 * math.pi * doy / DAYS_PER_YEAR - 1.39)
    ws = torch.acos(torch.clamp(-torch.tan(latitude) * torch.tan(decl), -1.0, 1.0))
    return torch.clamp(24.0 / math.pi * ws, 0.0, 24.0)


def growing_degree_days(tmin: torch.Tensor, tmax: torch.Tensor, tbase: torch.Tensor) -> torch.Tensor:
    """Daily GDD contribution (°C·d)."""
    tmean = 0.5 * (tmin + tmax)
    return torch.clamp(tmean - tbase, min=0.0)


def temperature_factor(tmin: torch.Tensor, tmax: torch.Tensor,
                       tbase: torch.Tensor, topt: torch.Tensor) -> torch.Tensor:
    """Asymmetric trapezoid growth response ∈ [0,1].

    Rises linearly tbase→(topt−6), stays 1 until 34 °C, declines to 0 at 45 °C.
    """
    tmean = 0.5 * (tmin + tmax)
    # Plateau reached at topt−6: C4 crops are near-optimal well below their
    # nominal optimum, matching documented growth response curves.
    plateau = topt - 6.0
    rise = torch.clamp((tmean - tbase) / torch.clamp(plateau - tbase, min=1.0),
                       0.0, 1.0)
    fall = torch.clamp((45.0 - tmean) / (45.0 - 34.0), 0.0, 1.0)
    return rise * fall
