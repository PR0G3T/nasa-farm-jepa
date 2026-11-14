"""NLDAS-3 forcing — public S3 (`nasa-waterinsight`), North America only.

Daily files are whole-domain ~1 km NetCDF (~0.7 GB) with no server-side
subsetting, so reads go through anonymous `s3fs` range requests: opening a
day costs a few metadata reads and each variable×point is one more range
read — kilobytes, not gigabytes. One open serves every field.

Vars (daily collection): Tair, Tair_min/max (K), Qair (kg/kg), PSurf (Pa),
Wind_E/N (m/s, 10 m), LWdown/SWdown (W/m²), Rainf (daily *total*, kg/m²).

Chunks are (time=1, lat=500, lon=900) per variable (README of
NASAWaterInsight/NLDAS-3): one point read pulls a full 5°×9° chunk — MBs,
not KBs — but every field inside that chunk is served by the same read.
Beta caveats (known_issues.md): a few negative SWdown values leak into the
daily means; a non-beta release is announced → VERSION is a constant.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr

from nasa_farm.acquire.base import BBox, Temporal
from nasa_farm.acquire.support import sample
from nasa_farm.acquire.power import vars_to_forcing
from nasa_farm.constants import DTYPE
from nasa_farm.types import Forcing

DOMAIN = BBox(-169.0, 7.0, -52.0, 72.0)
"""NLDAS-3 grid: 7–72 °N, 169–52 °W (README, 6,500 × 11,700 points)."""

BASE = "https://nasa-waterinsight.s3.amazonaws.com/NLDAS3/forcing/daily"
VERSION = "030.beta"
KEY = "{base}/{ym}/NLDAS_FOR0010_D.A{day}.{ver}.nc"
VARS = ("Tair", "Tair_min", "Tair_max", "Qair", "PSurf",
        "Wind_E", "Wind_N", "SWdown", "Rainf")


def covers(lat: float, lon: float) -> bool:
    return DOMAIN.west <= lon <= DOMAIN.east and DOMAIN.south <= lat <= DOMAIN.north


def day_url(day: dt.date) -> str:
    return KEY.format(base=BASE, ym=day.strftime("%Y%m"),
                      day=day.strftime("%Y%m%d"), ver=VERSION)


def open_day(day: dt.date, fs=None) -> xr.Dataset:
    import fsspec
    return xr.open_dataset(fsspec.open(day_url(day)).open(),
                           engine="h5netcdf")


def _fs():
    return None  # HTTP range reads — no credentials needed


def _rh(qair: float, tair_k: float, psurf_pa: float) -> float:
    """Specific humidity + T + P → RH% via Bolton/Clausius-Clapeyron."""
    es = 611.0 * math.exp(17.67 * (tair_k - 273.15) / (tair_k - 29.65))
    ws = 0.622 * es / max(psurf_pa - 0.378 * es, 1.0)
    return float(np.clip(100.0 * qair / max(ws, 1e-9), 1.0, 100.0))


def rain_is_rate(ds: xr.Dataset) -> bool:
    """True when Rainf is a flux (kg m-2 s-1) rather than a daily total.
    Decided from the units attribute — never from the value itself."""
    u = str(ds["Rainf"].attrs.get("units", "")) if "Rainf" in ds else ""
    return "s-1" in u or "/s" in u


def convert(vals: dict[str, float], rain_rate: bool = False
            ) -> dict[str, float]:
    """NLDAS-3 units → canonical forcing vars."""
    tmin = vals.get("Tair_min", np.nan) - 273.15
    tmax = vals.get("Tair_max", np.nan) - 273.15
    tair = vals.get("Tair", (tmin + tmax) / 2 + 273.15)
    wind = math.hypot(vals.get("Wind_E", 0.0), vals.get("Wind_N", 0.0))
    rainf = float(vals.get("Rainf", np.nan))
    if rain_rate:  # kg m-2 s-1 → mm/day; daily totals are already mm
        rainf *= 86400.0
    return {
        "tmin": tmin, "tmax": tmax,
        "srad": max(vals.get("SWdown", np.nan), 0.0) * 0.0864,  # W/m² → MJ/m²/day
        "precip": rainf,
        "rh": _rh(vals.get("Qair", np.nan), tair, vals.get("PSurf", 101325.0)),
        "u2": wind * 0.748,  # FAO-56: 10 m → 2 m
    }


def day_point(fs, day: dt.date, lat: float, lon: float) -> dict[str, float]:
    """One day, one point → canonical forcing values."""
    with open_day(day, fs) as ds:
        vals = {v: sample(ds, v, lat, lon) for v in VARS if v in ds}
        rate = rain_is_rate(ds)
    return convert(vals, rate)


def year_point(lat: float, lon: float, year: int) -> Forcing:
    """365-day Forcing for one point — sequential days, ~KB each."""
    if not covers(lat, lon):
        raise ValueError(f"({lat},{lon}) outside NLDAS-3 domain")
    fs = _fs()
    nd = (dt.date(year, 12, 31) - dt.date(year, 1, 1)).days + 1
    data: dict[str, dict[str, float]] = {v: {} for v in
                                         ("tmin", "tmax", "srad", "precip", "rh", "u2")}
    for i in range(nd):
        day = dt.date(year, 1, 1) + dt.timedelta(i)
        if day.month == 2 and day.day == 29:
            continue
        for var, v in day_point(fs, day, lat, lon).items():
            data[var][day.strftime("%Y%m%d")] = v
    return vars_to_forcing(data, year)


class NldasClient:
    """DataSource contract — forcing over the NLDAS-3 domain."""

    name = "nldas3-forcing"
    requires_auth = False

    def fetch(self, aoi: BBox | None, temporal: Temporal, *,
              lat: float, lon: float, **kw: Any) -> Forcing:
        year = int(temporal.start[:4])
        return year_point(lat, lon, year)

    def forcing_years(self, lat: float, lon: float, years: list[int],
                      **kw) -> list[Forcing]:
        return [year_point(lat, lon, y) for y in years]
