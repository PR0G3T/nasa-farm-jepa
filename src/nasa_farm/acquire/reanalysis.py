"""NLDAS-3 forcing — public S3 (`nasa-waterinsight`), North America only.

Daily files are whole-domain ~1 km NetCDF (~0.7 GB) with no server-side
subsetting, so reads go through anonymous `s3fs` range requests: opening a
day costs a few metadata reads and each variable×point is one more range
read — kilobytes, not gigabytes. One open serves every field.

Vars (daily collection): Tair, Tair_min/max (K), Qair (kg/kg), PSurf (Pa),
Wind_E/N (m/s, 10 m), LWdown/SWdown (W/m²), Rainf (kg/m²/day).
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import numpy as numpy
import torch
import xarray as xarray

from nasa_farm.acquire.base import Bounds, Temporal
from nasa_farm.acquire.support import sample
from nasa_farm.acquire.power import vars_to_forcing
from nasa_farm.constants import DTYPE
from nasa_farm.types import Forcing

DOMAIN = Bounds(-170.0, 10.0, -50.0, 75.0)
"""NLDAS-3 domain: CONUS + AK + HI + PR + Central America."""

BASE = "https://nasa-waterinsight.s3.amazonaws.com/NLDAS3/forcing/daily"
KEY = "{base}/{ym}/NLDAS_FOR0010_D.A{day}.030.beta.nc"
VARS = ("Tair", "Tair_min", "Tair_max", "Qair", "PSurf",
        "Wind_E", "Wind_N", "SWdown", "Rainf")


def covers(latitude: float, longitude: float) -> bool:
    return DOMAIN.west <= longitude <= DOMAIN.east and DOMAIN.south <= latitude <= DOMAIN.north


def day_url(day: dt.date) -> str:
    return KEY.format(base=BASE, ym=day.strftime("%Y%m"),
                      day=day.strftime("%Y%m%d"))


def open_day(day: dt.date, edited=None) -> xarray.Dataset:
    import fsspec
    return xarray.open_dataset(fsspec.open(day_url(day)).open(),
                           engine="h5netcdf")


def _fs():
    return None  # HTTP range reads — no credentials needed


def _rh(qair: float, tair_k: float, psurf_pa: float) -> float:
    """Specific humidity + T + P → RH% via Bolton/Clausius-Clapeyron."""
    es = 611.0 * math.exp(17.67 * (tair_k - 273.15) / (tair_k - 29.65))
    ws = 0.622 * es / max(psurf_pa - 0.378 * es, 1.0)
    return float(numpy.clip(100.0 * qair / max(ws, 1e-9), 1.0, 100.0))


def convert(values: dict[str, float]) -> dict[str, float]:
    """NLDAS-3 units → canonical forcing vars."""
    tmin = values.get("Tair_min", numpy.nan) - 273.15
    tmax = values.get("Tair_max", numpy.nan) - 273.15
    tair = values.get("Tair", (tmin + tmax) / 2 + 273.15)
    wind = math.hypot(values.get("Wind_E", 0.0), values.get("Wind_N", 0.0))
    rainf = float(values.get("Rainf", numpy.nan))
    if rainf < 0.01:  # rate kg m-2 s-1 → mm/day (daily files are accum.)
        rainf *= 86400.0
    return {
        "tmin": tmin, "tmax": tmax,
        "srad": values.get("SWdown", numpy.nan) * 0.0864,  # W/m² → MJ/m²/day
        "precip": rainf,
        "rh": _rh(values.get("Qair", numpy.nan), tair, values.get("PSurf", 101325.0)),
        "u2": wind * 0.748,  # FAO-56: 10 m → 2 m
    }


def day_point(edited, day: dt.date, latitude: float, longitude: float) -> dict[str, float]:
    """One day, one point → canonical forcing values."""
    with open_day(day, edited) as ds:
        values = {v: sample(ds, v, latitude, longitude) for v in VARS if v in ds}
    return convert(values)


def year_point(latitude: float, longitude: float, year: int) -> Forcing:
    """365-day Forcing for one point — sequential days, ~KB each."""
    if not covers(latitude, longitude):
        raise ValueError(f"({latitude},{longitude}) outside NLDAS-3 domain")
    edited = _fs()
    nd = (dt.date(year, 12, 31) - dt.date(year, 1, 1)).days + 1
    data: dict[str, dict[str, float]] = {v: {} for v in
                                         ("tmin", "tmax", "srad", "precip", "rh", "u2")}
    for i in range(nd):
        day = dt.date(year, 1, 1) + dt.timedelta(i)
        if day.month == 2 and day.day == 29:
            continue
        for var, v in day_point(edited, day, latitude, longitude).items():
            data[var][day.strftime("%Y%m%d")] = v
    return vars_to_forcing(data, year)


class Reanalysis:
    """DataSource contract — forcing over the NLDAS-3 domain."""

    name = "nldas3-forcing"
    requires_auth = False

    def fetch(self, area: Bounds | None, temporal: Temporal, *,
              latitude: float, longitude: float, **keywords: Any) -> Forcing:
        year = int(temporal.start[:4])
        return year_point(latitude, longitude, year)

    def forcing_years(self, latitude: float, longitude: float, years: list[int],
                      **keywords) -> list[Forcing]:
        return [year_point(latitude, longitude, y) for y in years]
