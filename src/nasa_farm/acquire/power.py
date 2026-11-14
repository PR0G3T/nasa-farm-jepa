"""NASA POWER Agroclimatology client — no API key required.

Endpoint: https://power.larc.nasa.gov/api/temporal/daily/point
Returns daily point time series; this module maps them straight into the
model's Forcing layout (365-day years, leap days dropped).
"""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import httpx2 as httpx
import torch

from nasa_farm.acquire.base import Temporal
from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.types import Forcing

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = ("T2M_MIN", "T2M_MAX", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN",
              "RH2M", "WS2M", "GWETROOT")
FILL_VALUE = -999.0


def request_params(lat: float, lon: float, temporal: Temporal,
                   parameters: tuple[str, ...] = PARAMETERS) -> dict:
    return {
        "parameters": ",".join(parameters),
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": temporal.start.replace("-", ""),
        "end": temporal.end.replace("-", ""),
        "format": "JSON",
    }


def parse_daily(payload: dict, parameters: tuple[str, ...] = PARAMETERS) -> dict[str, dict[str, float]]:
    raw = payload["properties"]["parameter"]
    return {
        p: {d: (float("nan") if v <= FILL_VALUE else float(v))
            for d, v in raw.get(p, {}).items()}
        for p in parameters
    }


class PowerClient:
    """Thin, testable wrapper around httpx."""

    name = "nasa-power-agro"
    requires_auth = False

    def __init__(self, client: httpx.Client | None = None, timeout: float = 120.0) -> None:
        self._client = client or httpx.Client(timeout=timeout)

    def fetch(self, aoi: None, temporal: Temporal, *,
              lat: float, lon: float,
              parameters: tuple[str, ...] = PARAMETERS) -> dict[str, dict[str, float]]:
        resp = self._client.get(BASE_URL, params=request_params(lat, lon, temporal, parameters))
        resp.raise_for_status()
        return parse_daily(resp.json(), parameters)

    def data_year(self, lat: float, lon: float, year: int) -> dict[str, dict[str, float]]:
        """Raw daily series for one calendar year — all PARAMETERS."""
        return self.fetch(None, Temporal(f"{year}-01-01", f"{year}-12-31"),
                          lat=lat, lon=lon)

    def forcing_year(self, lat: float, lon: float, year: int) -> Forcing:
        """One 365-day Forcing for a calendar year (B=1)."""
        return series_to_forcing(self.data_year(lat, lon, year), year)

    def data_years(self, lat: float, lon: float, years: list[int],
                   workers: int = 8) -> list[dict[str, dict[str, float]]]:
        """Fetch calendar years in parallel — pure I/O, near-linear speedup."""
        if len(years) <= 2 or workers <= 1:
            return [self.data_year(lat, lon, y) for y in years]
        with ThreadPoolExecutor(max_workers=min(workers, len(years))) as pool:
            return list(pool.map(lambda y: self.data_year(lat, lon, y), years))

    def forcing_years(self, lat: float, lon: float, years: list[int],
                      workers: int = 8) -> list[Forcing]:
        return [series_to_forcing(d, y)
                for d, y in zip(self.data_years(lat, lon, years, workers), years)]


def series_to_forcing(data: dict[str, dict[str, float]], year: int) -> Forcing:
    """POWER daily dicts → Forcing (1, 365); Feb 29 dropped, NaN→0-filled."""
    ndays = (dt.date(year, 12, 31) - dt.date(year, 1, 1)).days + 1
    days = [dt.date(year, 1, 1) + dt.timedelta(d) for d in range(ndays)]
    days = [d for d in days if not (d.month == 2 and d.day == 29)]
    keys = [d.strftime("%Y%m%d") for d in days]

    def col(param: str, default: float) -> torch.Tensor:
        s = data.get(param, {})
        vals = [s.get(k, float("nan")) for k in keys]
        t = torch.tensor(vals, dtype=DTYPE)
        return torch.nan_to_num(t, nan=default)

    return Forcing(
        tmin=col("T2M_MIN", 5.0).unsqueeze(0),
        tmax=col("T2M_MAX", 15.0).unsqueeze(0),
        srad=col("ALLSKY_SFC_SW_DWN", 12.0).unsqueeze(0).clamp(min=0.1),
        precip=col("PRECTOTCORR", 0.0).unsqueeze(0).clamp(min=0.0),
        rh=col("RH2M", 70.0).unsqueeze(0).clamp(1.0, 100.0),
        u2=col("WS2M", 2.0).unsqueeze(0).clamp(min=0.2),
        doy=torch.arange(1, DAYS_PER_YEAR + 1, dtype=DTYPE).unsqueeze(0),
    )


def series_to_wet(data: dict[str, dict[str, float]], year: int) -> torch.Tensor:
    """GWETROOT daily series → (365,) root-zone soil wetness fraction —
    the NASA GEOS-assimilated soil moisture observation."""
    ndays = (dt.date(year, 12, 31) - dt.date(year, 1, 1)).days + 1
    days = [dt.date(year, 1, 1) + dt.timedelta(d) for d in range(ndays)]
    days = [d for d in days if not (d.month == 2 and d.day == 29)]
    keys = [d.strftime("%Y%m%d") for d in days]
    s = data.get("GWETROOT", {})
    t = torch.tensor([s.get(k, float("nan")) for k in keys], dtype=DTYPE)
    return torch.nan_to_num(t, nan=0.5).clamp(0.0, 1.0)
