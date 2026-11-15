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
PARAMETERS = ("T2M_MIN", "T2M_MAX", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "RH2M", "WS2M", "GWETROOT")
CANON = {
    "T2M_MIN": "tmin",
    "T2M_MAX": "tmax",
    "ALLSKY_SFC_SW_DWN": "srad",
    "PRECTOTCORR": "precip",
    "RH2M": "rh",
    "WS2M": "u2",
}
FORCING_VARS = ("tmin", "tmax", "srad", "precip", "rh", "u2")
FILL_VALUE = -999.0


def request_params(
    latitude: float, longitude: float, temporal: Temporal, parameters: tuple[str, ...] = PARAMETERS
) -> dict:
    return {
        "parameters": ",".join(parameters),
        "community": "AG",
        "latitude": latitude,
        "longitude": longitude,
        "start": temporal.start.replace("-", ""),
        "end": temporal.end.replace("-", ""),
        "format": "JSON",
    }


def parse_daily(
    payload: dict, parameters: tuple[str, ...] = PARAMETERS
) -> dict[str, dict[str, float]]:
    raw = payload["properties"]["parameter"]
    return {
        p: {d: (float("nan") if v <= FILL_VALUE else float(v)) for d, v in raw.get(p, {}).items()}
        for p in parameters
    }


class Power:
    """Thin, testable wrapper around httpx."""

    name = "nasa-power-agro"
    requires_auth = False

    def __init__(self, client: httpx.Client | None = None, timeout: float = 120.0) -> None:
        self._client = client or httpx.Client(timeout=timeout)

    def fetch(
        self,
        area: None,
        temporal: Temporal,
        *,
        latitude: float,
        longitude: float,
        parameters: tuple[str, ...] = PARAMETERS,
    ) -> dict[str, dict[str, float]]:
        response = self._client.get(
            BASE_URL, params=request_params(latitude, longitude, temporal, parameters)
        )
        response.raise_for_status()
        return parse_daily(response.json(), parameters)

    def data_year(
        self, latitude: float, longitude: float, year: int
    ) -> dict[str, dict[str, float]]:
        """Raw daily series for one calendar year — all PARAMETERS."""
        return self.fetch(
            None, Temporal(f"{year}-01-01", f"{year}-12-31"), latitude=latitude, longitude=longitude
        )

    def forcing_year(self, latitude: float, longitude: float, year: int) -> Forcing:
        """One 365-day Forcing for a calendar year (B=1)."""
        return series_to_forcing(self.data_year(latitude, longitude, year), year)

    def data_years(
        self, latitude: float, longitude: float, years: list[int], workers: int = 8
    ) -> list[dict[str, dict[str, float]]]:
        """Fetch calendar years in parallel — pure I/O, near-linear speedup."""
        if len(years) <= 2 or workers <= 1:
            return [self.data_year(latitude, longitude, y) for y in years]
        with ThreadPoolExecutor(max_workers=min(workers, len(years))) as pool:
            return list(pool.map(lambda y: self.data_year(latitude, longitude, y), years))

    def forcing_years(
        self, latitude: float, longitude: float, years: list[int], workers: int = 8
    ) -> list[Forcing]:
        return [
            series_to_forcing(d, y)
            for d, y in zip(
                self.data_years(latitude, longitude, years, workers), years, strict=True
            )
        ]


def vars_to_forcing(data: dict[str, dict[str, float]], year: int) -> Forcing:
    """Canonical-name daily dicts → Forcing (1, 365); Feb 29 dropped."""
    ndays = (dt.date(year, 12, 31) - dt.date(year, 1, 1)).days + 1
    days = [dt.date(year, 1, 1) + dt.timedelta(d) for d in range(ndays)]
    days = [d for d in days if not (d.month == 2 and d.day == 29)]
    keys = [d.strftime("%Y%m%d") for d in days]

    def column(variable: str, default: float) -> torch.Tensor:
        s = data.get(variable, {})
        values = [s.get(k, float("nan")) for k in keys]
        t = torch.tensor(values, dtype=DTYPE)
        return torch.nan_to_num(t, nan=default)

    return Forcing(
        tmin=column("tmin", 5.0).unsqueeze(0),
        tmax=column("tmax", 15.0).unsqueeze(0),
        srad=column("srad", 12.0).unsqueeze(0).clamp(min=0.1),
        precip=column("precip", 0.0).unsqueeze(0).clamp(min=0.0),
        rh=column("rh", 70.0).unsqueeze(0).clamp(1.0, 100.0),
        u2=column("u2", 2.0).unsqueeze(0).clamp(min=0.2),
        doy=torch.arange(1, DAYS_PER_YEAR + 1, dtype=DTYPE).unsqueeze(0),
    )


def series_to_forcing(data: dict[str, dict[str, float]], year: int) -> Forcing:
    """POWER daily dicts → Forcing via canonical names."""
    return vars_to_forcing({CANON.get(p, p): s for p, s in data.items()}, year)


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
