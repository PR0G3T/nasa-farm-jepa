"""Open-Meteo clients — key-free operational ensemble forcing.

`ens`: NWP ensembles (default ECMWF IFS ENS, 51 members, ~15 days) via
ensemble-api.open-meteo.com. `seas5`: ECMWF SEAS5/EC46 seasonal ensembles
(51 members, ~7 months) via seasonal-api.open-meteo.com. Both return each
member as `{var}_member{NN}` daily columns; both map onto the canonical
Forcing contract (members on the batch axis, doy on the 365-day axis with
Feb 29 dropped — the POWER convention).
"""

from __future__ import annotations

import datetime as dt
import re

import httpx2 as httpx
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.types import FORCING_VARS, Forcing

ENS_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
SEAS_URL = "https://seasonal-api.open-meteo.com/v1/seasonal"

DAILY_VARS: dict[str, str] = {
    "tmin": "temperature_2m_min",
    "tmax": "temperature_2m_max",
    "precip": "precipitation_sum",
    "srad": "shortwave_radiation_sum",
    "rh": "relative_humidity_2m_mean",
    "u2": "wind_speed_10m_mean",
}

_MEMBER = re.compile(r"^(.+)_member(\d+)$")

_FILL = {"tmin": 10.0, "tmax": 20.0, "srad": 15.0, "precip": 0.0,
         "rh": 70.0, "u2": 2.0}
_LO = {"tmin": -60.0, "tmax": -60.0, "srad": 0.0, "precip": 0.0,
       "rh": 0.0, "u2": 0.0}
_HI = {"tmax": 60.0, "rh": 100.0, "u2": 60.0}


def _doy(date: dt.date) -> int:
    """Day-of-year on the 365-day axis: Feb 29 dropped (POWER convention)."""
    d = date.timetuple().tm_yday
    leap = date.year % 4 == 0 and (date.year % 100 != 0 or date.year % 400 == 0)
    return d - 1 if (leap and d > 59) else d


def members_to_forcing(dates: list[str],
                       cols: dict[int, dict[str, list[float]]],
                       members: int) -> Forcing:
    """{member_id: {api_var: [daily values]}} → Forcing (S, T).

    Members are sorted by id and capped at `members`; missing variables
    fall back to physical defaults so partial products still satisfy the
    FORCING_VARS contract."""
    ids = sorted(cols)[:members]
    keep = [i for i, d in enumerate(dates)
            if not d.endswith("-02-29")]        # POWER drops Feb 29
    dates = [dates[i] for i in keep]
    n = len(dates)

    def column(variable: str) -> torch.Tensor:
        api = DAILY_VARS[variable] if variable in DAILY_VARS else variable
        rows = []
        for i in ids:
            s = cols[i].get(api)
            v = (torch.full((n,), _FILL.get(variable, 0.0), dtype=DTYPE)
                 if s is None else
                 torch.tensor([s[i] for i in keep],
                              dtype=DTYPE).nan_to_num(_FILL.get(variable, 0.0)))
            rows.append(v.clamp(_LO.get(variable, float("-inf")),
                                _HI.get(variable, float("inf"))))
        return torch.stack(rows) if rows else torch.zeros(0, n, dtype=DTYPE)

    values = {k: column(k) for k in FORCING_VARS if k not in ("doy", "co2")}
    values["tmax"] = torch.maximum(values["tmax"], values["tmin"])
    doy = torch.tensor([_doy(dt.date.fromisoformat(d)) for d in dates],
                       dtype=DTYPE).expand(len(ids), -1)
    return Forcing(**values, doy=doy)


class OpenMeteo:
    """httpx2-backed fetcher for the Open-Meteo ensemble endpoints.
    `client` is injectable so tests run against a MockTransport."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(timeout=60.0)

    def _members(self, url: str, model: str, latitude: float, longitude: float,
                 extra: dict | None = None) -> tuple[list[str], dict]:
        parameters = {"latitude": latitude, "longitude": longitude, "models": model,
                  "daily": ",".join(DAILY_VARS.values()),
                  "wind_speed_unit": "ms", "timezone": "GMT", **(extra or {})}
        r = self.client.get(url, params=parameters)
        r.raise_for_status()
        daily = r.json()["daily"]
        cols: dict[int, dict[str, list[float]]] = {}
        for key, values in daily.items():
            m = _MEMBER.match(key)
            if m:
                cols.setdefault(int(m.group(2)), {})[m.group(1)] = values
        if not cols:
            raise RuntimeError(
                f"{url} returned no ensemble members for model {model!r}")
        return daily["time"], cols

    def ensemble(self, latitude: float, longitude: float, model: str = "ecmwf_ifs025",
            members: int = 51, **keywords) -> Forcing:
        """Medium-range ensemble → Forcing (members, ~15 days)."""
        dates, cols = self._members(ENS_URL, model, latitude, longitude, keywords)
        return members_to_forcing(dates, cols, members)

    def seas5(self, latitude: float, longitude: float, model: str = "ecmwf_seas5",
              members: int = 51, **keywords) -> Forcing:
        """Seasonal ensemble → Forcing (members, ~7 months daily)."""
        dates, cols = self._members(SEAS_URL, model, latitude, longitude, keywords)
        return members_to_forcing(dates, cols, members)
