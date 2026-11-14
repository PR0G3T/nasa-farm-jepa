"""Forecast-to-climatology bridging.

Operational ensembles end at ~15 days (medium range) or ~7 months
(seasonal); rotation horizons are years. `bridge` continues each member
with the historical year that best matches *its own* forecast-window
trajectory — analog-conditioned tails preserve member identity, temporal
coherence and the seasonal signal the ensemble carries.
"""

from __future__ import annotations

import math

import torch

from nasa_farm.constants import DAYS_PER_YEAR
from nasa_farm.forcing.analog import analog_weights, year_features
from nasa_farm.forcing.base import tile_to_horizon
from nasa_farm.types import FORCING_VARS, Forcing


def span(f: Forcing, a: int, b: int) -> Forcing:
    """Time slice [a, b) of a Forcing — the column-window constructor."""
    return Forcing(*(getattr(f, k)[:, a:b] for k in FORCING_VARS))


def roll_to_doy(f: Forcing, start: int) -> Forcing:
    """Rotate a 365-day Forcing so position 0 carries day-of-year `start`,
    aligning a historical year onto any forecast window's doy axis."""
    if f.days != DAYS_PER_YEAR:
        raise ValueError(f"roll_to_doy needs a 365-day Forcing, got {f.days}")
    shift = (int(f.doy[0, 0]) - start) % DAYS_PER_YEAR
    return f if not shift else Forcing(*(
        getattr(f, k).roll(shift, dims=-1) for k in FORCING_VARS))


def bridge(fcst: Forcing, hist: list[Forcing], days: int,
           seed: int = 0, bw: float = 1.0) -> Forcing:
    """Each forecast member → `days` total: forecast head + analog tail.

    Members share the doy axis, so historical years are rolled once onto
    it; per member, `analog_weights` over the rolled forecast window picks
    the year whose trajectory continues that member's realized weather.
    """
    s, f = fcst.tmin.shape
    if f >= days:
        return span(fcst, 0, days)
    start = int(fcst.doy[0, 0])
    rolled = [roll_to_doy(y, start) for y in hist]
    win = [span(y, 0, f) for y in rolled]
    tails = [span(tile_to_horizon(y, math.ceil(days / DAYS_PER_YEAR)), f, days)
             for y in rolled]
    gen = torch.Generator().manual_seed(seed)
    members = []
    for m in range(s):
        w = analog_weights(win, year_features(span(fcst, m, m + 1)), bw=bw)
        i = int(torch.multinomial(w, 1, generator=gen))
        members.append(Forcing(*(torch.cat(
            [getattr(fcst, k)[m:m + 1], getattr(tails[i], k)], -1)
            for k in FORCING_VARS)))
    return Forcing(*(torch.cat(
        [getattr(fx, k) for fx in members]) for k in FORCING_VARS))
