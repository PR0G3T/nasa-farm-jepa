"""Forcing providers — one dispatch point, pluggable backends.

`ensemble(source=...)` is the single entry the rest of the codebase uses;
new sources register a callable so forecast products (AIFS-ENS, WeatherNext,
GenCast-derived fields) plug in without touching call sites.
"""

from __future__ import annotations

from collections.abc import Callable

from nasa_farm.types import Forcing

_PROVIDERS: dict[str, Callable[..., Forcing]] = {}


def register(name: str, fn: Callable[..., Forcing]) -> None:
    """Register `fn(lat, lon, years, horizon, members, seed, **kw) → Forcing`."""
    _PROVIDERS[name] = fn


def _power(lat, lon, years, horizon, members, seed, **kw) -> Forcing:
    from nasa_farm.acquire.power import PowerClient
    from nasa_farm.forcing.base import climatology_ensemble
    hist = PowerClient().forcing_years(lat, lon, list(years))
    return climatology_ensemble(hist, horizon)


def _synth(lat, lon, years, horizon, members, seed, **kw) -> Forcing:
    from nasa_farm.forcing.base import augment, synthetic_year, tile_to_horizon
    base = tile_to_horizon(synthetic_year(lat, seed=seed), horizon)
    return augment(base, members - 1, seed=seed)


def _gen(lat, lon, years, horizon, members, seed, model=None, **kw) -> Forcing:
    if model is None:
        raise ValueError("source='gen' requires model=<FlowGen>")
    from nasa_farm.forcing.gen import sample_flow
    return sample_flow(model, horizon * 365, members, seed=seed)


def _file(lat, lon, years, horizon, members, seed, path=None, **kw) -> Forcing:
    """xarray dataset → Forcing. Variables named per FORCING_VARS on a
    (member, day) layout; extra dims are averaged out."""
    import xarray as xr
    ds = xr.open_dataset(path)
    get = lambda k: (ds[k].values.reshape(-1, ds[k].sizes["day"])
                     if "day" in ds[k].sizes else ds[k].values)
    import torch
    from nasa_farm.types import FORCING_VARS
    return Forcing(*(torch.as_tensor(get(k)) for k in FORCING_VARS))


register("power", _power)
register("synth", _synth)
register("gen", _gen)
register("file", _file)


def ensemble(source: str = "power", lat: float = 0.0, lon: float = 0.0,
             years: tuple[int, ...] = (), horizon: int = 1, members: int = 8,
             seed: int = 0, **kw) -> Forcing:
    """Dispatch to a registered forcing provider."""
    if source not in _PROVIDERS:
        raise KeyError(
            f"unknown forcing source {source!r}; have {sorted(_PROVIDERS)}")
    return _PROVIDERS[source](lat, lon, years, horizon, members, seed, **kw)
