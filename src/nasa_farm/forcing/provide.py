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
        raise ValueError("source='gen' requires model=<WxGen>")
    from nasa_farm.forcing.gen import sample_gen
    return sample_gen(model, horizon * 365, members, seed=seed, **kw)


def _ee_daily(lat, lon, years, horizon, members, seed, *,
              collection: str, bands: dict[str, str],
              member_band: str | None = None, **_kw) -> Forcing:
    """EE forecast-ensemble collection → daily Forcing at a point.

    `bands` maps forcing names to collection band names; `member_band`
    (if the product exposes one) indexes ensemble members — else members
    are the collection images. Aggregation is daily-mean over the AOI
    centroid. Requires the `gee` extra and an authenticated EE session.
    """
    import importlib
    ee = importlib.import_module("ee")
    ee.Initialize()
    pt = ee.Geometry.Point([lon, lat])
    coll = ee.ImageCollection(collection).select(*bands.values())
    start, end = ((f"{years[0]}-01-01", f"{years[-1] + 1}-01-01")
                  if years else ("2000-01-01", "2100-01-01"))
    imgs = coll.filterDate(start, end).filterBounds(pt)
    rows = []
    for feat in imgs.limit(1024).getInfo()["features"]:
        p = feat["properties"]
        row = {k: float(p[v]) for k, v in bands.items() if v in p}
        row["member"] = int(p.get(member_band, 0)) if member_band else len(rows)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"EE collection {collection!r} returned no images")
    by_member: dict[int, dict] = {}
    for r in rows:
        by_member.setdefault(r["member"], {k: [] for k in bands})
        for k in bands:
            by_member[r["member"]][k].append(r[k])
    mem = sorted(by_member.values(), key=lambda d: -len(d["tmin"]))[:members]
    t = min(len(m[k2]) for m in mem for k2 in bands)
    import torch
    ch = {k: torch.stack([torch.as_tensor(m[k][:t]) for m in mem]).to(
        torch.float64) for k in bands}
    doy = (torch.arange(t, dtype=torch.float64) % 365 + 1).expand(len(mem), -1)
    return Forcing(tmin=ch["tmin"], tmax=ch["tmax"], srad=ch["srad"],
                   precip=ch["precip"], rh=ch["rh"], u2=ch["u2"], doy=doy)


def _wn3(lat, lon, years, horizon, members, seed, **kw) -> Forcing:
    """WeatherNext 3 ensemble via Earth Engine (hourly → caller-side daily
    aggregation is already handled by `bands` daily means)."""
    return _ee_daily(
        lat, lon, years, horizon, members, seed,
        collection=kw.pop("collection",
                          "GOOGLE/WEATHER_NEXT_3/ENSEMBLE/DAILY"),
        bands={"tmin": "temperature_2m_min", "tmax": "temperature_2m_max",
               "srad": "surface_solar_radiation_downwards_sum",
               "precip": "total_precipitation_sum",
               "rh": "relative_humidity_2m_mean",
               "u2": "wind_speed_10m_mean"},
        member_band="ensemble_member", **kw)


def _aifs(lat, lon, years, horizon, members, seed, **kw) -> Forcing:
    """ECMWF AIFS-ENS open-data daily ensemble via Earth Engine."""
    return _ee_daily(
        lat, lon, years, horizon, members, seed,
        collection=kw.pop("collection", "ECMWF/AIFS_ENS/DAILY"),
        bands={"tmin": "2m_temperature_min", "tmax": "2m_temperature_max",
               "srad": "surface_net_solar_radiation_sum",
               "precip": "total_precipitation",
               "rh": "relative_humidity_2m",
               "u2": "10m_wind_speed"},
        member_band="number", **kw)


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
register("wn3", _wn3)
register("aifs", _aifs)


def ensemble(source: str = "power", lat: float = 0.0, lon: float = 0.0,
             years: tuple[int, ...] = (), horizon: int = 1, members: int = 8,
             seed: int = 0, **kw) -> Forcing:
    """Dispatch to a registered forcing provider."""
    if source not in _PROVIDERS:
        raise KeyError(
            f"unknown forcing source {source!r}; have {sorted(_PROVIDERS)}")
    return _PROVIDERS[source](lat, lon, years, horizon, members, seed, **kw)
