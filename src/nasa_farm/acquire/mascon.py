"""GRACE-FO JPL mascon — single-file monthly TWS anomaly, regional cell.

RL06.3 mascons resolve ~3° spherical caps, so no single field is
resolvable — the honest unit is the cell mean over a region. The product
is ONE netCDF (monthly, 0.5° posting, cm water equivalent) opened once;
`series` extracts the cell containing the region centroid and returns a
monthly mm series that lands as `obs/mascon` on every field — the
regional water-storage constraint the aquifer channel answers to.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from nasa_farm.acquire.earthdata import GRACE_MASCON


def open_mascon():
    """Search + open the mascon product → (xr.Dataset, handle)."""
    import earthaccess
    import xarray as xarray

    earthaccess.login(strategy="environment")
    results = earthaccess.search_data(short_name=GRACE_MASCON.short_name, count=5)
    if not results:
        raise FileNotFoundError(GRACE_MASCON.short_name)
    latest = sorted(results, key=lambda r: str(r["umm"]["GranuleUR"]))[-1]
    handle = earthaccess.open([latest])[0]
    file = handle.open() if hasattr(handle, "open") else handle
    return xarray.open_dataset(file, engine="h5netcdf", phony_dims="access"), handle


def _dates(ds) -> np.ndarray:
    """time coord → yyyymmdd int32 stamped on the 15th — mascons carry no
    sub-monthly timing. Handles datetime64, epoch-seconds and
    days-since-2002 encodings."""
    time = ds["time"].values
    if np.issubdtype(time.dtype, np.datetime64):
        parts = np.datetime_as_string(time.astype("datetime64[D]"))
        return np.array([int(p.replace("-", "")[:6] + "15") for p in parts], np.int32)
    seconds = np.asarray(time, np.float64)
    if seconds.max() > 10_000_000:  # epoch seconds
        epoch = seconds.astype(np.int64)
        dates = [dt.datetime.fromtimestamp(s, tz=dt.UTC).date() for s in epoch]
    else:  # days since 2002-01-01 (JPL mascon convention)
        dates = [dt.date(2002, 1, 1) + dt.timedelta(days=int(s)) for s in seconds]
    return np.array([d.year * 10000 + d.month * 100 + 15 for d in dates], np.int32)


def series(latitude: float, longitude: float) -> tuple[np.ndarray, np.ndarray]:
    """(dates yyyymmdd int32, tws mm f32) monthly at the containing cell.

    lwe_thickness is posted in cm of equivalent water height → ×10 mm.
    """
    ds, handle = open_mascon()
    try:
        variable = next(v for v in ("lwe_thickness", "lwe") if v in ds)
        cell = ds[variable].sel(lat=latitude, lon=longitude, method="nearest")
        values = np.asarray(cell.values, np.float64).reshape(-1) * 10.0
        return _dates(ds)[: len(values)], values.astype(np.float32)
    finally:
        ds.close()
        handle.close()


def crawl(fields: list[dict], lake, log=print) -> None:
    """One mascon open → regional monthly TWS on every field."""
    latitude = float(np.mean([f["lat"] for f in fields]))
    longitude = float(np.mean([f["lon"] for f in fields]))
    dates, values = series(latitude, longitude)
    ones = np.ones_like(values)
    with lake.transaction():
        for f in fields:
            lake.put(f["id"], "obs", "mascon", dates, values, GRACE_MASCON.short_name)
            lake.put(f["id"], "obs", "mascon_w", dates, ones, GRACE_MASCON.short_name)
    log(f"mascon {len(dates)} months ok")
