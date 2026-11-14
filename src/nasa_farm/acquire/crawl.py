"""Crawler — batch acquisition into the lake, resumable via granule marks.

Two regimes, one sink:
- point APIs (POWER, SoilGrids): per-field fetches, raw dumps land under
  `forcing/` (+ `obs/smg` for GWETROOT — a model proxy, not a retrieval)
  and `soil`/`soil_lo`/`soil_hi` attrs for profile-uncertainty sampling;
- granule products (SMAP, NISAR, ECOSTRESS): searched once over the
  union bbox, only the needed asset files fetched (HDF5 streamed in place
  when `Spec.stream`), every field extracted by
  footprint-weighted sampling under the product's QA confidence map —
  O(fields) extraction per O(1) download — then marked so re-runs skip
  them. Each obs lands with a `{var}_w` coverage weight. Files are
  deleted after sampling.

Every write lands inside `Lake.tx()` units sized so a crash can never
leave data written but unmarked: one tx per POWER field, per NLDAS
field-month (marks commit with the writes they cover — a mid-month crash
re-downloads the month, never silently skips it), and per satellite
granule (all field puts + the mark). `put` is a no-op on unchanged
merges, so re-crawls write nothing.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import xarray as xr

from nasa_farm.acquire.base import BBox, Temporal
from nasa_farm.acquire.earthdata import (
    granule_day, granule_id, login, umm_day,
)
from nasa_farm.acquire.power import CANON, PowerClient
from nasa_farm.acquire.products import (
    ALIASES, SPECS as GRANULE_SPECS, Spec,
)
from nasa_farm.acquire.soilgrid import SoilGridClient
from nasa_farm.lake import Lake


def _day_keys(years: list[int]) -> dict[int, list[str]]:
    """yyyymmdd keys per calendar year, Feb 29 dropped."""
    return {
        y: [d.strftime("%Y%m%d") for d in (
            dt.date(y, 1, 1) + dt.timedelta(i)
            for i in range((dt.date(y, 12, 31) - dt.date(y, 1, 1)).days + 1))
            if not (d.month == 2 and d.day == 29)]
        for y in years
    }


def _power(fields: list[dict], lake: Lake, years: list[int],
           workers: int, log,
           client: PowerClient | None = None) -> None:
    """POWER daily point series per field — one tx per field."""
    client = client or PowerClient()
    for f in fields:
        raw = client.data_years(f["lat"], f["lon"], years, workers)
        keys = _day_keys(years)
        dates = np.array(
            [int(k) for y in years for k in keys[y]], np.int32)
        with lake.tx():
            for param in raw[0]:
                vals = np.array(
                    [d.get(param, {}).get(k, np.nan)
                     for d, y in zip(raw, years) for k in keys[y]],
                    np.float32)
                grp, var = ("obs", "smg") if param == "GWETROOT" \
                    else ("forcing", CANON.get(param, param))
                lake.put(f["id"], grp, var, dates, vals, "power")
        log(f"power {f['id']} ok")


def _soil(fields: list[dict], lake: Lake, log,
          client: SoilGridClient | None = None) -> None:
    """SoilGrids Q05/Q50/Q95 profiles → field attrs (profile dicts)."""
    client = client or SoilGridClient()
    for f in fields:
        lo, mid, hi = client.bounds(f["lat"], f["lon"])
        lake.attrs(f["id"], soil=dataclasses.asdict(mid),
                   soil_lo=dataclasses.asdict(lo),
                   soil_hi=dataclasses.asdict(hi))
        log(f"soil {f['id']} ok")


def _nldas(fields: list[dict], lake: Lake, years: list[int], log) -> None:
    """NLDAS-3 daily forcing → canonical shards; one open per day feeds
    every in-domain field. Days accumulate in memory per field and flush
    with their marks in one tx per month — no read-modify-write per day."""
    from nasa_farm.acquire import nldas
    na = [f for f in fields if nldas.covers(f["lat"], f["lon"])]
    if not na:
        log("nldas: no fields in domain")
        return
    fs = nldas._fs()
    days = [dt.date(years[0], 1, 1) + dt.timedelta(i)
            for i in range((dt.date(years[-1], 12, 31)
                            - dt.date(years[0], 1, 1)).days + 1)]
    for (y, m), mdays in groupby(days, key=lambda d: (d.year, d.month)):
        pend: dict[str, dict[str, list]] = {f["id"]: {} for f in na}
        marked: list[str] = []
        for day in mdays:
            key = nldas.day_url(day)
            if lake.has_granule(key):
                continue
            try:
                ds = nldas.open_day(day, fs)
            except (OSError, KeyError, ValueError) as e:
                log(f"nldas {day} skip ({e})")
                continue
            di = int(day.strftime("%Y%m%d"))
            with ds:
                rate = nldas.rain_is_rate(ds)
                for f in na:
                    raw = {v: nldas.sample(ds, v, f["lat"], f["lon"])
                           for v in nldas.VARS if v in ds}
                    for var, val in nldas.convert(raw, rate).items():
                        slot = pend[f["id"]].setdefault(var, ([], []))
                        slot[0].append(di)
                        slot[1].append(val)
            marked.append(key)
        if not marked:
            continue
        with lake.tx():
            for fid, acc in pend.items():
                for var, (ds_, vs) in acc.items():
                    lake.put(fid, "forcing", var, np.array(ds_, np.int32),
                             np.array(vs, np.float32), "nldas3")
            for key in marked:
                lake.mark_granule(key, "nldas3")
        log(f"nldas {y}-{m:02d} ok")


def _open_h5(src, spec: Spec) -> xr.Dataset:
    """`src`: local path or file-like (streamed granule). SMAP HDF5 has no
    dimension scales → phony dims; L4 keeps lat/lon in the root group."""
    g = spec.source.group
    ds = xr.open_dataset(src, engine="h5netcdf", phony_dims="access",
                         **({"group": g} if g else {}))
    if spec.root_coords:
        var = spec.var(ds) or next(iter(ds.data_vars))
        with xr.open_dataset(src, engine="h5netcdf",
                             phony_dims="access") as root:
            for c in spec.root_coords:
                ds[c] = (ds[var].dims, root[c].values)
    return ds


def _open_cog(paths: list[str]) -> xr.Dataset:
    """One GeoTIFF per variable on the same tile grid → Dataset with 2-D
    lat/lon, so `support.sample` works unchanged. Var name = file suffix
    (`…_LST.tif` → `LST`)."""
    import rioxarray  # noqa: F401  (registers .rio)
    from pyproj import Transformer
    arrays = {Path(p).stem.rsplit("_", 1)[-1]:
              rioxarray.open_rasterio(p, masked=True).squeeze("band", drop=True)
              for p in paths}
    ref = next(iter(arrays.values()))
    xx, yy = np.meshgrid(ref.x.values, ref.y.values)
    lon, lat = Transformer.from_crs(ref.rio.crs, 4326,
                                    always_xy=True).transform(xx, yy)
    ds = xr.Dataset({k: (("y", "x"), np.asarray(a.values, np.float64))
                     for k, a in arrays.items()})
    ds["lat"], ds["lon"] = (("y", "x"), lat), (("y", "x"), lon)
    return ds


def _links(res, spec: Spec) -> list[str]:
    """Data URLs of the granule restricted to the assets the spec reads."""
    try:
        urls = res.data_links()
    except (AttributeError, TypeError):
        return []
    return [u for u in urls if u.endswith(spec.assets)]


def _granule(res, spec: Spec, fields: list[dict], tmp: str
             ) -> tuple[int, dict[str, tuple[float, float]]] | str:
    """Fetch + extract one granule → (day, {fid: (v, w)}), or a skip
    reason. Pure I/O — the caller serializes lake writes."""
    import earthaccess
    gid = granule_id(res)
    day = granule_day(gid) or umm_day(res)
    urls = _links(res, spec)
    if day is None or not urls:
        return f"date={day}, files={len(urls)}"
    paths: list[str] = []
    try:
        if spec.stream and spec.reader == "h5":
            ds = _open_h5((earthaccess.open(urls) or [None])[0], spec)
        else:
            paths = [str(p) for p in earthaccess.download(urls, tmp) or []]
            if not paths:
                return "download failed"
            ds = _open_cog(paths) if spec.reader == "cog" \
                else _open_h5(paths[0], spec)
    except (OSError, ValueError, KeyError, TypeError) as e:
        return f"read: {e}"
    out = {}
    with ds:
        if spec.var(ds) is None:
            return f"no var {spec.vars}"
        for f in fields:
            v = spec.read(ds, f["lat"], f["lon"])
            w = spec.coverage(ds, f["lat"], f["lon"]) \
                if spec.radius_km > 0 else float(np.isfinite(v))
            if np.isfinite(v) and w > 0.0:
                out[f["id"]] = (v, w)
    for p in paths:
        Path(p).unlink(missing_ok=True)
    return day, out


def _granules(spec: Spec, fields: list[dict], lake: Lake,
              temporal: Temporal, log, workers: int = 4) -> None:
    """Satellite granules → obs series + coverage weights — one tx per
    granule (all field puts + the mark), so re-crawl never re-downloads."""
    login()
    aoi = BBox(min(f["lon"] for f in fields), min(f["lat"] for f in fields),
               max(f["lon"] for f in fields), max(f["lat"] for f in fields))
    pick = re.compile(spec.pick) if spec.pick else None
    results = [r for r in spec.source.fetch(aoi, temporal, count=spec.limit)
               if not lake.has_granule(gid := granule_id(r))
               and (pick is None or pick.search(gid))]
    log(f"{spec.obs}: {len(results)} granules to fetch")
    if not results:
        return

    def work(res):
        with tempfile.TemporaryDirectory() as tmp:
            return granule_id(res), _granule(res, spec, fields, tmp)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for gid, got in pool.map(work, results):
            if isinstance(got, str):
                log(f"{spec.obs} {gid} skip ({got})")
                continue
            day, vals = got
            di = np.array([day], np.int32)
            src = spec.source.short_name
            with lake.tx():
                for fid, (v, w) in vals.items():
                    lake.put(fid, "obs", spec.obs, di,
                             np.array([v], np.float32), src)
                    lake.put(fid, "obs", f"{spec.obs}_w", di,
                             np.array([w], np.float32), src)
                lake.mark_granule(gid, src)
            log(f"{spec.obs} {gid} ok")


def crawl(fields: Iterable[dict], root: Path | str, years: list[int],
          sources: tuple[str, ...] = ("power", "soil"), workers: int = 8,
          log: Callable[[str], None] = print,
          power: PowerClient | None = None,
          soil: SoilGridClient | None = None) -> None:
    """Fetch everything `sources` covers for `fields` into `lake.db`."""
    root, fields = Path(root), [dict(f) for f in fields]
    temporal = Temporal(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    root.mkdir(parents=True, exist_ok=True)
    with Lake(root) as lake:
        with lake.tx():
            for f in fields:
                lake.field(f["id"], f["lat"], f["lon"], f.get("name", ""))
        if "power" in sources:
            _power(fields, lake, years, workers, log, power)
        if "soil" in sources:
            _soil(fields, lake, log, soil)
        if "nldas" in sources:
            _nldas(fields, lake, years, log)
        wanted = {ALIASES.get(s, s) for s in sources}
        for name, spec in GRANULE_SPECS.items():
            if name in wanted:
                _granules(spec, fields, lake, temporal, log)
