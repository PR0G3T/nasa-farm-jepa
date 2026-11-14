"""Crawler — batch acquisition into the lake, resumable via catalog.

Two regimes, one sink:
- point APIs (POWER, SoilGrids): per-field fetches, raw dumps land under
  `forcing/` (+ `obs/sm` for GWETROOT) and `soil` attrs;
- granule products (SMAP, ECOSTRESS, NLDAS-2): searched once over the
  union bbox, each file downloaded once, every field point sampled out —
  O(fields) extraction per O(1) download — then marked in the catalog so
  re-runs skip them. Files are deleted after sampling.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import xarray as xr

from nasa_farm.acquire.base import BBox, Temporal
from nasa_farm.acquire.power import PowerClient
from nasa_farm.acquire.soilgrids import SoilGridsClient
from nasa_farm.lake import Catalog
from nasa_farm.lake import store


def _dates(years: list[int]) -> tuple[list[str], Temporal]:
    return ([f"{y}{m:02d}{d:02d}" for y in years
             for m in range(1, 13)
             for d in range(1, 32)
             if _valid(y, m, d)],
            Temporal(f"{years[0]}-01-01", f"{years[-1]}-12-31"))


def _valid(y: int, m: int, d: int) -> bool:
    try:
        dt.date(y, m, d)
        return True
    except ValueError:
        return False


def _power(fields: list[dict], root: Path, cat: Catalog,
           years: list[int], workers: int, log) -> None:
    client = PowerClient()
    for f in fields:
        raw = client.data_years(f["lat"], f["lon"], years, workers)
        keys, _ = _dates(years)
        days = [k for k in keys if not k.endswith("0229")]
        for param, series in zip(raw[0].keys(),
                                 zip(*[[d.get(k, np.nan) for k in days]
                                       for d in raw])):
            grp, var = ("obs", "sm") if param == "GWETROOT" \
                else ("forcing", param)
            n = store.put(root, f["id"], grp, var,
                          np.array(days, dtype=np.int32),
                          np.array(series, dtype=np.float32))
            cat.set_series(f["id"], grp, var,
                           int(days[0]), int(days[-1]), n, "power")
        log(f"power {f['id']} ok")


def _soil(fields: list[dict], root: Path, cat: Catalog, log) -> None:
    client = SoilGridsClient()
    for f in fields:
        prof = client.profile(f["lat"], f["lon"])
        store.attrs(root, f["id"], soil=json.loads(prof.to_json())
                    if hasattr(prof, "to_json") else prof.__dict__)
        log(f"soil {f['id']} ok")


@dataclass(frozen=True, slots=True)
class GranuleSpec:
    """One granule product → one obs channel."""
    source: object          # EarthdataSource
    obs: str                # canonical observe/ channel
    var: str                # xarray variable name
    group: str | None = None  # h5 group for multi-group products
    date: Callable[[str], int] = lambda gid: int(
        re.search(r"(?<!\d)(19|20)\d{6}(?!\d)|(?<!\d)(19|20)\d{2}0\d{2}(?!\d)",
                  gid).group())


SMAP_SM = GranuleSpec(
    source=__import__("nasa_farm.acquire.earthdata", fromlist=["SMAP_L4"]).SMAP_L4,
    obs="sm", var="soil_moisture",
    group="Soil_Moisture_Retrieval_Data_AM")
ECO_ET = GranuleSpec(
    source=__import__("nasa_farm.acquire.earthdata", fromlist=["ECOSTRESS_SEB"]).ECOSTRESS_SEB,
    obs="et", var="ETdaily")
ECO_LST = GranuleSpec(
    source=__import__("nasa_farm.acquire.earthdata", fromlist=["ECOSTRESS_MET"]).ECOSTRESS_MET,
    obs="lst", var="LST")

GRANULE_SPECS = {"smap": SMAP_SM, "ecostress-seb": ECO_ET,
                 "ecostress-met": ECO_LST}


def sample(ds: xr.Dataset, var: str, lat: float, lon: float) -> float:
    """Nearest-point sample — regular (lat/lon coords) or 2-D grids."""
    da = ds[var]
    lats, lons = (ds[k].values for k in ("lat", "lon") if k in ds)
    if lats.ndim == 1 and "lat" in da.dims:
        return float(da.sel(lat=lat, lon=lon, method="nearest").values)
    flat = np.nanargmin((lats - lat) ** 2 + (lons - lon) ** 2)
    return float(da.values.flat[flat])


def _granules(spec: GranuleSpec, fields: list[dict], root: Path,
              cat: Catalog, temporal: Temporal, log) -> None:
    aoi = BBox(min(f["lon"] for f in fields), min(f["lat"] for f in fields),
               max(f["lon"] for f in fields), max(f["lat"] for f in fields))
    results = spec.source.fetch(aoi, temporal, count=10_000)
    with tempfile.TemporaryDirectory() as tmp:
        for res in results:
            gid = str(getattr(res, "granule_ur", res))
            if cat.has_granule(gid):
                continue
            files = spec.source.fetch(aoi, temporal, count=1,
                                      download_dir=tmp) or []
            for path in files:
                day = spec.date(gid)
                kw = {"group": spec.group} if spec.group else {}
                with xr.open_dataset(path, engine="h5netcdf", **kw) as ds:
                    for f in fields:
                        v = sample(ds, spec.var, f["lat"], f["lon"])
                        if np.isfinite(v):
                            n = store.put(root, f["id"], "obs", spec.obs,
                                          np.array([day], np.int32),
                                          np.array([v], np.float32))
                            cat.set_series(f["id"], "obs", spec.obs,
                                           0, day, n, spec.source.short_name)
                Path(path).unlink(missing_ok=True)
            cat.mark_granule(gid, spec.source.short_name)
            log(f"{spec.obs} {gid} ok")


def crawl(fields: Iterable[dict], root: Path | str, years: list[int],
          sources: tuple[str, ...] = ("power", "soil"), workers: int = 8,
          log: Callable[[str], None] = print) -> None:
    root = Path(root)
    fields = [dict(f) for f in fields]
    _, temporal = _dates(years)
    with Catalog(root / "catalog.db") as cat:
        for f in fields:
            cat.field(f["id"], f["lat"], f["lon"], f.get("name", ""))
        if "power" in sources:
            _power(fields, root, cat, years, workers, log)
        if "soil" in sources:
            _soil(fields, root, cat, log)
        for name, spec in GRANULE_SPECS.items():
            if name in sources:
                _granules(spec, fields, root, cat, temporal, log)
