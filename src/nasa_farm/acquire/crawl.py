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

import dataclasses
import datetime as dt
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import xarray as xr

from nasa_farm.acquire.base import BBox, Temporal
from nasa_farm.acquire.earthdata import (
    ECOSTRESS_MET, ECOSTRESS_SEB, EarthdataSource, SMAP_L4,
)
from nasa_farm.acquire.power import PowerClient
from nasa_farm.acquire.soilgrids import SoilGridsClient
from nasa_farm.lake import Catalog, store


def _day_keys(years: list[int]) -> dict[int, list[str]]:
    out = {}
    for y in years:
        nd = (dt.date(y, 12, 31) - dt.date(y, 1, 1)).days + 1
        out[y] = [(dt.date(y, 1, 1) + dt.timedelta(d)).strftime("%Y%m%d")
                  for d in range(nd)
                  if not (dt.date(y, 1, 1) + dt.timedelta(d)).day == 29
                  or (dt.date(y, 1, 1) + dt.timedelta(d)).month != 2]
    return out


def _power(fields: list[dict], root: Path, cat: Catalog,
           years: list[int], workers: int, log,
           client: PowerClient | None = None) -> None:
    client = client or PowerClient()
    for f in fields:
        raw = client.data_years(f["lat"], f["lon"], years, workers)
        keys = _day_keys(years)
        for param in raw[0]:
            dates = np.array([k for y in years for k in keys[y]], np.int32)
            vals = np.array([d.get(param, {}).get(k, np.nan)
                             for d, y in zip(raw, years) for k in keys[y]],
                            np.float32)
            grp, var = ("obs", "sm") if param == "GWETROOT" \
                else ("forcing", param)
            n = store.put(root, f["id"], grp, var, dates, vals)
            cat.set_series(f["id"], grp, var,
                           int(dates[0]), int(dates[-1]), n, "power")
        log(f"power {f['id']} ok")


def _soil(fields: list[dict], root: Path, cat: Catalog, log,
          client: SoilGridsClient | None = None) -> None:
    client = client or SoilGridsClient()
    for f in fields:
        store.attrs(root, f["id"],
                    soil=dataclasses.asdict(client.profile(f["lat"], f["lon"])))
        log(f"soil {f['id']} ok")


def _granule_date(gid: str) -> int:
    """yyyymmdd int from a granule UR — ISO date or yyyyddd day-of-year."""
    for m in re.finditer(r"(?<!\d)(19|20)\d{6}(?!\d)", gid):
        return int(m.group())
    m = re.search(r"(?<!\d)((19|20)\d{2})(\d{3})(?!\d)", gid)
    d = dt.date(int(m[1]), 1, 1) + dt.timedelta(int(m[3]) - 1)
    return int(d.strftime("%Y%m%d"))


def _granule_id(res) -> str:
    for k in ("GranuleUR", "ProducerGranuleId"):
        try:
            return str(res["umm"][k])
        except (KeyError, TypeError):
            continue
    return str(res)


@dataclass(frozen=True, slots=True)
class GranuleSpec:
    """One granule product → one obs channel."""
    source: EarthdataSource
    obs: str
    var: str
    group: str | None = None


SMAP_SM = GranuleSpec(SMAP_L4, obs="sm", var="soil_moisture",
                      group="Soil_Moisture_Retrieval_Data_AM")
ECO_ET = GranuleSpec(ECOSTRESS_SEB, obs="et", var="ETdaily")
ECO_LST = GranuleSpec(ECOSTRESS_MET, obs="lst", var="LST")
GRANULE_SPECS = {"smap": SMAP_SM, "ecostress-seb": ECO_ET,
                 "ecostress-met": ECO_LST}


def sample(ds: xr.Dataset, var: str, lat: float, lon: float) -> float:
    """Nearest-point sample — regular (lat/lon dims) or 2-D grids."""
    da = ds[var]
    if "lat" in da.dims and "lon" in da.dims:
        return float(da.sel(lat=lat, lon=lon, method="nearest").values)
    lats, lons = ds["lat"].values, ds["lon"].values
    flat = np.nanargmin((lats - lat) ** 2 + (lons - lon) ** 2)
    return float(da.values.flat[flat])


def _granules(spec: GranuleSpec, fields: list[dict], root: Path,
              cat: Catalog, temporal: Temporal, log) -> None:
    import earthaccess
    aoi = BBox(min(f["lon"] for f in fields), min(f["lat"] for f in fields),
               max(f["lon"] for f in fields), max(f["lat"] for f in fields))
    results = spec.source.fetch(aoi, temporal, count=10_000)
    with tempfile.TemporaryDirectory() as tmp:
        for res in results:
            gid = _granule_id(res)
            if cat.has_granule(gid):
                continue
            earthaccess.login(strategy="environment")
            for path in earthaccess.download([res], tmp) or []:
                day = _granule_date(gid)
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
          log: Callable[[str], None] = print,
          power: PowerClient | None = None,
          soil: SoilGridsClient | None = None) -> None:
    root, fields = Path(root), [dict(f) for f in fields]
    temporal = Temporal(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    with Catalog(root / "catalog.db") as cat:
        for f in fields:
            cat.field(f["id"], f["lat"], f["lon"], f.get("name", ""))
        if "power" in sources:
            _power(fields, root, cat, years, workers, log, power)
        if "soil" in sources:
            _soil(fields, root, cat, log, soil)
        for name, spec in GRANULE_SPECS.items():
            if name in sources:
                _granules(spec, fields, root, cat, temporal, log)
