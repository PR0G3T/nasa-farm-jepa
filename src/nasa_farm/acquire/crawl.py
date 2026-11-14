"""Crawler — batch acquisition into the lake, resumable via catalog.

Two regimes, one sink:
- point APIs (POWER, SoilGrids): per-field fetches, raw dumps land under
  `forcing/` (+ `obs/smg` for GWETROOT — a model proxy, not a retrieval)
  and `soil`/`soil_lo`/`soil_hi` attrs for profile-uncertainty sampling;
- granule products (SMAP, NISAR, ECOSTRESS, NLDAS-2): searched once over
  the union bbox, each file downloaded once, every field extracted by
  footprint-weighted sampling under the product's QA confidence map —
  O(fields) extraction per O(1) download — then marked in the catalog so
  re-runs skip them. Each obs lands with a `{var}_w` coverage weight.
  Files are deleted after sampling.
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
    ECOSTRESS_MET, ECOSTRESS_SEB, EarthdataSource, NISAR_SME, SMAP_L4,
)
from nasa_farm.acquire.power import CANON, PowerClient
from nasa_farm.acquire.qc import FLAG_QC, SMAP_QC, QcSpec
from nasa_farm.acquire.soilgrids import SoilGridsClient
from nasa_farm.acquire.support import coverage, sample
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
            grp, var = ("obs", "smg") if param == "GWETROOT" \
                else ("forcing", CANON.get(param, param))
            n = store.put(root, f["id"], grp, var, dates, vals)
            cat.set_series(f["id"], grp, var,
                           int(dates[0]), int(dates[-1]), n, "power")
        log(f"power {f['id']} ok")


def _soil(fields: list[dict], root: Path, cat: Catalog, log,
          client: SoilGridsClient | None = None) -> None:
    client = client or SoilGridsClient()
    for f in fields:
        lo, mid, hi = client.bounds(f["lat"], f["lon"])
        store.attrs(root, f["id"], soil=dataclasses.asdict(mid),
                    soil_lo=dataclasses.asdict(lo),
                    soil_hi=dataclasses.asdict(hi))
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
    """One granule product → one obs channel.

    vars: candidate dataset names, resolved in order — products rename
    bands between versions; radius_km: nominal footprint for the
    extraction kernel (0 = nearest); qc: QA-band confidence map.
    """
    source: EarthdataSource
    obs: str
    vars: tuple[str, ...]
    group: str | None = None
    radius_km: float = 0.0
    qc: QcSpec | None = None


SMAP_SM = GranuleSpec(
    SMAP_L4, obs="sm", vars=("soil_moisture",),
    group="Soil_Moisture_Retrieval_Data_AM", radius_km=18.0, qc=SMAP_QC)
NISAR_SM = GranuleSpec(
    NISAR_SME, obs="sme2", radius_km=0.2, qc=FLAG_QC,
    vars=("soil_moisture", "soilMoisture", "soil_moisture_pmi",
          "PMI", "DSG", "TSR"))
ECO_ET = GranuleSpec(ECOSTRESS_SEB, obs="et", vars=("ETdaily",),
                     radius_km=0.5, qc=FLAG_QC)
ECO_LST = GranuleSpec(ECOSTRESS_MET, obs="lst", vars=("LST",),
                      radius_km=0.5, qc=FLAG_QC)
GRANULE_SPECS = {"smap": SMAP_SM, "nisar": NISAR_SM,
                 "ecostress-seb": ECO_ET, "ecostress-met": ECO_LST}


def _resolve(ds: xr.Dataset, vars: tuple[str, ...]) -> str | None:
    return next((v for v in vars if v in ds), None)


def _nldas(fields: list[dict], root: Path, cat: Catalog,
           years: list[int], log) -> None:
    """NLDAS-3 daily forcing → canonical shards; one open per day feeds
    every in-domain field (bbox slice per var, nearest in-memory)."""
    from nasa_farm.acquire import nldas
    na = [f for f in fields if nldas.covers(f["lat"], f["lon"])]
    if not na:
        log("nldas: no fields in domain")
        return
    fs = nldas._fs()
    days = [dt.date(years[0], 1, 1) + dt.timedelta(i)
            for i in range((dt.date(years[-1], 12, 31)
                            - dt.date(years[0], 1, 1)).days + 1)]
    for day in days:
        key = nldas.day_url(day)
        if cat.has_granule(key):
            continue
        try:
            ds = nldas.open_day(day, fs)
        except (OSError, KeyError, ValueError) as e:
            log(f"nldas {day} skip ({e})")
            continue
        with ds:
            for f in na:
                raw = {v: nldas.sample(ds, v, f["lat"], f["lon"])
                       for v in nldas.VARS if v in ds}
                di = np.array([int(day.strftime("%Y%m%d"))], np.int32)
                for var, val in nldas.convert(raw).items():
                    store.put(root, f["id"], "forcing", var, di,
                              np.array([val], np.float32))
        cat.mark_granule(key, "nldas3")
        log(f"nldas {day} ok")


def _granule(res, spec: GranuleSpec, fields: list[dict], tmp: str
             ) -> tuple[int, dict[str, tuple[float, float]]] | None:
    """Download + extract one granule → {fid: (v, w)}. Pure I/O — the
    caller serializes lake writes."""
    import earthaccess
    gid = _granule_id(res)
    out = {}
    for path in earthaccess.download([res], tmp) or []:
        day = _granule_date(gid)
        kw = {"group": spec.group} if spec.group else {}
        with xr.open_dataset(path, engine="h5netcdf", **kw) as ds:
            var = _resolve(ds, spec.vars)
            if var is None:
                return None
            conf = spec.qc.mask(ds) if spec.qc else None
            for f in fields:
                v = sample(ds, var, f["lat"], f["lon"],
                           spec.radius_km, conf)
                w = coverage(ds, var, f["lat"], f["lon"],
                             spec.radius_km, conf)
                if np.isfinite(v) and w > 0.0:
                    out[f["id"]] = (v, w)
        Path(path).unlink(missing_ok=True)
    return day, out


def _granules(spec: GranuleSpec, fields: list[dict], root: Path,
              cat: Catalog, temporal: Temporal, log,
              workers: int = 4) -> None:
    import earthaccess
    earthaccess.login(strategy="environment")
    aoi = BBox(min(f["lon"] for f in fields), min(f["lat"] for f in fields),
               max(f["lon"] for f in fields), max(f["lat"] for f in fields))
    results = [r for r in spec.source.fetch(aoi, temporal, count=10_000)
               if not cat.has_granule(_granule_id(r))]
    log(f"{spec.obs}: {len(results)} granules to fetch")
    from concurrent.futures import ThreadPoolExecutor

    def work(res):
        with tempfile.TemporaryDirectory() as tmp:
            return _granule_id(res), _granule(res, spec, fields, tmp)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for gid, got in pool.map(work, results):
            if got is None:
                log(f"{spec.obs} {gid} skip (no var)")
                continue
            day, vals = got
            di = np.array([day], np.int32)
            for fid, (v, w) in vals.items():
                n = store.put(root, fid, "obs", spec.obs, di,
                              np.array([v], np.float32))
                store.put(root, fid, "obs", f"{spec.obs}_w", di,
                          np.array([w], np.float32))
                cat.set_series(fid, "obs", spec.obs, 0, day, n,
                               spec.source.short_name)
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
        if "nldas" in sources:
            _nldas(fields, root, cat, years, log)
        for name, spec in GRANULE_SPECS.items():
            if name in sources:
                _granules(spec, fields, root, cat, temporal, log)
