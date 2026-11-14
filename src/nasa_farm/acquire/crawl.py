"""Batch acquisition → lake: POWER + soil always; SMAP/ECOSTRESS/NLDAS
when configured.

Every source lands inside `Lake.tx()` units sized so a crash can never
leave data written but unmarked: one tx per POWER field, per NLDAS
field-month (marks replayed in the same tx as the writes they cover),
and per satellite granule (all field puts + its mark). `has_granule`
makes every granule-level source resumable; `put` merging makes POWER
re-crawls converge for free.
"""

from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path

import numpy as np

from nasa_farm.acquire.base import BBox, Temporal
from nasa_farm.acquire.power import CANON, PowerClient
from nasa_farm.acquire.soilgrid import SoilGridClient
from nasa_farm.lake import Lake

from .products import SPECS as GRANULE_SPECS


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
           workers: int, log, client: PowerClient | None = None) -> None:
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
    """SoilGrid profiles → field attrs (single-shot per field)."""
    client = client or SoilGridClient()
    for f in fields:
        d = client.sample(f["lat"], f["lon"])
        lo = dict(d, clay=d["clay_lo"], sand=d["sand_lo"], bd=d["bd_lo"])
        hi = dict(d, clay=d["clay_hi"], sand=d["sand_hi"], bd=d["bd_hi"])
        lake.attrs(f["id"], soil=d, soil_lo=lo, soil_hi=hi)
        log(f"soil {f['id']} ok")


def _nldas(fields: list[dict], lake: Lake, years: list[int], log) -> None:
    """NLDAS-2 CONUS forcing — one tx per field-month: writes + marks land
    together, so a crash mid-month re-downloads the month, never silently
    skips it."""
    from nasa_farm.acquire import nldas
    na = [f for f in fields if nldas.covers(f["lat"], f["lon"])]
    if not na:
        log("nldas: no fields in domain")
        return
    days = [dt.date(y, m, d)
            for y in years for m in range(1, 13)
            for d in range(1, (dt.date(y + (m == 12), m % 12 + 1, 1)
                               - dt.date(y, m, 1)).days + 1)]
    for (y, m), mdays in groupby(days, key=lambda d: (d.year, d.month)):
        pend: dict[str, dict[str, list]] = {f["id"]: {} for f in na}
        marked: list[str] = []
        for day in mdays:
            key = nldas.day_url(day)
            if lake.has_granule(key):
                continue
            try:
                ds = nldas.open_day(day, nldas._fs())
            except (OSError, KeyError, ValueError) as e:
                log(f"nldas {day} skip ({e})")
                continue
            di = int(day.strftime("%Y%m%d"))
            with ds:
                for f in na:
                    raw = {v: nldas.sample(ds, v, f["lat"], f["lon"])
                           for v in nldas.VARS if v in ds}
                    for var, val in nldas.convert(raw).items():
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


def _granule(work):
    spec, g, fields = work
    from nasa_farm.observe import fetch
    try:
        ds = fetch(spec.source, g, BBox(*spec.bbox), Temporal(spec.span))
    except (OSError, ValueError, KeyError) as e:
        return g.id, f"skip ({type(e).__name__})"
    with ds:
        day = spec.day_of(ds)
        return g.id, (day, {
            f["id"]: (float(spec.read(ds, f["lat"], f["lon"])),
                      float(spec.coverage(ds, f["lat"], f["lon"])))
            for f in fields
        })


def _granules(spec, fields: list[dict], lake: Lake,
              temporal: Temporal, log) -> None:
    """Satellite granules → obs series + coverage weights — one tx per
    granule (all field puts + the mark), so re-crawl never re-downloads."""
    results = spec.source.search(
        BBox(*spec.bbox), temporal, limit=spec.limit)
    todo = [g for g in results if not lake.has_granule(g.id)]
    log(f"{spec.obs}: {len(todo)}/{len(results)} new granules")
    if not todo:
        return
    def work(g): return _granule((spec, g, fields))
    with ThreadPoolExecutor(max_workers=spec.workers) as pool:
        for gid, got in pool.map(work, todo):
            if not isinstance(got, tuple):
                log(f"{spec.obs} {gid} {got}")
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


def crawl(fields: list[dict], root: Path | str,
          years: list[int] | None = None,
          sources: tuple[str, ...] = ("power", "soil"),
          workers: int = 8, log=print,
          power: PowerClient | None = None,
          soil: SoilGridClient | None = None) -> Path:
    """Fetch everything `sources` covers for `fields` into `lake.db`."""
    fields = [dict(f) for f in fields]
    years = years or [dt.date.today().year]
    temporal = Temporal(f"{min(years)}-01-01", f"{max(years)}-12-31")
    root = Path(root)
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
        for name, spec in GRANULE_SPECS.items():
            if name in sources:
                _granules(spec, fields, lake, temporal, log)
    return root


def load_fields(path: Path | str) -> list[dict]:
    """Field list JSON: [{id, lat, lon, name?}] — '-' reads stdin."""
    import sys
    raw = sys.stdin.read() if str(path) == "-" else Path(path).read_text()
    return json.loads(raw)
