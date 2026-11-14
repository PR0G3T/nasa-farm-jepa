"""Crawler — batch acquisition into the lake, resumable via granule marks.

Two regimes, one sink:
- point APIs (POWER, SoilGrid): per-field fetches land under `forcing/`
  (+ `obs/smg` for GWETROOT — a model proxy, not a retrieval) and the
  `soil` attr for profile-uncertainty sampling;
- granule products (SMAP, NISAR, ECOSTRESS, NLDAS-3): searched once over
  the field-union bbox, each file opened once by range-read, every field
  extracted by footprint-weighted sampling under the product's QA
  confidence map — O(fields) extraction per O(1) open — then marked so
  re-runs skip them. Each obs lands with a `{var}_w` coverage weight.

Every write lands inside `Lake.transaction()` units sized so a crash can never
leave data written but unmarked: one tx per POWER field, per NLDAS
field-month (marks commit with the writes they cover — a mid-month crash
re-downloads the month, never silently skips it), and per satellite
granule (all field puts + the mark). `put` is a no-op on unchanged
merges, so re-crawls write nothing.
"""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path
from typing import Callable, Iterable

import numpy as numpy

from nasa_farm.acquire.base import Bounds, Temporal
from nasa_farm.acquire.power import CANON, Power
from nasa_farm.acquire.soilgrid import SoilGrids
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
           workers: int, log,
           client: Power | None = None) -> None:
    """POWER daily point series per field — one tx per field."""
    client = client or Power()
    for f in fields:
        raw = client.data_years(f["lat"], f["lon"], years, workers)
        keys = _day_keys(years)
        dates = numpy.array(
            [int(k) for y in years for k in keys[y]], numpy.int32)
        with lake.transaction():
            for param in raw[0]:
                values = numpy.array(
                    [d.get(param, {}).get(k, numpy.nan)
                     for d, y in zip(raw, years) for k in keys[y]],
                    numpy.float32)
                group, var = ("obs", "smg") if param == "GWETROOT" \
                    else ("forcing", CANON.get(param, param))
                lake.put(f["id"], group, var, dates, values, "power")
        log(f"power {f['id']} ok")


def _soil(fields: list[dict], lake: Lake, log,
          client: SoilGrids | None = None) -> None:
    """SoilGrid flat dict (median + Q05/Q95 bounds) → `soil` attr."""
    client = client or SoilGrids()
    for f in fields:
        lake.attrs(f["id"], soil=client.sample(f["lat"], f["lon"]))
        log(f"soil {f['id']} ok")


def _nldas(fields: list[dict], lake: Lake, years: list[int], log) -> None:
    """NLDAS-3 daily forcing → canonical shards; one open per day feeds
    every in-domain field. Days accumulate in memory per field and flush
    with their marks in one tx per month — no read-modify-write per day."""
    from nasa_farm.acquire import reanalysis as reanalysis
    na = [f for f in fields if reanalysis.covers(f["lat"], f["lon"])]
    if not na:
        log("nldas: no fields in domain")
        return
    edited = reanalysis._fs()
    days = [dt.date(years[0], 1, 1) + dt.timedelta(i)
            for i in range((dt.date(years[-1], 12, 31)
                            - dt.date(years[0], 1, 1)).days + 1)]
    for (y, m), mdays in groupby(days, key=lambda d: (d.year, d.month)):
        pend: dict[str, dict[str, list]] = {f["id"]: {} for f in na}
        marked: list[str] = []
        for day in mdays:
            key = reanalysis.day_url(day)
            if lake.has_granule(key):
                continue
            try:
                ds = reanalysis.open_day(day, edited)
            except (OSError, KeyError, ValueError) as e:
                log(f"nldas {day} skip ({e})")
                continue
            di = int(day.strftime("%Y%m%d"))
            with ds:
                for f in na:
                    raw = {v: reanalysis.sample(ds, v, f["lat"], f["lon"])
                           for v in reanalysis.VARS if v in ds}
                    for var, val in reanalysis.convert(raw).items():
                        slot = pend[f["id"]].setdefault(var, ([], []))
                        slot[0].append(di)
                        slot[1].append(val)
            marked.append(key)
        if not marked:
            continue
        with lake.transaction():
            for fid, accumulate in pend.items():
                for var, (ds_, vs) in accumulate.items():
                    lake.put(fid, "forcing", var, numpy.array(ds_, numpy.int32),
                             numpy.array(vs, numpy.float32), "nldas3")
            for key in marked:
                lake.mark_granule(key, "nldas3")
        log(f"nldas {y}-{m:02d} ok")


def _aoi(specification, fields: list[dict]) -> Bounds | None:
    """Field-union bbox ∩ the product's domain — None when disjoint."""
    w = max(min(f["lon"] for f in fields), specification.bbox[0])
    s = max(min(f["lat"] for f in fields), specification.bbox[1])
    e = min(max(f["lon"] for f in fields), specification.bbox[2])
    n = min(max(f["lat"] for f in fields), specification.bbox[3])
    return None if w > e or s > n else Bounds(w, s, e, n)


def _granule(work):
    """Range-read one granule → {fid: (v, w)}. Pure I/O — the caller
    serializes lake writes."""
    import socket
    socket.setdefaulttimeout(300)
    specification, g, fields = work
    from nasa_farm.observe import fetch
    try:
        ds = fetch(specification.source, g)
        with ds:
            margin = specification.radius_km / 111.0 + 0.05
            box = (min(f["lon"] for f in fields) - margin,
                   min(f["lat"] for f in fields) - margin,
                   max(f["lon"] for f in fields) + margin,
                   max(f["lat"] for f in fields) + margin)
            if (grid := specification.prep(ds, box)) is None:
                return g.id, "skip (no var)"
            output = {}
            for f in fields:
                v, w = specification.at(grid, f["lat"], f["lon"])
                if numpy.isfinite(v) and w > 0.0:
                    output[f["id"]] = (v, w)
            return g.id, (specification.day_of(ds), output)
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        return g.id, f"skip ({type(e).__name__})"


def _granules(specification, fields: list[dict], lake: Lake,
              temporal: Temporal, log) -> None:
    """Satellite granules → obs series + coverage weights — one tx per
    granule (all field puts + the mark), so re-crawl never re-opens."""
    area, span = _aoi(specification, fields), Temporal(
        max(temporal.start, specification.span[0]),
        min(temporal.end, specification.span[1]))
    if area is None or span.start > span.end:
        log(f"{specification.observed}: fields outside product domain")
        return
    todo = [g for g in specification.source.search(area, span, specification.limit)
            if not lake.has_granule(g.id)]
    log(f"{specification.observed}: {len(todo)} granules to fetch")
    if not todo:
        return

    def work(g): return _granule((specification, g, fields))
    with ThreadPoolExecutor(max_workers=specification.workers) as pool:
        for gid, got in pool.map(work, todo):
            if not isinstance(got, tuple):
                log(f"{specification.observed} {gid} {got}")
                continue
            day, values = got
            di = numpy.array([day], numpy.int32)
            source = specification.source.short_name
            with lake.transaction():
                for fid, (v, w) in values.items():
                    lake.put(fid, "obs", specification.observed, di,
                             numpy.array([v], numpy.float32), source)
                    lake.put(fid, "obs", f"{specification.observed}_w", di,
                             numpy.array([w], numpy.float32), source)
                lake.mark_granule(gid, source)
            log(f"{specification.observed} {gid} ok")


def crawl(fields: Iterable[dict], root: Path | str, years: list[int],
          sources: tuple[str, ...] = ("power", "soil"), workers: int = 8,
          log: Callable[[str], None] = print,
          power: Power | None = None,
          soil: SoilGrids | None = None) -> None:
    """Fetch everything `sources` covers for `fields` into `lake.db`."""
    root, fields = Path(root), [dict(f) for f in fields]
    temporal = Temporal(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    root.mkdir(parents=True, exist_ok=True)
    with Lake(root) as lake:
        with lake.transaction():
            for f in fields:
                lake.field(f["id"], f["lat"], f["lon"], f.get("name", ""))
        if "power" in sources:
            _power(fields, lake, years, workers, log, power)
        if "soil" in sources:
            _soil(fields, lake, log, soil)
        if "nldas" in sources:
            _nldas(fields, lake, years, log)
        for name, specification in GRANULE_SPECS.items():
            if name in sources:
                _granules(specification, fields, lake, temporal, log)
