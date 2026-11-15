"""Download one sample file per training product and regrid it to 1 km.

    uv run python -m nasa_farm.grid.run --aoi ames --date 2024-07-15
    uv run python -m nasa_farm.grid.run --only smap-l4,cdl,3dep
    uv run python -m nasa_farm.grid.run --list

Outputs (default `data/grid1km/<aoi>_<date>/`):
- `<product>.nc` — that product on the common grid (CF-1.10, EPSG:5070);
  point products also write `<product>_points.csv`;
- `stack.nc` — every successful product merged, vars named
  `<product>__<var>`;
- `report.md` / `report.json` — status per product (ok / skipped /
  failed), source granules, regrid method, coverage, timing, errors.
Raw downloads are cached in `data/raw/<product>/` (re-runs only regrid).

Default date 2024-07-15: mid-season, inside every product's record, and
outside the Planetary Computer MODIS hole (Jul–Oct 2025).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from nasa_farm.grid.catalog import ALTERNATES, REGISTRY, Product
from nasa_farm.grid.fetch import Context, Skip, available
from nasa_farm.grid.output import Out
from nasa_farm.grid.target import AOIS, TargetGrid

DEFAULT_DATE = dt.date(2024, 7, 15)


def _netcdf_engine() -> str | None:
    try:
        import h5netcdf  # noqa: F401
        import h5py  # noqa: F401  (h5netcdf imports without it, then fails on write)
        return "h5netcdf"
    except ImportError:
        pass
    try:
        import scipy  # noqa: F401  (NetCDF3 fallback, uncompressed)
        return "scipy"
    except ImportError:
        return None


def write(ds, path: Path) -> None:
    eng = _netcdf_engine()
    enc = {}
    if eng == "h5netcdf":
        enc = {v: {"compression": "gzip", "compression_opts": 4}
               for v in ds.data_vars if ds[v].dtype.kind == "f"}
    ds.to_netcdf(path, engine=eng, encoding=enc)


def run_one(p: Product, grid: TargetGrid, date: dt.date, out_dir: Path,
            raw: Path, creds: dict[str, bool]) -> dict:
    rec = {"product": p.key, "section": p.section, "title": p.title,
           "route": p.route, "route_confidence": p.conf, "status": "",
           "reason": "", "notes": [], "seconds": 0.0, "vars": [],
           "coverage": None, "methods": {}}
    t0 = time.monotonic()
    missing = [n for n in p.needs if not creds.get(n)]
    if missing:
        rec.update(status="skipped", reason=f"needs credential: {', '.join(missing)}")
        return rec
    context = Context(grid, date, raw, key=p.key)
    out = Out(grid)
    try:
        p.fn(context, out)
        if not out.vars:
            raise RuntimeError("product produced no variable")
        out.attrs.update(product=p.key, title=p.title, route=p.route,
                         requested_date=str(date), sources=" | ".join(context.notes))
        ds = out.to_dataset()
        write(ds, out_dir / f"{p.key}.nc")
        if out.points is not None:
            out.points.to_csv(out_dir / f"{p.key}_points.csv", index=False)
        finite = [float(np.isfinite(ds[v].values).mean()) for v in ds.data_vars]
        rec.update(status="ok", vars=list(ds.data_vars),
                   coverage=round(float(np.mean(finite)), 3),
                   methods={v: ds[v].attrs.get("method", "") for v in ds.data_vars})
    except Skip as e:
        rec.update(status="skipped", reason=str(e))
    except Exception as e:  # noqa: BLE001 — one product must never stop the run
        rec.update(status="failed", reason=f"{type(e).__name__}: {e}"[:600],
                   traceback=traceback.format_exc(limit=6)[-2000:])
    rec["notes"] = context.notes
    rec["seconds"] = round(time.monotonic() - t0, 1)
    return rec


def stack(out_dir: Path, recs: list[dict]) -> Path | None:
    import xarray as xr
    parts = []
    for r in recs:
        if r["status"] != "ok":
            continue
        with xr.open_dataset(out_dir / f"{r['product']}.nc",
                             engine=_netcdf_engine()) as ds:
            ds = ds.load()
        k = r["product"]
        dims = {d: f"{k}__{d}" for d in ds.dims if d not in ("x", "y")}
        ds = ds.rename({**{v: f"{k}__{v}" for v in ds.data_vars}, **dims})
        ds.attrs = {}
        parts.append(ds)
    if not parts:
        return None
    st = xr.merge(parts, compat="override", join="exact", combine_attrs="drop")
    st.attrs.update(Conventions="CF-1.10", products=len(parts))
    p = out_dir / "stack.nc"
    write(st, p)
    return p


def report(out_dir: Path, recs: list[dict], grid: TargetGrid, date: dt.date) -> None:
    (out_dir / "report.json").write_text(json.dumps(
        {"grid": grid.describe(), "date": str(date), "products": recs}, indent=1))
    n = {s: sum(r["status"] == s for r in recs) for s in ("ok", "skipped", "failed")}
    lines = [f"# 1 km sample build — {date}",
             "", f"Grid: {grid.describe()}", "",
             f"**{n['ok']} ok · {n['skipped']} skipped · {n['failed']} failed** "
             f"of {len(recs)} products", "",
             "| # | Product | Status | Route (conf.) | Vars | Coverage | s | Reason / source |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(recs, key=lambda r: (r["section"], r["product"])):
        why = (r["reason"] or "; ".join(r["notes"]))[:220].replace("|", "/")
        cov = "" if r["coverage"] is None else f"{r['coverage']:.0%}"
        lines.append(f"| {r['section']} | `{r['product']}` | {r['status']} | "
                     f"{r['route']} ({r['route_confidence']}) | {len(r['vars'])} | "
                     f"{cov} | {r['seconds']} | {why} |")
    lines += ["", "Alternates not downloaded (one file per CMR collection):", ""]
    lines += [f"- `{k}`: {v}" for k, v in ALTERNATES.items()]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aoi", default="ames", choices=sorted(AOIS))
    ap.add_argument("--date", type=dt.date.fromisoformat, default=DEFAULT_DATE)
    ap.add_argument("--res", type=float, default=1000.0, help="metres")
    ap.add_argument("--crs", default="EPSG:5070")
    ap.add_argument("--out", type=Path, default=Path("data/grid1km"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--only", default="", help="comma-separated product keys")
    ap.add_argument("--skip", default="", help="comma-separated product keys")
    ap.add_argument("--jobs", type=int, default=4, help="products in parallel")
    ap.add_argument("--list", action="store_true", help="list products and exit")
    a = ap.parse_args(argv)
    creds = available()
    if a.list:
        for p in sorted(REGISTRY.values(), key=lambda p: (p.section, p.key)):
            need = ",".join(p.needs) or "-"
            ok = all(creds.get(n) for n in p.needs)
            print(f"{p.section:>2} {p.key:<22} {'ready' if ok else 'needs ' + need:<18} "
                  f"[{p.conf}] {p.title}")
        return 0
    grid = TargetGrid.named(a.aoi, a.res, a.crs)
    only = {k for k in a.only.split(",") if k}
    skip = {k for k in a.skip.split(",") if k}
    unknown = (only | skip) - set(REGISTRY)
    if unknown:
        ap.error(f"unknown product(s): {sorted(unknown)}")
    prods = [p for k, p in REGISTRY.items() if (not only or k in only) and k not in skip]
    out_dir = a.out / f"{a.aoi}_{a.date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"grid {grid.describe()} · {len(prods)} products · credentials "
          f"{ {k: v for k, v in creds.items()} }", flush=True)
    with ThreadPoolExecutor(max_workers=max(a.jobs, 1)) as ex:
        futs = {ex.submit(run_one, p, grid, a.date, out_dir, a.raw, creds): p
                for p in prods}
        recs = []
        for f in futs:
            r = f.result()
            recs.append(r)
            print(f"  {r['status']:<8} {r['product']:<22} {r['seconds']:>6}s "
                  f"{r['reason'][:110]}", flush=True)
    report(out_dir, recs, grid, a.date)
    st = stack(out_dir, recs)
    print(f"report → {out_dir / 'report.md'}" + (f" · stack → {st}" if st else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
