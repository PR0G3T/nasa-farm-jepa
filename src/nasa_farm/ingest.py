"""Planetary Computer ingest — scenes + parcel observations → Postgres.

Incremental: only scenes missing from `observations` hit the TiTiler
statistics API, so a re-run over the same window costs one STAC search.
Landsat `lwir11` digital numbers convert to °C in-flight; physically
impossible days (partial cloud swaths) are dropped.

`bake` renders the same webp tiles the app serves statically — grayscale
item tiles colorized with the app's exact ramps.
"""

from __future__ import annotations

import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
DATA = "https://planetarycomputer.microsoft.com/api/data/v1"

EXPR = {
    "ndvi": "(B08-B04)/(B08+B04)",
    "ndmi": "(B08-B11)/(B08+B11)",
}
KELVIN = lambda v: round(v * 0.00341802 + 149 - 273.15, 4)
DN_LO = (15 + 273.15 - 149) / 0.00341802
DN_HI = (45 + 273.15 - 149) / 0.00341802

RAMPS = {
    "ndvi": ["#d00000", "#f07000", "#e8bf1f", "#8ac248", "#2ca25f", "#006d2c"],
    "ndmi": ["#c2181b", "#f06a1d", "#f2b01e", "#2ab0a5", "#1d7ec4", "#08306b"],
    "lst": ["#24479e", "#2b7fc0", "#1fb0a6", "#f2b01e", "#e85a24", "#a50026"],
}

POWER_VARS = ("T2M_MIN", "T2M_MAX", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR",
              "RH2M", "WS2M", "GWETROOT")


def centroid(feature: dict) -> tuple[float, float]:
    """Mean of the exterior ring — fine at POWER/NLDAS resolution."""
    ring = feature["geometry"]["coordinates"][0]
    n = len(ring) - 1 or 1
    return (sum(p[0] for p in ring) / n, sum(p[1] for p in ring) / n)


def _pivot(data: dict[str, dict[str, float]]) -> list[tuple]:
    """POWER {param: {yyyymmdd: v}} → (date, tmin…gwetroot) rows."""
    days = sorted(next(iter(data.values()), {}))
    out = []
    for day in days:
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        vals = [data.get(p, {}).get(day) for p in POWER_VARS]
        out.append((iso, *[None if v != v else v for v in vals]))
    return out


def forcing(features: list[dict], years: int, connection, log) -> int:
    """POWER daily → forcing_daily; backfills missing years, tails the
    current one. Serving needs the N last *complete* years plus the live
    year, so the window is ``today.year - years .. today.year``."""
    import datetime as dt

    from nasa_farm.acquire.base import Temporal
    from nasa_farm.acquire.power import Power
    from nasa_farm import store

    client, made = Power(), 0
    today = dt.date.today()
    wanted = set(range(today.year - years, today.year + 1))
    for feature in features:
        fid = str(feature["properties"]["id"])
        lon, lat = centroid(feature)
        present = store.forcing_years(connection, fid, "nasa-power")
        blocks = [client.data_year(lat, lon, y)
                  for y in sorted(wanted - present)]
        if today.year in present:
            last = store.forcing_max(connection, fid, "nasa-power")
            start = (dt.date.fromisoformat(str(last)) + dt.timedelta(days=1)
                     if last else dt.date(today.year, 1, 1))
            if start <= today:
                blocks.append(client.fetch(
                    None, Temporal(str(start), str(today)),
                    latitude=lat, longitude=lon))
        if not blocks:
            continue
        rows = [row for data in blocks for row in _pivot(data)]
        store.put_forcing(connection, fid, "nasa-power", rows)
        connection.commit()
        made += len(rows)
        log(f"forcing {fid} {len(rows)} days")
    return made


def soil(features: list[dict], connection, log) -> int:
    """SoilGrids median + PI → soil_profile, once per field."""
    from nasa_farm.acquire.soilgrid import SoilGrids
    from nasa_farm import store

    client, made = SoilGrids(), 0
    for feature in features:
        fid = str(feature["properties"]["id"])
        if store.has_soil(connection, fid):
            continue
        lon, lat = centroid(feature)
        store.put_soil(connection, fid, client.sample(lat, lon))
        connection.commit()
        made += 1
        log(f"soil {fid}")
    return made


def rotation(features: list[dict], connection) -> int:
    """Feature kinds/sequence arrays → rotation(field, year)."""
    import datetime as dt

    from nasa_farm import store

    made = 0
    for feature in features:
        fid = str(feature["properties"]["id"])
        props = feature["properties"]
        kinds = props.get("kinds") or []
        labels = props.get("sequence") or []
        first = dt.date.today().year - len(kinds)
        rows = [(first + i, kinds[i], labels[i] if i < len(labels) else None)
                for i in range(len(kinds))]
        store.put_rotation(connection, fid, rows)
        made += len(rows)
    connection.commit()
    return made


def _fetch(url: str, body: bytes | None = None, tries: int = 4):
    for i in range(tries):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"content-type": "application/json"} if body else {})
            return urllib.request.urlopen(req, timeout=90).read()
        except urllib.error.HTTPError as e:
            if e.code < 500 or i == tries - 1:
                return None
            time.sleep(1.2 * (i + 1))
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(1.2 * (i + 1))


def scenes(collection: str, bbox: list[float], since: str, until: str,
           query: dict) -> dict[str, tuple[str, float]]:
    """Lowest-cloud item id per acquired date."""
    body = json.dumps({
        "collections": [collection], "bbox": bbox,
        "datetime": f"{since}T00:00:00Z/{until}T23:59:59Z",
        "query": query, "limit": 400}).encode()
    payload = _fetch(STAC + "/search", body)
    best: dict[str, tuple[str, float]] = {}
    for feature in json.loads(payload).get("features", []):
        if collection == "sentinel-2-l2a" and "T15TVG" not in feature["id"]:
            continue
        date = feature["properties"]["datetime"][:10]
        cloud = feature["properties"].get("eo:cloud_cover", 0)
        if date not in best or cloud < best[date][1]:
            best[date] = (feature["id"], cloud)
    return dict(sorted(best.items()))


def _stat(collection: str, item: str, params: str, key: str, feature: dict):
    url = f"{DATA}/item/statistics?collection={collection}&item={item}&{params}"
    payload = _fetch(url, json.dumps(
        {"type": "FeatureCollection", "features": [feature]}).encode())
    if not payload:
        return None
    try:
        table = (json.loads(payload)["features"][0]["properties"]
                 .get("statistics", {}) or {})
        cell = table.get(key) or next(iter(table.values()), {})
        if not cell or cell.get("mean") is None:
            return None
        hist = cell.get("histogram")

        def pct(q: float) -> float:
            if not hist:
                return cell["mean"]
            counts, edges = hist
            total = sum(counts)
            seen = 0.0
            for i, c in enumerate(counts):
                seen += c
                if seen / total >= q:
                    w = edges[1] - edges[0]
                    return edges[i] + w * max(
                        0, (q - (seen - c) / total) / (c / total if c else 1))
            return edges[-1]

        return {"mean": cell["mean"], "p10": pct(0.1), "p90": pct(0.9),
                "n": int(cell.get("count", 0))}
    except Exception:
        return None


def _lut(hexes: list[str]) -> list[tuple[int, int, int]]:
    stops = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in hexes]
    out = []
    for k in range(256):
        p = k / 255 * (len(stops) - 1)
        i = min(int(p), len(stops) - 2)
        t = p - i
        out.append(tuple(round(a + (b - a) * t)
                         for a, b in zip(stops[i], stops[i + 1])))
    return out


def _grid(bbox: list[float], zmin: int, zmax: int):
    """Web-mercator tile coords covering bbox."""
    lon_min, lat_min, lon_max, lat_max = bbox

    def xy(lon, lat, z):
        n = 2 ** z
        x = int((lon + 180) / 360 * n)
        y = int((1 - math.log(math.tan(math.radians(lat))
                 + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
        return x, y

    for z in range(zmin, zmax + 1):
        x0, y1 = xy(lon_min, lat_min, z)
        x1, y0 = xy(lon_max, lat_max, z)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                yield z, x, y


def _tile_query(layer: str):
    if layer == "ndvi":
        expr = urllib.parse.quote(EXPR["ndvi"], safe="")
        return f"expression={expr}&rescale=0,0.85&asset_as_band=true"
    if layer == "ndmi":
        expr = urllib.parse.quote(EXPR["ndmi"], safe="")
        return f"expression={expr}&rescale=-0.35,0.55&asset_as_band=true"
    if layer == "lst":
        return (f"assets=lwir11&rescale={DN_LO:.0f},{DN_HI:.0f}"
                "&asset_as_band=true")
    return "assets=visual"


def bake(layer: str, date: str, item: str, collection: str, root: Path,
         bbox: list[float], zmax: int, log) -> int:
    from PIL import Image

    out_dir = Path(root) / layer / date
    grid = list(_grid(bbox, 8, zmax))
    table = _lut(RAMPS[layer]) if layer in RAMPS else None
    made = 0

    def one(zxy) -> int:
        z, x, y = zxy
        path = out_dir / str(z) / str(x) / f"{y}.webp"
        if path.exists():
            return 0
        url = (f"{DATA}/item/tiles/WebMercatorQuad/{z}/{x}/{y}.png"
               f"?collection={collection}&item={item}&{_tile_query(layer)}")
        raw = _fetch(url, tries=3)
        if not raw:
            return 0
        try:
            im = Image.open(io.BytesIO(raw))
            if table is not None:
                gray, alpha = im.convert("LA").split()
                pal = gray.convert("P")
                pal.putpalette([c for rgb in table for c in rgb])
                im = Image.merge("RGBA", (*pal.convert("RGB").split(), alpha))
            else:
                im = im.convert("RGBA")
            path.parent.mkdir(parents=True, exist_ok=True)
            im.save(path, "WEBP")
            return 1
        except Exception:
            return 0

    with ThreadPoolExecutor(8) as pool:
        made = sum(pool.map(one, grid))
    log(f"baked {layer} {date} {made}/{len(grid)}")
    return made


def ingest(fields: list[dict], days: int, tiles: str | None, bbox: list[float],
           connect, log, years: int = 5) -> dict:
    """fields: geojson features. connect: lazy DB connector."""
    from nasa_farm import store

    since = time.strftime("%Y-%m-%d", time.gmtime(time.time() - days * 86400))
    until = time.strftime("%Y-%m-%d", time.gmtime())
    s2 = scenes("sentinel-2-l2a", bbox, since, until,
                {"eo:cloud_cover": {"lt": 60}})
    ls = scenes("landsat-c2-l2", bbox, since, until,
                {"eo:cloud_cover": {"lt": 60}, "landsat:wrs_type": {"eq": "2"}})
    log(f"S2 {len(s2)} dates · Landsat {len(ls)} dates")

    connection = connect()
    store.init(connection)
    for feature in fields:
        store.put_field(connection, feature)
    connection.commit()
    fids = [str(f["properties"]["id"]) for f in fields]

    hist = rotation(fields, connection)
    soils = soil(fields, connection, log)
    forced = forcing(fields, years, connection, log)
    log(f"rotation {hist} · soil {soils} · forcing {forced} days")

    params = {lid: f"expression={urllib.parse.quote(expr, safe='')}"
              "&asset_as_band=true" for lid, expr in EXPR.items()}
    done = 0
    for layer, source, found in (("ndvi", "sentinel-2-l2a", s2),
                                 ("ndmi", "sentinel-2-l2a", s2),
                                 ("lst", "landsat-c2-l2", ls)):
        for date, (item, cloud) in found.items():
            sid = store.put_scene(connection, item, source, date, cloud)
            if store.attempted(connection, sid, layer) is not None:
                continue
            cell = {}
            if layer == "lst":
                run = lambda f: _stat(source, sid, "assets=lwir11",
                                      "lwir11_b1", f)
                for feature in fields:
                    s = run(feature)
                    if s:
                        s = {k: KELVIN(v) if k in ("mean", "p10", "p90") else v
                             for k, v in s.items()}
                        cell[str(feature["properties"]["id"])] = s
                means = [v["mean"] for v in cell.values()]
                if cell and (max(means) < 5 or min(means) < -20):
                    log(f"lst {date} dropped {[round(m, 1) for m in means]}")
                    cell = {}
            else:
                run = lambda f: _stat(source, sid, params[layer], "b1", f)
                for feature in fields:
                    s = run(feature)
                    if s:
                        cell[str(feature["properties"]["id"])] = s
            for fid, stat in cell.items():
                store.put_observation(connection, fid, sid, layer, stat)
            store.put_coverage(connection, sid, layer, len(cell))
            connection.commit()
            if cell:
                done += 1
                log(f"{layer} {date} {len(cell)} parcels")

    if tiles:
        for layer, source, found, zmax in (
                ("ndvi", "sentinel-2-l2a", s2, 13),
                ("ndmi", "sentinel-2-l2a", s2, 13),
                ("rgb", "sentinel-2-l2a", s2, 13),
                ("lst", "landsat-c2-l2", ls, 12)):
            for date, (item, cloud) in found.items():
                baked = Path(tiles) / layer / date
                count = sum(len(f) for _, _, f in os.walk(baked)) \
                    if baked.is_dir() else 0
                grid = sum(1 for _ in _grid(bbox, 8, zmax))
                if count < grid // 2:
                    sid = store.put_scene(connection, item, source, date, cloud)
                    bake(layer, date, sid, source, Path(tiles), bbox, zmax, log)
    connection.close()
    return {"s2": len(s2), "landsat": len(ls), "ingested": done}
