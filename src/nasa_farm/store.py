"""Postgres/PostGIS store — source of truth for scenes and observations.

`NASA_FARM_DSN` points at the database
(default `postgresql://postgres:postgres@localhost:5432/nasafarm`).

Schema — star layout:

- `fields`   parcels, with PostGIS `geography` for neighbor queries
- `sources`  satellite collections (sentinel-2-l2a, landsat-c2-l2, …)
- `scenes`   acquired items, one per source per day (lowest cloud wins)
- `layers`   products derived from sources (ndvi, ndmi, lst, rgb)
- `observations` fact table: parcel × scene × layer → mean/p10/p90/n
"""

from __future__ import annotations

import json
import os

import pg8000.dbapi

DDL = """
CREATE TABLE IF NOT EXISTS sources(
    code TEXT PRIMARY KEY,
    label TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS fields(
    fid TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    crop TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT '',
    kinds JSONB,
    sequence JSONB,
    ha REAL,
    geom geography(Polygon, 4326));
CREATE INDEX IF NOT EXISTS fields_geom ON fields USING gist(geom);
CREATE TABLE IF NOT EXISTS scenes(
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL REFERENCES sources(code),
    acquired DATE NOT NULL,
    cloud REAL,
    UNIQUE(source, acquired));
CREATE TABLE IF NOT EXISTS layers(
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL DEFAULT '',
    unit TEXT NOT NULL DEFAULT '',
    lo REAL, hi REAL,
    source TEXT NOT NULL REFERENCES sources(code));
CREATE TABLE IF NOT EXISTS observations(
    field TEXT NOT NULL REFERENCES fields(fid),
    scene TEXT NOT NULL REFERENCES scenes(id),
    layer TEXT NOT NULL REFERENCES layers(id),
    mean REAL, p10 REAL, p90 REAL, n INTEGER,
    PRIMARY KEY(field, scene, layer));
CREATE TABLE IF NOT EXISTS coverage(
    scene TEXT NOT NULL REFERENCES scenes(id),
    layer TEXT NOT NULL REFERENCES layers(id),
    parcels INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(scene, layer));
CREATE TABLE IF NOT EXISTS forcing_daily(
    field TEXT NOT NULL REFERENCES fields(fid),
    acquired DATE NOT NULL,
    source TEXT NOT NULL REFERENCES sources(code),
    tmin REAL, tmax REAL, srad REAL, precip REAL, rh REAL, u2 REAL,
    gwetroot REAL,
    PRIMARY KEY(field, acquired, source));
CREATE INDEX IF NOT EXISTS forcing_daily_date ON forcing_daily(acquired);
CREATE TABLE IF NOT EXISTS soil_profile(
    field TEXT PRIMARY KEY REFERENCES fields(fid),
    depth REAL, clay REAL, sand REAL, bdod REAL, soc REAL, phh2o REAL,
    lo JSONB, hi JSONB,
    source TEXT NOT NULL REFERENCES sources(code));
CREATE TABLE IF NOT EXISTS rotation(
    field TEXT NOT NULL REFERENCES fields(fid),
    year INTEGER NOT NULL, kind TEXT, label TEXT,
    PRIMARY KEY(field, year));
"""

LAYERS = {
    "ndvi": ("NDVI", "", 0.0, 0.85, "sentinel-2-l2a"),
    "ndmi": ("Water", "", -0.35, 0.55, "sentinel-2-l2a"),
    "lst": ("Heat", "°C", 15.0, 45.0, "landsat-c2-l2"),
    "rgb": ("True colour", "", 0.0, 1.0, "sentinel-2-l2a"),
}
SOURCES = {
    "sentinel-2-l2a": "Sentinel-2 L2A, ESA via Planetary Computer",
    "landsat-c2-l2": "Landsat 8/9 C2 L2, USGS via Planetary Computer",
    "nasa-power": "NASA POWER Agroclimatology daily point",
    "soilgrids": "ISRIC SoilGrids v2.0",
    "cdl": "USDA NASS Cropland Data Layer via CSB",
}


def dsn() -> str | None:
    return os.environ.get("NASA_FARM_DSN")


def connect(url: str | None = None):
    """pg8000 handle from a `postgresql://user:pass@host:port/db` DSN."""
    from urllib.parse import urlparse

    parts = urlparse(url or dsn() or
                     "postgresql://postgres:postgres@localhost:5432/nasafarm")
    connection = pg8000.dbapi.connect(
        user=parts.username or "postgres",
        password=parts.password or "",
        host=parts.hostname or "localhost",
        port=parts.port or 5432,
        database=parts.path.lstrip("/") or "postgres",
        timeout=3,
    )
    connection.autocommit = False
    return connection


def init(connection) -> None:
    """Idempotent schema + vocab."""
    cursor = connection.cursor()
    try:
        for statement in DDL.split(";"):
            if statement.strip():
                cursor.execute(statement)
        for code, label in SOURCES.items():
            cursor.execute(
                "INSERT INTO sources(code, label) VALUES (%s, %s)"
                " ON CONFLICT(code) DO NOTHING",
                (code, label))
        for lid, (label, unit, lo, hi, source) in LAYERS.items():
            cursor.execute(
                "INSERT INTO layers(id, label, unit, lo, hi, source)"
                " VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(id) DO NOTHING",
                (lid, label, unit, lo, hi, source))
        cursor.execute(
            """INSERT INTO coverage(scene, layer, parcels)
            SELECT scene, layer, count(DISTINCT field) FROM observations
            GROUP BY scene, layer ON CONFLICT DO NOTHING""")
    finally:
        cursor.close()
    connection.commit()


def put_field(connection, feature: dict) -> None:
    props = feature.get("properties", {})
    cursor = connection.cursor()
    try:
        cursor.execute(
            """INSERT INTO fields(fid, name, crop, kind, kinds, sequence, ha, geom)
            VALUES (%s,%s,%s,%s,%s,%s,%s, ST_GeomFromGeoJSON(%s)::geography)
            ON CONFLICT(fid) DO UPDATE SET
                name = EXCLUDED.name, crop = EXCLUDED.crop,
                kind = EXCLUDED.kind, kinds = EXCLUDED.kinds,
                sequence = EXCLUDED.sequence, ha = EXCLUDED.ha,
                geom = EXCLUDED.geom""",
            (str(props.get("id")), props.get("name", ""),
             props.get("crop", ""), props.get("kind", ""),
             json.dumps(props.get("kinds")) if props.get("kinds") else None,
             json.dumps(props.get("sequence")) if props.get("sequence") else None,
             props.get("ha"), json.dumps(feature["geometry"])))
    finally:
        cursor.close()


def put_scene(connection, item: str, source: str, acquired: str,
              cloud: float | None) -> str:
    """Insert-if-missing; returns the canonical item id stored for that day."""
    cursor = connection.cursor()
    try:
        cursor.execute(
            """INSERT INTO scenes(id, source, acquired, cloud)
            VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
            (item, source, acquired, cloud))
        cursor.execute(
            "SELECT id FROM scenes WHERE source = %s AND acquired = %s",
            (source, acquired))
        return cursor.fetchone()[0]
    finally:
        cursor.close()


def put_observation(connection, field: str, scene: str, layer: str,
                    stat: dict) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute(
            """INSERT INTO observations(field, scene, layer, mean, p10, p90, n)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(field, scene, layer) DO UPDATE SET
                mean = EXCLUDED.mean, p10 = EXCLUDED.p10,
                p90 = EXCLUDED.p90, n = EXCLUDED.n""",
            (field, scene, layer, stat.get("mean"), stat.get("p10"),
             stat.get("p90"), stat.get("n")))
    finally:
        cursor.close()


def attempted(connection, scene: str, layer: str) -> int | None:
    """Parcels collected for (scene, layer); None = never attempted."""
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT parcels FROM coverage WHERE scene = %s AND layer = %s",
            (scene, layer))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        cursor.close()


def put_coverage(connection, scene: str, layer: str, parcels: int) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute(
            """INSERT INTO coverage(scene, layer, parcels) VALUES (%s,%s,%s)
            ON CONFLICT(scene, layer) DO UPDATE SET
                parcels = EXCLUDED.parcels""",
            (scene, layer, parcels))
    finally:
        cursor.close()


def forcing_max(connection, field: str, source: str) -> str | None:
    """Latest forced day; None = never fetched."""
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT max(acquired) FROM forcing_daily"
            " WHERE field = %s AND source = %s", (field, source))
        return cursor.fetchone()[0]
    finally:
        cursor.close()


def forcing_years(connection, field: str, source: str) -> set[int]:
    """Calendar years having at least one forced day."""
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT DISTINCT EXTRACT(year FROM acquired)::int"
            " FROM forcing_daily WHERE field = %s AND source = %s",
            (field, source))
        return {row[0] for row in cursor}
    finally:
        cursor.close()


def put_forcing(connection, field: str, source: str, rows) -> None:
    """rows: (date, tmin, tmax, srad, precip, rh, u2, gwetroot)."""
    cursor = connection.cursor()
    try:
        for row in rows:
            cursor.execute(
                """INSERT INTO forcing_daily
                (field, acquired, source, tmin, tmax, srad, precip, rh, u2,
                 gwetroot)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(field, acquired, source) DO UPDATE SET
                    tmin = EXCLUDED.tmin, tmax = EXCLUDED.tmax,
                    srad = EXCLUDED.srad, precip = EXCLUDED.precip,
                    rh = EXCLUDED.rh, u2 = EXCLUDED.u2,
                    gwetroot = EXCLUDED.gwetroot""",
                (field, str(row[0]), source, *row[1:]))
    finally:
        cursor.close()


def has_soil(connection, field: str) -> bool:
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT 1 FROM soil_profile WHERE field = %s", (field,))
        return cursor.fetchone() is not None
    finally:
        cursor.close()


def put_soil(connection, field: str, sample: dict) -> None:
    """sample: SoilGrids().sample() output (median + lo/hi bounds)."""
    cursor = connection.cursor()
    try:
        cursor.execute(
            """INSERT INTO soil_profile
            (field, depth, clay, sand, bdod, soc, phh2o, lo, hi, source)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'soilgrids')
            ON CONFLICT(field) DO UPDATE SET
                depth = EXCLUDED.depth, clay = EXCLUDED.clay,
                sand = EXCLUDED.sand, bdod = EXCLUDED.bdod,
                soc = EXCLUDED.soc, phh2o = EXCLUDED.phh2o,
                lo = EXCLUDED.lo, hi = EXCLUDED.hi""",
            (field, sample.get("depth"), sample.get("clay"),
             sample.get("sand"), sample.get("bd"), sample.get("soc"),
             sample.get("ph"),
             json.dumps({"clay": sample.get("clay_lo"),
                        "sand": sample.get("sand_lo"),
                        "bdod": sample.get("bd_lo")}),
             json.dumps({"clay": sample.get("clay_hi"),
                        "sand": sample.get("sand_hi"),
                        "bdod": sample.get("bd_hi")})))
    finally:
        cursor.close()


def put_rotation(connection, field: str, rows) -> None:
    """rows: (year, kind, label)."""
    cursor = connection.cursor()
    try:
        for year, kind, label in rows:
            cursor.execute(
                """INSERT INTO rotation(field, year, kind, label)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT(field, year) DO UPDATE SET
                    kind = EXCLUDED.kind, label = EXCLUDED.label""",
                (field, year, kind, label))
    finally:
        cursor.close()


FORCING_PARAMS = ("T2M_MIN", "T2M_MAX", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR",
                  "RH2M", "WS2M", "GWETROOT")


def forcing_data(connection, field: str, source: str,
                 year: int) -> dict[str, dict[str, float]]:
    """Stored days → POWER-shaped {param: {yyyymmdd: v}} for series_to_*."""
    cursor = connection.cursor()
    try:
        cursor.execute(
            """SELECT acquired, tmin, tmax, srad, precip, rh, u2, gwetroot
            FROM forcing_daily
            WHERE field = %s AND source = %s
                AND acquired BETWEEN %s AND %s""",
            (field, source, f"{year}-01-01", f"{year}-12-31"))
        out = {p: {} for p in FORCING_PARAMS}
        for row in cursor:
            key = row[0].strftime("%Y%m%d")
            for param, value in zip(FORCING_PARAMS, row[1:]):
                if value is not None:
                    out[param][key] = value
        return out
    finally:
        cursor.close()


def soil_sample(connection, field: str) -> dict | None:
    """soil_profile row → SoilGrids.sample() dict shape."""
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT depth, clay, sand, bdod, soc, phh2o, lo, hi"
            " FROM soil_profile WHERE field = %s", (field,))
        row = cursor.fetchone()
        if not row:
            return None
        lo, hi = row[6] or {}, row[7] or {}
        return {
            "clay": row[1], "sand": row[2], "bd": row[3],
            "soc": row[4], "ph": row[5], "depth": row[0],
            "clay_lo": lo.get("clay"), "sand_lo": lo.get("sand"),
            "bd_lo": lo.get("bdod"), "clay_hi": hi.get("clay"),
            "sand_hi": hi.get("sand"), "bd_hi": hi.get("bdod"),
        }
    finally:
        cursor.close()


def field_detail(connection, fid: str) -> dict | None:
    """Agronomist view: parcel props + soil + rotation + recent weather."""
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT fid, name, crop, kind, ha FROM fields WHERE fid = %s",
            (fid,))
        row = cursor.fetchone()
        if not row:
            return None
        out = {"field": {"fid": row[0], "name": row[1], "crop": row[2],
                         "kind": row[3], "ha": row[4]}}
        out["soil"] = soil_sample(connection, fid)
        cursor.execute(
            "SELECT year, kind, label FROM rotation WHERE field = %s"
            " ORDER BY year", (fid,))
        out["rotation"] = [{"year": y, "kind": k, "label": l}
                           for y, k, l in cursor]
        cursor.execute(
            """SELECT acquired, tmin, tmax, precip, gwetroot FROM forcing_daily
            WHERE field = %s AND source = 'nasa-power' AND tmin IS NOT NULL
            ORDER BY acquired DESC LIMIT 30""", (fid,))
        days = cursor.fetchall()
        if days:
            latest = days[0]
            out["weather"] = {
                "acquired": latest[0].isoformat(),
                "tmin": latest[1], "tmax": latest[2],
                "precip": latest[3], "gwetroot": latest[4],
                "precip_7d": round(sum(d[3] or 0 for d in days[:7]), 1),
                "precip_30d": round(sum(d[3] or 0 for d in days), 1),
            }
        return out
    finally:
        cursor.close()


def catalog(connection) -> dict:
    """Everything the app needs: {stats, scenes} in the layers.json shape."""
    stats: dict = {}
    scenes: dict = {}
    cursor = connection.cursor()
    try:
        cursor.execute(
            """SELECT o.layer, s.acquired, s.id, o.field, o.mean, o.p10, o.p90, o.n
            FROM observations o JOIN scenes s ON s.id = o.scene
            ORDER BY s.acquired""")
        for layer, acquired, scene, field, mean, p10, p90, n in cursor:
            iso = acquired.isoformat()
            stats.setdefault(layer, {}).setdefault(iso, {})[field] = {
                "mean": mean, "p10": p10, "p90": p90, "n": n}
            scenes.setdefault(layer, {})[iso] = scene
    finally:
        cursor.close()
    return {"stats": stats, "scenes": scenes}
