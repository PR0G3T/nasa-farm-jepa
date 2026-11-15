"""USDA Crop Sequence Boundaries — field vectors over HTTP ranges.

The national CSB vector (16.2 M fields) is mirrored as a single 4.3 GB
GeoParquet on Source Cooperative. DuckDB httpfs+spatial range-reads it
and pushes `bbox` struct predicates down to row-group statistics, so a
county extract lands in seconds — the full download never happens.
Geometry is EPSG:5070 (NAD83 CONUS Albers): `ST_Transform` reprojects in
SQL and the Albers centroid doubles as the point where `sequence` samples
the yearly CDL rasters for the real rotation history.

Fields are the cultivated CSB classes (CDL 1–61): grass, forest,
developed and water polygons are dropped — the model cannot rotate them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nasa_farm.acquire.base import Bounds

if TYPE_CHECKING:
    from collections.abc import Iterable

PARQUET = ("https://data.source.coop/fiboa/us-usda-cropland/"
           "us_usda_cropland.parquet")

CDL_STORE = {"bucket": "chill",
             "prefix": "usda-cropland-data-layer/v0.1.0.icechunk",
             "endpoint_url": "https://data.source.coop",
             "region": "us-east-1"}
"""CDL 2008–2025 as one Icechunk Zarr on Source Cooperative — sharded
(1, 512, 512), so a point time series costs one range request per year."""

CULTIVATED = 61
"""CDL classes ≤ 61 are cropland; grass (176) and everything above is not."""

ALBERS = 5070
"""Native CRS of both the CSB parquet and the CDL rasters."""

CDL: dict[int, tuple[str, str | None]] = {
    1: ("Corn", "maize"), 2: ("Cotton", None), 3: ("Rice", None),
    4: ("Sorghum", "sorghum"), 5: ("Soybeans", "soybean"),
    6: ("Sunflower", None), 10: ("Peanuts", None), 11: ("Tobacco", None),
    12: ("Sweet corn", "maize"), 13: ("Pop / ornamental corn", "maize"),
    14: ("Mint", None), 21: ("Barley", "wheat"), 22: ("Durum wheat", "wheat"),
    23: ("Spring wheat", "wheat"), 24: ("Winter wheat", "wheat"),
    25: ("Other small grains", "wheat"), 26: ("Winter wheat / soybeans", "soybean"),
    27: ("Rye", "wheat"), 28: ("Oats", "wheat"), 29: ("Millet", None),
    30: ("Speltz", "wheat"), 31: ("Canola", None), 32: ("Flaxseed", None),
    33: ("Safflower", None), 34: ("Rape seed", None), 35: ("Mustard", None),
    36: ("Alfalfa", "alfalfa"), 37: ("Other hay", None), 38: ("Camelina", None),
    39: ("Buckwheat", None), 41: ("Sugarbeets", None), 42: ("Dry beans", None),
    43: ("Potatoes", None), 44: ("Other crops", None), 45: ("Sugarcane", None),
    46: ("Sweet potatoes", None), 47: ("Misc vegetables & fruits", None),
    48: ("Watermelons", None), 49: ("Onions", None), 50: ("Cucumbers", None),
    51: ("Chick peas", None), 52: ("Lentils", None), 53: ("Peas", None),
    54: ("Tomatoes", None), 55: ("Caneberries", None), 56: ("Hops", None),
    57: ("Herbs", None), 58: ("Clover / wildflowers", "clover"),
    59: ("Sod / grass seed", None), 60: ("Switchgrass", None),
    61: ("Fallow / idle cropland", "fallow"),
}
"""CDL cropland class → (label, model crop); unmapped crops keep `kind=None`."""


@dataclass(frozen=True, slots=True)
class Boundary:
    """One CSB field polygon.

    `ring`/`center` are WGS84 for display and planning; `x`,`y` are the
    EPSG:5070 centroid — the sample point `sequence` needs for the CDL
    rasters, which share that CRS."""

    id: str
    name: str
    crop: str
    kind: str | None
    ha: float
    center: tuple[float, float]
    ring: list[tuple[float, float]]
    x: float
    y: float


def _centroid(ring: list[tuple[float, float]]) -> tuple[float, float]:
    """Polygon centroid of a closed ring; first vertex on degenerate input."""
    area = x = y = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        cross = x0 * y1 - x1 * y0
        area += cross
        x += (x0 + x1) * cross
        y += (y0 + y1) * cross
    return (x / (3 * area), y / (3 * area)) if area else ring[0]


def _ring(geometry: dict) -> list[tuple[float, float]]:
    """Largest outer ring of a (Multi)Polygon GeoJSON geometry, closed."""
    match geometry["type"]:
        case "Polygon":
            rings = [geometry["coordinates"][0]]
        case "MultiPolygon":
            rings = [part[0] for part in geometry["coordinates"]]
        case other:
            raise ValueError(f"unsupported geometry {other!r}")
    def area(ring: list) -> float:
        return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:]))
    outer = [tuple(point) for point in max(rings, key=lambda r: abs(area(r)))]
    return outer if outer[0] == outer[-1] else [*outer, outer[0]]


def _albers(bounds: Bounds) -> Bounds:
    """WGS84 search window → EPSG:5070 with margin — 4-corner transform is
    exact for the small windows fields are queried on."""
    from pyproj import Transformer
    points = Transformer.from_crs(4326, ALBERS, always_xy=True).itransform(
        [(bounds.west, bounds.south), (bounds.east, bounds.south),
         (bounds.east, bounds.north), (bounds.west, bounds.north)])
    xs, ys = zip(*points)
    return Bounds(min(xs), min(ys), max(xs), max(ys))


def _sql(bounds: Bounds | None, county: str | None, limit: int) -> str:
    clauses = [f'TRY_CAST("crop:code" AS INTEGER) BETWEEN 1 AND {CULTIVATED}']
    if county:
        clauses.append(f"administrative_area_level_2 = '{county}'")
    if bounds:
        b = _albers(bounds)
        clauses.append(f"bbox.xmax > {b.west} AND bbox.xmin < {b.east} "
                       f"AND bbox.ymax > {b.south} AND bbox.ymin < {b.north}")
    return f"""
SELECT id, TRY_CAST("crop:code" AS INTEGER), ST_Area(geometry) / 1e4,
       ST_X(ST_Centroid(geometry)), ST_Y(ST_Centroid(geometry)),
       ST_AsGeoJSON(ST_Transform(geometry, 'EPSG:5070', 'EPSG:4326'))
FROM read_parquet('{PARQUET}')
WHERE {' AND '.join(clauses)}
ORDER BY 3 DESC LIMIT {int(limit)}"""


def _stratify(rows: list[tuple], limit: int, per_kind: int) -> list[tuple]:
    """Area-ranked rows → `limit` with at most `per_kind` of each model
    crop; leftover slots backfill by area so the total holds."""
    kinds: dict[str, list[tuple]] = {}
    for row in rows:
        if (kind := CDL.get(row[1], (None, None))[1]) is not None:
            kinds.setdefault(kind, []).append(row)
    quota = [row for group in kinds.values() for row in group[:per_kind]]
    chosen = sorted(quota, key=lambda r: r[2], reverse=True)[:limit]
    if len(chosen) < limit:
        seen = {r[0] for r in chosen}
        chosen += [r for r in rows if r[0] not in seen][:limit - len(chosen)]
    return sorted(chosen, key=lambda r: r[2], reverse=True)


def fields(bounds: Bounds | None = None, *, county: str | None = None,
           limit: int = 40, per_kind: int | None = None) -> list[Boundary]:
    """CSB fields intersecting `bounds` (WGS84) / in `county`,
    largest first. `per_kind` caps each model crop for crop-diverse
    demos. Needs the `csb` extra (duckdb + spatial, pyproj)."""
    import duckdb

    connection = duckdb.connect()
    connection.execute(
        "INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; "
        "SET geometry_always_xy = true")
    fetch = limit if per_kind is None else 10_000
    rows = connection.execute(_sql(bounds, county, fetch)).fetchall()
    if per_kind is not None:
        rows = _stratify(rows, limit, per_kind)
    stem = county or "Field"
    output = []
    for rank, (fid, code, ha, x, y, geo) in enumerate(rows):
        ring = _ring(json.loads(geo))
        crop, kind = CDL.get(code, ("Other", None))
        output.append(Boundary(
            id=str(fid), name=f"{stem} {rank + 1}", crop=crop, kind=kind,
            ha=round(float(ha), 2), center=_centroid(ring), ring=ring,
            x=float(x), y=float(y)))
    return output


def sequence(boundaries: Iterable[Boundary], years: range
             ) -> dict[str, list[int]]:
    """CDL class at each field centroid per year — the real rotation the
    field ran, e.g. corn→soy→corn. Reads only the touched Zarr shards."""
    import icechunk
    import xarray as xarray

    storage = icechunk.s3_storage(**CDL_STORE, anonymous=True,
                                  force_path_style=True)
    session = icechunk.Repository.open(storage).readonly_session("main")
    dataset = xarray.open_zarr(session.store, group="30m", chunks=None)
    have = [y for y in years if y in set(dataset.year.values.tolist())]
    items = list(boundaries)
    crop = dataset["crop_type"].sel(
        year=have,
        x=xarray.DataArray([b.x for b in items], dims="point"),
        y=xarray.DataArray([b.y for b in items], dims="point"),
        method="nearest")
    values = crop.transpose("point", "year").values
    return {b.id: [int(v) for v in row] for b, row in zip(items, values)}


def label(code: int) -> str:
    return CDL.get(code, ("Other",))[0]


def feature(boundary: Boundary, rotation: list[int] | None = None) -> dict:
    """Boundary → GeoJSON Feature the app consumes verbatim."""
    properties: dict = {
        "id": boundary.id, "name": boundary.name, "crop": boundary.crop,
        "kind": boundary.kind, "ha": boundary.ha}
    if rotation is not None:
        properties["sequence"] = [label(code) for code in rotation]
    return {"type": "Feature", "properties": properties,
            "geometry": {"type": "Polygon", "coordinates": [boundary.ring]}}
