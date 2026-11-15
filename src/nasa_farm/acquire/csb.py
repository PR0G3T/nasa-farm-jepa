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
CDL_URL = "/vsicurl/https://data.source.coop/tyler/usda-nass-cdl/{y}/{y}_30m_cdls.tif"

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


def fields(bounds: Bounds | None = None, *, county: str | None = None,
           limit: int = 40) -> list[Boundary]:
    """CSB fields intersecting `bounds` (WGS84) / in `county`,
    largest first. Needs the `csb` extra (duckdb + spatial, pyproj)."""
    import duckdb

    connection = duckdb.connect()
    connection.execute(
        "INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; "
        "SET geometry_always_xy = true")
    rows = connection.execute(_sql(bounds, county, limit)).fetchall()
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


def sequence(boundaries: Iterable[Boundary], years: range,
             url: str = CDL_URL) -> dict[str, list[int]]:
    """CDL class at each field centroid, one raster open per year — the
    real rotation the field ran, e.g. corn→soy→corn."""
    import rasterio

    items = list(boundaries)
    series: dict[str, list[int]] = {b.id: [] for b in items}
    for year in years:
        with rasterio.open(url.format(y=year)) as raster:
            for boundary, (value,) in zip(
                    items, raster.sample([(b.x, b.y) for b in items])):
                series[boundary.id].append(int(value))
    return series


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
