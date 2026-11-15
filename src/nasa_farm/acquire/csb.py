"""USDA Crop Sequence Boundaries — field vectors over HTTP ranges.

The national CSB vector (16.2 M fields) is mirrored as one 4.3 GB
GeoParquet on Source Cooperative. An anonymous S3-range pyarrow scan
pushes `bbox` struct predicates down to row-group statistics, so a
county extract lands in well under a minute while streaming — the full
download never happens. Geometry is EPSG:5070 (NAD83 CONUS Albers): the
exterior ring is decoded straight out of WKB (no GIS dependency),
reprojected to WGS84 with pyproj, and the Albers centroid doubles as
the point where `sequence` samples the yearly CDL rasters for the real
rotation history.

Fields are the cultivated CSB classes (CDL 1–61): grass, forest,
developed and water polygons are dropped — the model cannot rotate them.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nasa_farm.acquire.base import Bounds

if TYPE_CHECKING:
    from collections.abc import Iterable

BUCKET = "fiboa/us-usda-cropland/us_usda_cropland.parquet"

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

COLUMNS = ["id", "crop:code", "geometry"]

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


def _wkb(blob: bytes) -> list[list[tuple[float, float]]]:
    """Exterior rings of a Polygon/MultiPolygon WKB — one ring per part.

    Reads only what `fields` needs (no holes, no Z/M): byte order flag,
    type tag, then ring headers of uint32 count + 16 B coordinate pairs.
    """
    def polygon(offset: int, endian: str) -> tuple[list, int]:
        count, = struct.unpack_from(endian + "I", blob, offset)
        offset += 4
        rings = []
        for _ in range(count):
            n, = struct.unpack_from(endian + "I", blob, offset)
            offset += 4
            xy = struct.unpack_from(endian + f"{2 * n}d", blob, offset)
            rings.append(list(zip(xy[::2], xy[1::2])))
            offset += 16 * n
        return rings, offset

    endian = "<" if blob[0] else ">"
    kind, = struct.unpack_from(endian + "I", blob, 1)
    if kind == 3:  # Polygon
        rings, _ = polygon(5, endian)
        return rings[:1]
    if kind == 6:  # MultiPolygon
        count, = struct.unpack_from(endian + "I", blob, 5)
        offset, outer = 9, []
        for _ in range(count):
            e = "<" if blob[offset] else ">"
            tag, = struct.unpack_from(e + "I", blob, offset + 1)
            if tag != 3:
                raise ValueError(f"unsupported WKB member {tag}")
            rings, offset = polygon(offset + 5, e)
            outer.append(rings[0])
        return outer
    raise ValueError(f"unsupported WKB {kind}")


def _shoelace(ring: list[tuple[float, float]]) -> float:
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:]))


def _centroid(ring: list[tuple[float, float]]) -> tuple[float, float]:
    """Polygon centroid of a closed ring; first vertex on degenerate input."""
    area = x = y = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        cross = x0 * y1 - x1 * y0
        area += cross
        x += (x0 + x1) * cross
        y += (y0 + y1) * cross
    return (x / (3 * area), y / (3 * area)) if area else ring[0]


def _albers(bounds: Bounds) -> Bounds:
    """WGS84 search window → EPSG:5070 — 4-corner transform is exact for
    the small windows fields are queried on."""
    from pyproj import Transformer
    points = Transformer.from_crs(4326, ALBERS, always_xy=True).itransform(
        [(bounds.west, bounds.south), (bounds.east, bounds.south),
         (bounds.east, bounds.north), (bounds.west, bounds.north)])
    xs, ys = zip(*points)
    return Bounds(min(xs), min(ys), max(xs), max(ys))


def _scan(bounds: Bounds | None, county: str | None) -> list[tuple]:
    """(id, code, county, wkb) tuples — bbox predicates prune row groups
    server-side via footer statistics; only matching chunks are fetched."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.fs as fs
    import pyarrow.types as types

    store = fs.S3FileSystem(anonymous=True, region="us-east-1",
                            endpoint_override="data.source.coop")
    dataset = ds.dataset(BUCKET, filesystem=store, format="parquet")
    code = pc.field("crop:code")
    if not types.is_integer(dataset.schema.field("crop:code").type):
        code = code.cast("int32")
    clauses = (code >= 1) & (code <= CULTIVATED)
    if county:
        clauses &= pc.field("administrative_area_level_2") == county
    if bounds:
        b = _albers(bounds)
        clauses &= ((pc.field("bbox", "xmax") > b.west)
                    & (pc.field("bbox", "xmin") < b.east)
                    & (pc.field("bbox", "ymax") > b.south)
                    & (pc.field("bbox", "ymin") < b.north))
    table = dataset.to_table(columns=COLUMNS, filter=clauses)
    return list(zip(table["id"].to_pylist(),
                    [int(v) for v in table["crop:code"].to_pylist()],
                    table["geometry"].to_pylist()))


def _ring(blob: bytes) -> list[tuple[float, float]]:
    """Largest exterior ring of a WKB blob, closed (Albers metres)."""
    ring = max(_wkb(blob), key=lambda r: abs(_shoelace(r)))
    return ring if ring[0] == ring[-1] else [*ring, ring[0]]


def _boundary(fid: str, code: int, name: str,
              ring: list[tuple[float, float]], transform) -> Boundary:
    """One Albers ring → reprojected WGS84 boundary."""
    x, y = _centroid(ring)
    crop, kind = CDL.get(code, ("Other", None))
    wgs = [tuple(p) for p in transform.itransform(ring)]
    return Boundary(id=fid, name=name, crop=crop, kind=kind,
                    ha=round(abs(_shoelace(ring)) / 2e4, 2),
                    center=_centroid(wgs), ring=wgs, x=x, y=y)


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
    demos. Needs the `csb` extra (pyarrow, icechunk, pyproj)."""
    from pyproj import Transformer

    rows = _scan(bounds, county)
    candidates = []
    for fid, code, blob in rows:
        ring = _ring(blob)
        candidates.append((fid, code, abs(_shoelace(ring)) / 2e4, ring))
    candidates.sort(key=lambda r: r[2], reverse=True)
    if per_kind is not None:
        candidates = _stratify(candidates, limit, per_kind)
    else:
        candidates = candidates[:limit]
    stem = county or "Field"
    transform = Transformer.from_crs(ALBERS, 4326, always_xy=True)
    return [_boundary(str(fid), code, f"{stem} {rank + 1}", ring, transform)
            for rank, (fid, code, _ha, ring) in enumerate(candidates)]


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
