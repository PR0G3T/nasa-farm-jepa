"""Cloud-Optimized GeoTIFF granules — windowed rasterio reads over fsspec.

Products like HLS *_VI and ECOSTRESS L*T* ship one GeoTIFF per variable
inside a single granule: h5netcdf cannot open them, and each granule
carries ~10 bands though only one or two are wanted. `prep` opens the
granule's file handles lazily (earthaccess handles cost nothing until
first read), picks files by their `.{BAND}.tif` suffix, reads only the
local window in the native CRS, and projects the pixel grid to WGS84 —
the same (values, glat, glon, conf) contract `support.prep` produces,
so one open feeds every field's footprint-weighted extraction.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator

import numpy as np

from nasa_farm.acquire.earthdata import Granule, granule_day


def band_of(name: str) -> str:
    """Per-file variable suffix: `X.v2.0.NDVI.tif` → `NDVI` (HLS dot
    naming), `ECO_..._LST.tif` → `LST` (ECOSTRESS underscore naming)."""
    stem = name.rsplit("/", 1)[-1]
    if stem.count(".") >= 2:
        return stem.rsplit(".", 2)[-2].upper()
    return stem.rsplit(".", 1)[0].rsplit("_", 1)[-1].upper()


@contextlib.contextmanager
def bands(granule: Granule) -> Iterator[dict[str, object]]:
    """granule → {BAND: open fsspec handle}; every handle closed on exit."""
    import earthaccess

    earthaccess.login(strategy="environment")
    handles = earthaccess.open([granule.raw])
    try:
        yield {band_of(getattr(h, "path", getattr(h, "full_name", ""))): h for h in handles}
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                h.close()


def pick(mapping: dict[str, object], variables: tuple[str, ...]) -> object | None:
    """First candidate band name present — products rename bands between
    versions, so specs declare tuples resolved in order."""
    upper = {k.upper(): v for k, v in mapping.items()}
    return next((upper[v.upper()] for v in variables if v.upper() in upper), None)


def window(handle: object, bbox: tuple[float, float, float, float]) -> tuple | None:
    """bbox (w, s, e, n) WGS84 → windowed (values, glat, glon) float64 —
    the read stays in the raster's native CRS and only the pixel grid is
    reprojected, so remote cost is the local window, never the tile."""
    import rasterio
    from pyproj import Transformer

    west, south, east, north = bbox
    source = (
        handle
        if isinstance(handle, rasterio.io.DatasetReader)
        else rasterio.open(handle)
    )
    with source as src:
        to_crs = Transformer.from_crs(4326, src.crs, always_xy=True)
        xs, ys = zip(
            *(
                to_crs.transform(lo, la)
                for lo, la in ((west, south), (east, south), (east, north), (west, north))
            ),
            strict=True,
        )
        if not all(np.isfinite(v) for v in (*xs, *ys)):
            return None  # bbox outside the raster's CRS validity (e.g. UTM zone)
        try:
            win = (
                rasterio.windows.from_bounds(
                    min(xs), min(ys), max(xs), max(ys), src.transform
                )
                .round_offsets()
                .round_lengths()
                .intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            )
        except rasterio.errors.WindowError:
            return None
        if win.width <= 0 or win.height <= 0:
            return None
        values = src.read(1, window=win).astype(np.float64)
        rows, cols = np.indices((int(win.height), int(win.width)))
        x, y = src.transform * (
            cols + win.col_off + 0.5,
            rows + win.row_off + 0.5,
        )
        glon, glat = Transformer.from_crs(src.crs, 4326, always_xy=True).transform(x, y)
        nodata = src.nodata
        scale = src.scales[0] or 1.0
        offset = src.offsets[0] or 0.0
    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)
    return values * scale + offset, np.asarray(glat), np.asarray(glon)


def prep(
    specification,
    granule: Granule,
    bbox: tuple[float, float, float, float],
) -> tuple | None:
    """Granule → (values, glat, glon, conf) grid for `specification.at`.

    The data band and its QA band are read over the same bbox window; a
    missing data band (renamed product) yields None so the granule is
    skipped, a missing QA band yields `quality=None` semantics upstream
    (finite-data confidence). `specification.scale`/`offset` apply after
    the file's own scale — e.g. mm/hr → mm/day for IMERG.
    """
    with bands(granule) as mapping:
        data = pick(mapping, specification.variables)
        if data is None:
            return None
        if (grid := window(data, bbox)) is None:
            return None
        values, glat, glon = grid
        values = values * specification.scale + specification.offset
        confidence = None
        if specification.quality is not None and (
            qc := pick(mapping, specification.quality.variables)
        ) is not None:
            if (qgrid := window(qc, bbox)) is not None:
                confidence = np.asarray(specification.quality.to_conf(qgrid[0]), np.float64)
        return values, glat, glon, confidence


def day_of(granule: Granule) -> int:
    """yyyymmdd of a COG granule — same id-parse as HDF products."""
    return granule_day(granule.id)


_MONTH = re.compile(r"(?<!\d)(?:19|20)\d{2}(\d{2})\d{2}(?!\d)")


def month_of(granule: Granule) -> int:
    """Calendar month 1-12 parsed from the granule id — cheap filter that
    keeps crawl spans honest without opening files."""
    if m := _MONTH.search(granule.id):
        return int(m[1])
    return day_of(granule) // 100 % 100
