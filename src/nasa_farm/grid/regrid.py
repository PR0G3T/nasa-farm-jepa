"""Native grid → common 1 km grid, with the method chosen from the data.

Rules (`method_for`), in order:

| Variable kind | Source finer than ~1 km | Source coarser than 1 km |
|---|---|---|
| continuous | area-weighted **average** of valid pixels | **bilinear** |
| categorical | per-class **area fraction** + **mode** | **nearest** (fractions 0/1) |
| count-like (points, polygons) | binned / rasterized at 100 m, then summed or averaged | — |

Why: averaging is the only aggregation that keeps the cell mean equal to
the area mean of the source (GDAL ≥ 3.1 weights partial pixels by their
overlap); averaging a coarse source just copies it into blocks, so above
1 km we interpolate between cell centres instead. Bilinear is not
conservative — a 9 km SMAP cell's mean is not exactly preserved — but at
a 9–25× ratio the error is small next to the products' own uncertainty;
use a conservative remapper (xESMF) if exact budgets matter (precip
totals over a basin). Class codes must never be averaged, hence
fractions. Quality masks are applied *at native resolution before
aggregation* (masked pixels become NaN and simply do not contribute);
the share of valid area is returned as `valid_frac`, which the training
loss can use as an observation weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from nasa_farm.grid.target import TargetGrid

FINE_RATIO = 1.0
"""Source pixel ≤ FINE_RATIO × target res → aggregate (average/fractions);
above → interpolate (bilinear/nearest)."""


@dataclass(slots=True)
class Raster:
    """One source array on its native regular grid.

    `data` is (band, y, x); float bands use NaN as nodata (readers convert
    fill values and apply QA before building a Raster); categorical bands
    keep integer codes with `nodata`. `bands` names each band — the same
    name repeated with different `times` is a time axis (hourly ERA5-Land).
    """

    data: np.ndarray
    transform: object  # affine.Affine
    crs: object        # anything rasterio.crs.CRS accepts
    bands: list[str]
    kind: str = "continuous"   # "continuous" | "categorical"
    nodata: float | int | None = None
    times: list[str | None] | None = None
    units: dict[str, str] = field(default_factory=dict)
    classes: Sequence[int] | None = None       # categorical: fixed class list
    class_names: dict[int, str] | None = None

    def __post_init__(self) -> None:
        if self.data.ndim == 2:
            self.data = self.data[None]
        if len(self.bands) != self.data.shape[0]:
            raise ValueError(f"{len(self.bands)} names for "
                             f"{self.data.shape[0]} bands")
        if self.times is None:
            self.times = [None] * len(self.bands)


def native_res_m(r: Raster) -> float:
    """Geometric-mean pixel size in metres (degrees converted at the
    raster's centre latitude for geographic CRSs)."""
    from rasterio.crs import CRS
    crs = CRS.from_user_input(r.crs)
    dx, dy = abs(r.transform.a), abs(r.transform.e)
    if crs.is_geographic:
        ny = r.data.shape[-2]
        lat = r.transform.f + r.transform.e * ny / 2
        dx *= 111_320.0 * math.cos(math.radians(lat))
        dy *= 110_540.0
    else:
        dx *= crs.linear_units_factor[1] if crs.linear_units_factor else 1.0
        dy *= crs.linear_units_factor[1] if crs.linear_units_factor else 1.0
    return math.sqrt(dx * dy)


def method_for(kind: str, src_res_m: float, dst_res_m: float) -> str:
    fine = src_res_m <= FINE_RATIO * dst_res_m
    if kind == "continuous":
        return "average" if fine else "bilinear"
    if kind == "categorical":
        return "fractions" if fine else "nearest"
    raise ValueError(f"unknown kind {kind!r}")


def _reproject(arr: np.ndarray, r: Raster, grid: TargetGrid, how: str,
               src_nodata, dst_dtype=np.float32, dst_nodata=np.nan
               ) -> np.ndarray:
    from rasterio.warp import Resampling, reproject
    dst = np.full(grid.shape, dst_nodata, dtype=dst_dtype)
    reproject(source=arr, destination=dst,
              src_transform=r.transform, src_crs=r.crs,
              dst_transform=grid.transform, dst_crs=grid.crs,
              src_nodata=src_nodata, dst_nodata=dst_nodata,
              resampling=getattr(Resampling, how))
    return dst


def regrid_continuous(r: Raster, grid: TargetGrid, method: str | None = None
                      ) -> tuple[np.ndarray, np.ndarray | None, str]:
    """→ (values (band, ny, nx) float32, valid_frac (ny, nx) | None, method)."""
    how = method or method_for("continuous", native_res_m(r), grid.res)
    data = r.data.astype(np.float32, copy=False)
    if r.nodata is not None and not (isinstance(r.nodata, float)
                                     and math.isnan(r.nodata)):
        data = np.where(data == r.nodata, np.nan, data)
    out = np.stack([_reproject(np.ascontiguousarray(b), r, grid, how, np.nan)
                    for b in data])
    valid = None
    if how == "average":
        # share of valid pixels among the source pixels inside each cell
        # (average of a 1/0 mask); NaN where the source does not reach.
        ok = np.isfinite(data).any(axis=0).astype(np.float32)
        inside = np.ones_like(ok)
        num = _reproject(ok, r, grid, "average", None)
        cov = _reproject(inside, r, grid, "average", None)
        valid = np.where(cov > 0, num, np.nan).astype(np.float32)
    return out, valid, how


def regrid_categorical(r: Raster, grid: TargetGrid, method: str | None = None
                       ) -> tuple[np.ndarray, np.ndarray, list[int], str]:
    """→ (fractions (class, ny, nx), mode (ny, nx), classes, method).

    Only the first band is used (one class map per Raster). Classes not in
    `r.classes` are pooled into the last fraction, "other", so fractions
    of valid area always sum to 1."""
    how = method or method_for("categorical", native_res_m(r), grid.res)
    band = r.data[0]
    nod = r.nodata
    valid = np.ones(band.shape, bool) if nod is None else band != nod
    if np.issubdtype(band.dtype, np.floating):
        valid &= np.isfinite(band)
    classes = (list(r.classes) if r.classes is not None
               else sorted(int(c) for c in np.unique(band[valid])))
    known = np.isin(band, classes) & valid
    layers = [np.where(valid, band == c, np.nan).astype(np.float32)
              for c in classes]
    layers.append(np.where(valid, valid & ~known, np.nan).astype(np.float32))
    frac_how = "average" if how == "fractions" else "nearest"
    fr = np.stack([_reproject(l, r, grid, frac_how, np.nan) for l in layers])
    mode_src = np.where(valid, band.astype(np.int32), -1).astype(np.int32)
    mode = _reproject(mode_src, r, grid,
                      "mode" if how == "fractions" else "nearest",
                      -1, dst_dtype=np.int32, dst_nodata=-1)
    return fr, mode, classes, how


def mosaic_first(stack: Iterable[np.ndarray]) -> np.ndarray:
    """First finite value along a sequence of same-shape arrays (tiles)."""
    out = None
    for a in stack:
        out = a.copy() if out is None else np.where(np.isfinite(out), out, a)
    return out


# --- points and polygons ----------------------------------------------------
def bin_points(lon: np.ndarray, lat: np.ndarray, values: dict[str, np.ndarray],
               grid: TargetGrid, pts_crs: str = "EPSG:4326"
               ) -> dict[str, np.ndarray]:
    """Footprints / gauges → per-cell mean of each value + `count`.

    For sparse samplers (GEDI shots are ~25 m and 60 m apart along track,
    tracks ~600 m apart): most 1 km cells get 0 shots in one granule —
    NaN, not 0 — so carry `count` as the weight."""
    from pyproj import Transformer
    x, y = Transformer.from_crs(pts_crs, grid.crs,
                                always_xy=True).transform(lon, lat)
    row, col, inside = grid.cell_index(np.asarray(x), np.asarray(y))
    ny, nx = grid.shape
    flat = (row * nx + col)[inside]
    cnt = np.bincount(flat, minlength=ny * nx).astype(np.float32)
    out = {"count": cnt.reshape(ny, nx)}
    for name, v in values.items():
        v = np.asarray(v, np.float64)[inside]
        ok = np.isfinite(v)
        s = np.bincount(flat[ok], weights=v[ok], minlength=ny * nx)
        n = np.bincount(flat[ok], minlength=ny * nx)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(n > 0, s / n, np.nan).reshape(ny, nx
                                                               ).astype(np.float32)
    return out


def _block(a: np.ndarray, f: int, how: str) -> np.ndarray:
    ny, nx = a.shape[0] // f, a.shape[1] // f
    b = a[:ny * f, :nx * f].reshape(ny, f, nx, f)
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks → NaN
        return {"mean": np.nanmean, "sum": np.nansum}[how](b, axis=(1, 3))


def rasterize_cover(geoms: Sequence, grid: TargetGrid, factor: int = 10
                    ) -> np.ndarray:
    """Fraction of each cell covered by the union of polygons (in grid CRS),
    estimated on a factor² sub-lattice (100 m for factor=10)."""
    from rasterio.features import rasterize
    fg = grid.fine(factor)
    if not len(geoms):
        return np.zeros(grid.shape, np.float32)
    m = rasterize(((g, 1) for g in geoms), out_shape=fg.shape,
                  transform=fg.transform, fill=0, dtype="uint8")
    return _block(m.astype(np.float32), factor, "mean").astype(np.float32)


def rasterize_values(geoms: Sequence, values: Sequence[float],
                     grid: TargetGrid, factor: int = 10) -> np.ndarray:
    """Area-weighted mean of a per-polygon value over each cell (NaN where
    no polygon) — polygons burned at 100 m, then block-averaged."""
    from rasterio.features import rasterize
    fg = grid.fine(factor)
    if not len(geoms):
        return np.full(grid.shape, np.nan, np.float32)
    m = rasterize(((g, float(v)) for g, v in zip(geoms, values)
                   if v is not None and np.isfinite(v)),
                  out_shape=fg.shape, transform=fg.transform,
                  fill=np.nan, dtype="float32")
    return _block(m, factor, "mean").astype(np.float32)
