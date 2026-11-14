"""Footprint-aware point extraction — kernel-weighted areal sampling.

Satellite pixels are areal averages over the sensor footprint, not point
values: nearest-pixel extraction keeps the whole representativeness
mismatch between a field point and a 200 m–36 km cell. `sample` replaces
it with a Gaussian kernel over the nominal footprint — the sensor's own
spatial response — so the estimate is a convex weighted mean of the cells
the instrument actually integrates. `coverage` reports the fraction of
kernel mass that lands on valid retrievals, which is stored as the
series' per-observation weight.

`radius_km=0` degenerates to nearest-pixel, used for fine grids where the
footprint is already smaller than the field.
"""

from __future__ import annotations

import numpy as numpy
import xarray as xarray

KM_PER_DEG = 111.195


_COORD_NAMES = (("lat", "lon"), ("latitude", "longitude"),
                ("cell_lat", "cell_lon"))


def _coords(ds: xarray.Dataset, var: str) -> tuple[numpy.ndarray, numpy.ndarray]:
    """(lat, lon) arrays broadcast onto the var's spatial grid."""
    data = ds[var]
    for la, low in _COORD_NAMES:
        if la in ds and low in ds:
            latitude, longitude = ds[la].values, ds[low].values
            if latitude.ndim == 1 and la in data.dims and low in data.dims:
                latitude, longitude = numpy.meshgrid(latitude, longitude, indexing="ij")
                if data.dims.index(la) > data.dims.index(low):
                    latitude, longitude = latitude.T, longitude.T
            return numpy.asarray(latitude, numpy.float64), numpy.asarray(longitude, numpy.float64)
    raise KeyError(f"no lat/lon coords among {[v for v in ds][:12]}")


def distance(glat: numpy.ndarray, glon: numpy.ndarray,
            latitude: float, longitude: float) -> numpy.ndarray:
    """Equirectangular great-circle distance, km — exact within footprints."""
    dy = (glat - latitude) * KM_PER_DEG
    dx = (glon - longitude) * KM_PER_DEG * numpy.cos(numpy.deg2rad(latitude))
    return numpy.hypot(dx, dy)


def kernel(d: numpy.ndarray, radius_km: float) -> numpy.ndarray:
    """Gaussian footprint response; hard cutoff at the nominal radius."""
    sigma = radius_km / 2.0
    return numpy.where(d <= radius_km, numpy.exp(-0.5 * (d / sigma) ** 2), 0.0)


def _window(ds: xarray.Dataset, var: str,
            bbox: tuple[float, float, float, float]) -> tuple | None:
    """(row slice, col slice, glat, glon) covering bbox — found on a strided
    coordinate subsample so remote grids cost ~1/64th of a coord read."""
    west, south, east, north = bbox
    for la, lo in _COORD_NAMES:
        if la not in ds or lo not in ds:
            continue
        la_da, lo_da = ds[la], ds[lo]
        if la_da.ndim == 1 and lo_da.ndim == 1:
            lat, lon = la_da.values, lo_da.values
            rows = numpy.where((lat >= south) & (lat <= north))[0]
            cols = numpy.where((lon >= west) & (lon <= east))[0]
            if not rows.size or not cols.size:
                return None
            rs = slice(rows[0], rows[-1] + 1)
            cs = slice(cols[0], cols[-1] + 1)
            glat, glon = numpy.meshgrid(lat[rs], lon[cs], indexing="ij")
            da = ds[var]
            if da.dims.index(la) > da.dims.index(lo):
                glat, glon = glat.T, glon.T
            return rs, cs, glat, glon
        step = max(1, min(la_da.shape) // 256)
        slat = la_da[::step, ::step].values
        slon = lo_da[::step, ::step].values
        m = (slat >= south) & (slat <= north) \
            & (slon >= west) & (slon <= east)
        if not m.any():
            return None
        rows = numpy.where(m.any(axis=1))[0]
        cols = numpy.where(m.any(axis=0))[0]
        rs = slice(max(rows[0] * step - step, 0),
                   min(rows[-1] * step + step + 1, la_da.shape[0]))
        cs = slice(max(cols[0] * step - step, 0),
                   min(cols[-1] * step + step + 1, la_da.shape[1]))
        return rs, cs, la_da[rs, cs].values, lo_da[rs, cs].values
    return None


def prep(ds: xarray.Dataset, var: str,
         confidence: numpy.ndarray | xarray.DataArray | None = None,
         bbox: tuple[float, float, float, float] | None = None) -> tuple:
    """Pre-materialized (values, glat, glon, conf) grid — one dataset read
    for any number of point queries via `at`. `confidence` may be a ready
    array or the QA-band DataArray itself — either way it is cropped with
    `bbox` (w, s, e, n) so remote reads cost only the local slice."""
    if bbox is not None and (win := _window(ds, var, bbox)) is not None:
        rs, cs, glat, glon = win
        values = ds[var][..., rs, cs].values.astype(numpy.float64)
        if confidence is None:
            c = None
        elif isinstance(confidence, xarray.DataArray):
            c = confidence[..., rs, cs].values
        else:
            c = confidence[..., rs, cs]
        return values, glat, glon, c
    c = None if confidence is None else numpy.asarray(
        confidence.values if isinstance(confidence, xarray.DataArray)
        else confidence, numpy.float64)
    glat, glon = _coords(ds, var)
    values = numpy.asarray(ds[var].values, numpy.float64)
    return values, glat, glon, c


def at(grid: tuple, latitude: float, longitude: float,
       radius_km: float) -> tuple[float, float]:
    """(footprint-weighted value, coverage) at one point on a `prep` grid —
    the dist/kernel math runs once per call instead of twice."""
    v, glat, glon, c = grid
    w = kernel(distance(glat, glon, latitude, longitude), radius_km)
    c = numpy.ones_like(w) if c is None else c
    ok = numpy.isfinite(v) & (c > 0.0) & (w > 0.0)
    value = float((w[ok] * c[ok] * v[ok]).sum() / (w[ok] * c[ok]).sum()) \
        if ok.any() else float("nan")
    total = w.sum()
    covariance = float((w * c * ok).sum() / total) if total > 0.0 else 0.0
    return value, covariance


def sample(ds: xarray.Dataset, var: str, latitude: float, longitude: float,
           radius_km: float = 0.0, confidence: numpy.ndarray | None = None) -> float:
    """Footprint-weighted estimate at (lat, lon); NaN when nothing valid."""
    if radius_km <= 0.0:
        return _nearest(ds, var, latitude, longitude)
    return at(prep(ds, var, confidence), latitude, longitude, radius_km)[0]


def coverage(ds: xarray.Dataset, var: str, latitude: float, longitude: float,
             radius_km: float, confidence: numpy.ndarray | None = None) -> float:
    """Fraction of kernel mass over valid cells — the stored obs weight."""
    return at(prep(ds, var, confidence), latitude, longitude, radius_km)[1]


def _nearest(ds: xarray.Dataset, var: str, latitude: float, longitude: float) -> float:
    """Nearest-point sample — regular (lat/lon dims) or 2-D grids."""
    data = ds[var]
    if "lat" in data.dims and "lon" in data.dims:
        return float(data.sel(lat=latitude, lon=longitude, method="nearest").values.ravel()[0])
    glat, glon = _coords(ds, var)
    flat = numpy.nanargmin((glat - latitude) ** 2 + (glon - longitude) ** 2)
    return float(data.values.flat[flat])
