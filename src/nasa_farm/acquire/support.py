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


def dist_km(glat: numpy.ndarray, glon: numpy.ndarray,
            latitude: float, longitude: float) -> numpy.ndarray:
    """Equirectangular great-circle distance, km — exact within footprints."""
    dy = (glat - latitude) * KM_PER_DEG
    dx = (glon - longitude) * KM_PER_DEG * numpy.cos(numpy.deg2rad(latitude))
    return numpy.hypot(dx, dy)


def kernel(d: numpy.ndarray, radius_km: float) -> numpy.ndarray:
    """Gaussian footprint response; hard cutoff at the nominal radius."""
    sigma = radius_km / 2.0
    return numpy.where(d <= radius_km, numpy.exp(-0.5 * (d / sigma) ** 2), 0.0)


def prep(ds: xarray.Dataset, var: str,
         confidence: numpy.ndarray | None = None) -> tuple:
    """Pre-materialized (values, glat, glon, conf) grid — one dataset read
    for any number of point queries via `at`."""
    glat, glon = _coords(ds, var)
    return (numpy.asarray(ds[var].values, numpy.float64), glat, glon,
            None if confidence is None else numpy.asarray(confidence, numpy.float64))


def at(grid: tuple, latitude: float, longitude: float,
       radius_km: float) -> tuple[float, float]:
    """(footprint-weighted value, coverage) at one point on a `prep` grid —
    the dist/kernel math runs once per call instead of twice."""
    v, glat, glon, c = grid
    w = kernel(dist_km(glat, glon, latitude, longitude), radius_km)
    c = numpy.ones_like(w) if c is None else c
    ok = numpy.isfinite(v) & (c > 0.0) & (w > 0.0)
    val = float((w[ok] * c[ok] * v[ok]).sum() / (w[ok] * c[ok]).sum()) \
        if ok.any() else float("nan")
    total = w.sum()
    covariance = float((w * c * ok).sum() / total) if total > 0.0 else 0.0
    return val, covariance


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
        return float(data.sel(latitude=latitude, longitude=longitude, method="nearest").values.ravel()[0])
    glat, glon = _coords(ds, var)
    flat = numpy.nanargmin((glat - latitude) ** 2 + (glon - longitude) ** 2)
    return float(data.values.flat[flat])
