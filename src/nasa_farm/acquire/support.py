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

import numpy as np
import xarray as xr

KM_PER_DEG = 111.195


_COORD_NAMES = (("lat", "lon"), ("latitude", "longitude"),
                ("cell_lat", "cell_lon"))


def _coords(ds: xr.Dataset, var: str) -> tuple[np.ndarray, np.ndarray]:
    """(lat, lon) arrays broadcast onto the var's spatial grid."""
    da = ds[var]
    for la, lo in _COORD_NAMES:
        if la in ds and lo in ds:
            lat, lon = ds[la].values, ds[lo].values
            if lat.ndim == 1 and la in da.dims and lo in da.dims:
                lat, lon = np.meshgrid(lat, lon, indexing="ij")
                if da.dims.index(la) > da.dims.index(lo):
                    lat, lon = lat.T, lon.T
            return np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    raise KeyError(f"no lat/lon coords among {[v for v in ds][:12]}")


def dist_km(glat: np.ndarray, glon: np.ndarray,
            lat: float, lon: float) -> np.ndarray:
    """Equirectangular great-circle distance, km — exact within footprints."""
    dy = (glat - lat) * KM_PER_DEG
    dx = (glon - lon) * KM_PER_DEG * np.cos(np.deg2rad(lat))
    return np.hypot(dx, dy)


def kernel(d: np.ndarray, radius_km: float) -> np.ndarray:
    """Gaussian footprint response; hard cutoff at the nominal radius."""
    sigma = radius_km / 2.0
    return np.where(d <= radius_km, np.exp(-0.5 * (d / sigma) ** 2), 0.0)


def _grid(ds: xr.Dataset, var: str, lat: float, lon: float,
          radius_km: float, conf: np.ndarray | None,
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(values, kernel weights, per-pixel confidence) on the var's grid."""
    glat, glon = _coords(ds, var)
    w = kernel(dist_km(glat, glon, lat, lon), radius_km)
    c = np.ones_like(w) if conf is None else np.asarray(conf, np.float64)
    return np.asarray(ds[var].values, np.float64), w, c


def sample(ds: xr.Dataset, var: str, lat: float, lon: float,
           radius_km: float = 0.0, conf: np.ndarray | None = None) -> float:
    """Footprint-weighted estimate at (lat, lon); NaN when nothing valid."""
    if radius_km <= 0.0:
        return _nearest(ds, var, lat, lon)
    v, w, c = _grid(ds, var, lat, lon, radius_km, conf)
    ok = np.isfinite(v) & (c > 0.0) & (w > 0.0)
    if not ok.any():
        return float("nan")
    return float((w[ok] * c[ok] * v[ok]).sum() / (w[ok] * c[ok]).sum())


def coverage(ds: xr.Dataset, var: str, lat: float, lon: float,
             radius_km: float, conf: np.ndarray | None = None) -> float:
    """Fraction of kernel mass over valid cells — the stored obs weight."""
    v, w, c = _grid(ds, var, lat, lon, radius_km, conf)
    tot = w.sum()
    if tot <= 0.0:
        return 0.0
    ok = np.isfinite(v) & (c > 0.0)
    return float((w * c * ok).sum() / tot)


def _nearest(ds: xr.Dataset, var: str, lat: float, lon: float) -> float:
    """Nearest-point sample — regular (lat/lon dims) or 2-D grids."""
    da = ds[var]
    if "lat" in da.dims and "lon" in da.dims:
        return float(da.sel(lat=lat, lon=lon, method="nearest").values.ravel()[0])
    glat, glon = _coords(ds, var)
    flat = np.nanargmin((glat - lat) ** 2 + (glon - lon) ** 2)
    return float(da.values.flat[flat])
