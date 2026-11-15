"""Common training grid — one equal-area lattice shared by every product.

Why EPSG:5070 (CONUS Albers, NAD83): it is the native CRS of CDL, NLCD,
CSB and LANDFIRE, it is equal-area (a 1 km cell is 1 km² everywhere in the
lower 48, so area-weighted averages and class fractions are exact), and
snapping cell edges to multiples of `res` from the CRS origin means grids
built for different AOIs tile seamlessly — two AOIs never disagree on
where a cell is. Outside CONUS swap the CRS (e.g. EPSG:6933 EASE-2 global,
also equal-area); nothing else in the pipeline depends on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nasa_farm.acquire.base import Bounds

AOIS: dict[str, tuple[float, float, float]] = {
    "ames": (42.0, -93.6, 15.0),  # 30 × 30 km test AOI (data-inventory.md)
}
"""name → (lat, lon, half-width km)."""


@dataclass(frozen=True, slots=True)
class TargetGrid:
    """Regular north-up grid; bounds are cell edges in `crs` units (m)."""

    west: float
    south: float
    east: float
    north: float
    res: float = 1000.0
    crs: str = "EPSG:5070"

    # --- construction ----------------------------------------------------
    @classmethod
    def _snapped(cls, x0: float, y0: float, x1: float, y1: float,
                 res: float, crs: str) -> "TargetGrid":
        r = res
        return cls(math.floor(x0 / r) * r, math.floor(y0 / r) * r,
                   math.ceil(x1 / r) * r, math.ceil(y1 / r) * r, r, crs)

    @classmethod
    def around(cls, lat: float, lon: float, half_km: float = 15.0,
               res: float = 1000.0, crs: str = "EPSG:5070") -> "TargetGrid":
        """Square of ±half_km around a point, snapped outward to the lattice."""
        from pyproj import Transformer
        cx, cy = Transformer.from_crs("EPSG:4326", crs,
                                      always_xy=True).transform(lon, lat)
        h = half_km * 1000.0
        return cls._snapped(cx - h, cy - h, cx + h, cy + h, res, crs)

    @classmethod
    def from_bbox(cls, bbox: Bounds, res: float = 1000.0,
                  crs: str = "EPSG:5070") -> "TargetGrid":
        """Lon/lat box → smallest snapped grid that contains it."""
        from rasterio.warp import transform_bounds
        b = transform_bounds("EPSG:4326", crs, *bbox.as_tuple(),
                             densify_pts=21)
        return cls._snapped(*b, res, crs)

    @classmethod
    def named(cls, name: str, res: float = 1000.0,
              crs: str = "EPSG:5070") -> "TargetGrid":
        lat, lon, half = AOIS[name]
        return cls.around(lat, lon, half, res, crs)

    # --- geometry ----------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return (round((self.north - self.south) / self.res),
                round((self.east - self.west) / self.res))

    @property
    def transform(self):
        from affine import Affine
        return Affine(self.res, 0.0, self.west, 0.0, -self.res, self.north)

    @property
    def x(self) -> np.ndarray:
        """Cell-centre x, west → east."""
        return self.west + self.res * (np.arange(self.shape[1]) + 0.5)

    @property
    def y(self) -> np.ndarray:
        """Cell-centre y, north → south (row 0 is the top)."""
        return self.north - self.res * (np.arange(self.shape[0]) + 0.5)

    def lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """2-D (lon, lat) of cell centres."""
        from pyproj import Transformer
        xx, yy = np.meshgrid(self.x, self.y)
        lon, lat = Transformer.from_crs(self.crs, "EPSG:4326",
                                        always_xy=True).transform(xx, yy)
        return np.asarray(lon), np.asarray(lat)

    def bbox_ll(self, buffer_km: float = 0.0) -> Bounds:
        """Lon/lat box covering the grid (+ buffer) — for searches and
        source windows. The buffer matters for bilinear upsampling of
        coarse products: a 25 km GLDAS cell centre can sit outside a 30 km
        AOI and still be needed to interpolate inside it."""
        from rasterio.warp import transform_bounds
        b = self.buffered(buffer_km)
        w, s, e, n = transform_bounds(self.crs, "EPSG:4326", b.west, b.south,
                                      b.east, b.north, densify_pts=21)
        return Bounds(w, s, e, n)

    def buffered(self, buffer_km: float) -> "TargetGrid":
        d = buffer_km * 1000.0
        return TargetGrid(self.west - d, self.south - d, self.east + d,
                          self.north + d, self.res, self.crs)

    def fine(self, factor: int) -> "TargetGrid":
        """Same extent, res / factor — exact sub-lattice for rasterizing."""
        return TargetGrid(self.west, self.south, self.east, self.north,
                          self.res / factor, self.crs)

    def cell_index(self, x: np.ndarray, y: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row, col, inside) of points given in grid CRS."""
        col = np.floor((np.asarray(x) - self.west) / self.res).astype(int)
        row = np.floor((self.north - np.asarray(y)) / self.res).astype(int)
        ny, nx = self.shape
        inside = (row >= 0) & (row < ny) & (col >= 0) & (col < nx)
        return row, col, inside

    def describe(self) -> dict:
        ny, nx = self.shape
        return {"crs": self.crs, "res_m": self.res, "ny": ny, "nx": nx,
                "bounds": [self.west, self.south, self.east, self.north]}
