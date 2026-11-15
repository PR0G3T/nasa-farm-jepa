"""Per-product output on the common grid → CF-compliant xarray Dataset.

Naming on disk (`<out>/<product>.nc`):
- continuous: one var per band name, (y, x) or (time, y, x) when the file
  has several time steps (hourly ERA5-Land);
- categorical `name`: `name_frac` (name_class, y, x) area fractions whose
  last class is "other", plus `name_mode` (y, x);
- `valid_frac` (y, x): share of the cell's source pixels that passed QA;
- points (gauges, samples): a `site` dimension with lon/lat/row/col.
Every var carries `method` (how it was regridded) and `native_res_m`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nasa_farm.grid.regrid import (
    Raster,
    mosaic_first,
    native_res_m,
    regrid_categorical,
    regrid_continuous,
)
from nasa_farm.grid.target import TargetGrid


class Out:
    def __init__(self, grid: TargetGrid) -> None:
        self.grid = grid
        self.vars: dict[str, tuple[tuple[str, ...], np.ndarray, dict]] = {}
        self.coords: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        self.attrs: dict[str, object] = {}
        self.points = None  # pandas.DataFrame

    # --- rasters -----------------------------------------------------------
    def continuous(
        self,
        rasters: Raster | Sequence[Raster],
        *,
        prefix: str = "",
        valid: str | None = "valid_frac",
        method: str | None = None,
    ) -> None:
        """Regrid one raster, or several tiles of one acquisition (mosaic:
        first valid value wins, tiles in the order given)."""
        rs = [rasters] if isinstance(rasters, Raster) else list(rasters)
        if not rs:
            raise ValueError("no raster")
        outs, valids, how = [], [], None
        for r in rs:
            o, v, how = regrid_continuous(r, self.grid, method)
            outs.append(o)
            if v is not None:
                valids.append(v)
        data = mosaic_first(outs)
        r0 = rs[0]
        res = native_res_m(r0)
        names = r0.bands
        for name in dict.fromkeys(names):
            idx = [k for k, n in enumerate(names) if n == name]
            attrs = {"method": how, "native_res_m": round(res, 1)}
            u = r0.units.get(name)
            if u:
                attrs["units"] = u
            key = prefix + name
            if len(idx) == 1:
                self.vars[key] = (("y", "x"), data[idx[0]], attrs)
                if r0.times[idx[0]]:
                    attrs["time"] = r0.times[idx[0]]
            else:
                times = np.array([r0.times[k] or str(k) for k in idx])
                dim = self._time_dim(times)
                self.vars[key] = ((dim, "y", "x"), data[idx], attrs)
        if valids and valid:
            self.vars[prefix + valid] = (
                ("y", "x"),
                mosaic_first(valids),
                {"method": "average of QA mask", "units": "1"},
            )

    def _time_dim(self, times: np.ndarray) -> str:
        for d, (_, t) in self.coords.items():
            if d.startswith("time") and len(t) == len(times) and (t == times).all():
                return d
        d = "time" if "time" not in self.coords else f"time{len(self.coords)}"
        self.coords[d] = ((d,), times)
        return d

    def categorical(
        self, rasters: Raster | Sequence[Raster], name: str, method: str | None = None
    ) -> None:
        rs = [rasters] if isinstance(rasters, Raster) else list(rasters)
        frs, modes, classes, how = [], [], None, None
        for r in rs:
            fr, mode, classes, how = regrid_categorical(r, self.grid, method)
            frs.append(fr)
            modes.append(np.where(mode < 0, np.nan, mode).astype(np.float32))
        fr = mosaic_first(frs)
        mode = mosaic_first(modes)
        cn = rs[0].class_names or {}
        labels = [f"{c}:{cn.get(c, c)}" for c in classes] + ["other"]
        dim = f"{name}_class"
        self.coords[dim] = ((dim,), np.array(labels))
        res = round(native_res_m(rs[0]), 1)
        self.vars[f"{name}_frac"] = (
            (dim, "y", "x"),
            fr,
            {
                "method": "class area fraction" if how == "fractions" else "nearest (one-hot)",
                "native_res_m": res,
                "units": "1",
            },
        )
        self.vars[f"{name}_mode"] = (
            ("y", "x"),
            mode,
            {"method": "mode" if how == "fractions" else "nearest", "native_res_m": res},
        )

    def add(self, name: str, arr: np.ndarray, **attrs) -> None:
        self.vars[name] = (("y", "x"), np.asarray(arr, np.float32), attrs)

    # --- export ------------------------------------------------------------
    def to_dataset(self):
        import xarray as xr
        from pyproj import CRS

        g = self.grid
        lon, lat = g.lonlat()
        crs = CRS.from_user_input(g.crs)
        sref = crs.to_cf()
        sref["crs_wkt"] = crs.to_wkt()
        t = g.transform
        sref["GeoTransform"] = " ".join(str(v) for v in (t.c, t.a, t.b, t.f, t.d, t.e))
        coords = {
            "x": ("x", g.x, {"units": "m", "standard_name": "projection_x_coordinate"}),
            "y": ("y", g.y, {"units": "m", "standard_name": "projection_y_coordinate"}),
            "lon": (("y", "x"), lon.astype(np.float64), {"units": "degrees_east"}),
            "lat": (("y", "x"), lat.astype(np.float64), {"units": "degrees_north"}),
            "spatial_ref": ((), 0, sref),
        }
        coords |= {d: (dims, v) for d, (dims, v) in self.coords.items()}
        data = {
            k: (dims, np.asarray(v, np.float32), dict(a, grid_mapping="spatial_ref"))
            for k, (dims, v, a) in self.vars.items()
        }
        ds = xr.Dataset(data, coords=coords)
        ds.attrs.update(
            {k: (v if isinstance(v, (str, int, float)) else str(v)) for k, v in self.attrs.items()}
        )
        ds.attrs["Conventions"] = "CF-1.10"
        return ds
