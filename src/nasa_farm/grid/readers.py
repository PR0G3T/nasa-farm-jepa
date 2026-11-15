"""Format readers → `Raster` (native grid, window around the target grid).

Every reader reads only the window the target grid needs (+ a few source
pixels of buffer for interpolation), converts fill values to NaN and
applies scale/offset, so `regrid` never sees raw codes.

- `read_tif` — GeoTIFF / COG, local or remote (https, /vsis3): ECOSTRESS,
  HLS, S1, S2, MODIS (PC), 3DEP, gNATSGO, CDL, NLCD, SOLUS, SoilGrids,
  GAEZ, LANDFIRE, GEDI L3, OPERA
- `read_latlon` — netCDF / HDF5 on a regular lat/lon grid: NLDAS-2/3,
  GLDAS, GRACE-DA, ERA5-Land
- `read_ease2` — SMAP HDF5 on EASE-2 (grid derived from the file's own
  lat/lon): SPL4SMGP, SPL4CMDL, SPL3FTP
- `read_sinusoidal_h5` — HDF-EOS5 sinusoidal tiles (grid from
  StructMetadata.0): VIIRS VNP21/VNP09
- `read_gedi` — GEDI L2A/L2B/L4A beam groups → points
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from nasa_farm.grid.regrid import Raster
from nasa_farm.grid.target import TargetGrid

SINUSOIDAL = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
"""MODIS/VIIRS land tile projection (sphere R = 6371007.181 m)."""


class NoOverlap(RuntimeError):
    """The source does not intersect the target grid."""


def _fill_to_nan(a: np.ndarray, fill=None, valid_range=None, scale=None, offset=None) -> np.ndarray:
    a = a.astype(np.float32)
    bad = ~np.isfinite(a)
    if fill is not None:
        for f in np.atleast_1d(fill):
            bad |= a == np.float32(f)
    if valid_range is not None:
        lo, hi = valid_range
        if lo is not None:
            bad |= a < lo
        if hi is not None:
            bad |= a > hi
    if scale is not None:
        a = a * np.float32(scale)
    if offset is not None:
        a = a + np.float32(offset)
    a[bad] = np.nan
    return a


# --- GeoTIFF / COG --------------------------------------------------------
def read_tif(
    href: str,
    grid: TargetGrid,
    *,
    name: str | Sequence[str] = "v",
    kind: str = "continuous",
    band: int | Sequence[int] | None = None,
    buffer_px: int = 2,
    scale: float | None = None,
    offset: float | None = None,
    valid_range=None,
    classes=None,
    class_names=None,
    units: str | None = None,
    env: dict | None = None,
) -> Raster:
    """Window of a (cloud-optimised) GeoTIFF covering `grid`.

    Continuous bands → float32 with NaN; scale/offset default to the file's
    own (`ds.scales/offsets`) when not given. Categorical → raw codes."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    with rasterio.Env(**(env or {})), rasterio.open(href) as ds:
        if ds.crs is None:
            raise ValueError(f"{href}: no CRS")
        b = transform_bounds(
            grid.crs, ds.crs, grid.west, grid.south, grid.east, grid.north, densify_pts=21
        )
        win = from_bounds(*b, transform=ds.transform)
        win = Window(
            math.floor(win.col_off) - buffer_px,
            math.floor(win.row_off) - buffer_px,
            math.ceil(win.width) + 2 * buffer_px + 1,
            math.ceil(win.height) + 2 * buffer_px + 1,
        )
        full = Window(0, 0, ds.width, ds.height)
        try:
            win = win.intersection(full)
        except Exception as e:  # rasterio raises WindowError when disjoint
            raise NoOverlap(href) from e
        idx = (
            [band]
            if isinstance(band, int)
            else list(band)
            if band is not None
            else list(range(1, ds.count + 1))
        )
        data = ds.read(idx, window=win)
        tr = ds.window_transform(win)
        nod = ds.nodata
        sc = [ds.scales[i - 1] for i in idx]
        of = [ds.offsets[i - 1] for i in idx]
        crs = ds.crs
    names = [name] if isinstance(name, str) else list(name)
    if len(names) == 1 and len(idx) > 1:
        names = [f"{names[0]}_{i}" for i in idx]
    if kind == "categorical":
        return Raster(
            data, tr, crs, names, kind, nodata=nod, classes=classes, class_names=class_names
        )
    out = np.stack(
        [
            _fill_to_nan(
                data[k],
                nod,
                valid_range,
                scale if scale is not None else (sc[k] if sc[k] != 1 else None),
                offset if offset is not None else (of[k] if of[k] != 0 else None),
            )
            for k in range(len(idx))
        ]
    )
    return Raster(out, tr, crs, names, kind, units=dict.fromkeys(names, units) if units else {})


# --- regular lat/lon netCDF ---------------------------------------------
_LAT = ("lat", "latitude", "Latitude", "y")
_LON = ("lon", "longitude", "Longitude", "x")


def _axis_transform(lat: np.ndarray, lon: np.ndarray):
    """Affine for cell-centre 1-D coords (any order) → (transform, flipy)."""
    from affine import Affine

    dlon = np.diff(lon)
    dlat = np.diff(lat)
    if not (np.allclose(dlon, dlon[0], rtol=1e-3) and np.allclose(dlat, dlat[0], rtol=1e-3)):
        raise ValueError("lat/lon not regular")
    rx, ry = abs(float(dlon.mean())), abs(float(dlat.mean()))
    flip = lat[0] < lat[-1]  # ascending latitude → flip rows to north-up
    north = float(lat.max()) + ry / 2
    west = float(lon.min()) - rx / 2
    return Affine(rx, 0, west, 0, -ry, north), flip


def read_latlon(
    ds,
    grid: TargetGrid,
    variables: Iterable[str] | None = None,
    *,
    buffer_deg: float = 0.5,
    skip: Iterable[str] = (),
    rename: dict[str, str] | None = None,
    scale: dict[str, float] | None = None,
) -> Raster:
    """xarray Dataset on a regular lat/lon grid → one Raster.

    Every non-spatial dim (time, depth, ensemble…) becomes bands; names are
    `var` (single slice) or `var_<dim><value>`, times are kept per band.
    Longitudes in 0–360 are wrapped to −180–180."""
    la = next(n for n in _LAT if n in ds.coords or n in ds.variables)
    lo = next(n for n in _LON if n in ds.coords or n in ds.variables)
    if float(ds[lo].max()) > 180:
        ds = ds.assign_coords({lo: ((ds[lo] + 180) % 360) - 180}).sortby(lo)
    b = grid.bbox_ll()
    lat = ds[la].values
    ds = ds.sel(
        {
            lo: slice(b.west - buffer_deg, b.east + buffer_deg),
            la: (
                slice(b.south - buffer_deg, b.north + buffer_deg)
                if lat[0] < lat[-1]
                else slice(b.north + buffer_deg, b.south - buffer_deg)
            ),
        }
    )
    if ds.sizes[la] < 2 or ds.sizes[lo] < 2:
        raise NoOverlap("lat/lon subset is empty")
    tr, flip = _axis_transform(ds[la].values, ds[lo].values)
    skip = set(skip)
    names = (
        list(variables)
        if variables is not None
        else [
            v
            for v, da in ds.data_vars.items()
            if la in da.dims
            and lo in da.dims
            and v not in skip
            and np.issubdtype(da.dtype, np.number)
        ]
    )
    arrs, bands, times, units = [], [], [], {}
    for v in names:
        if v not in ds:
            continue
        da = ds[v]
        other = [d for d in da.dims if d not in (la, lo)]
        da = da.transpose(*other, la, lo)
        stacked = da.stack(_b=other) if other else da.expand_dims(_b=[0])
        out_name = (rename or {}).get(v, v)
        units[out_name] = str(da.attrs.get("units", ""))
        for k in range(stacked.sizes["_b"]):
            sl = stacked.isel(_b=k)
            a = np.asarray(sl.values, np.float32)
            if scale and v in scale:
                a = a * np.float32(scale[v])
            if flip:
                a = a[::-1]
            arrs.append(a)
            t, label = None, out_name
            for d in other:
                val = sl[d].values
                if np.issubdtype(np.asarray(val).dtype, np.datetime64):
                    t = str(np.datetime_as_string(val, unit="s"))
                elif stacked.sizes["_b"] > 1:
                    label += f"_{d}{np.asarray(val).item()}"
            bands.append(label)
            times.append(t)
    if not arrs:
        raise KeyError(f"none of {names} on the lat/lon grid")
    return Raster(np.stack(arrs), tr, "EPSG:4326", bands, times=times, units=units)


# --- EASE-2 (SMAP) -------------------------------------------------------
EASE2_GLOBAL = "EPSG:6933"
EASE2_NORTH = "EPSG:6931"


def ease2_transform(lat2d: np.ndarray, lon2d: np.ndarray, fill=-9999.0):
    """Exact EASE-2 affine from the file's own cell-centre lat/lon.

    Global EASE-2 (EPSG:6933) is cylindrical: x depends on lon only, y on lat
    only, so one robust value per column/row pins the grid, and fill values
    (ocean cells of L3 products) are skipped. Deriving it from the data
    avoids hard-coding the 3/9/36 km grid constants. Polar EASE-2 grids are
    not separable and are rejected (raise) rather than guessed."""
    from affine import Affine
    from pyproj import Transformer

    lat = np.where(lat2d <= fill + 1, np.nan, lat2d).astype(np.float64)
    lon = np.where(lon2d <= fill + 1, np.nan, lon2d).astype(np.float64)
    for crs in (EASE2_GLOBAL,):
        t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = t.transform(lon, lat)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                xc = np.nanmedian(x, axis=0)
                yr = np.nanmedian(y, axis=1)
        ci = np.flatnonzero(np.isfinite(xc))
        ri = np.flatnonzero(np.isfinite(yr))
        if len(ci) < 2 or len(ri) < 2:
            continue
        rx = np.polyfit(ci, xc[ci], 1)
        ry = np.polyfit(ri, yr[ri], 1)
        resid = max(
            np.nanmax(np.abs(np.polyval(rx, ci) - xc[ci])),
            np.nanmax(np.abs(np.polyval(ry, ri) - yr[ri])),
        )
        if resid < 0.01 * abs(rx[0]):  # regular to 1 % of a cell
            dx, x0 = rx
            dy, y0 = ry
            return Affine(dx, 0, x0 - dx / 2, 0, dy, y0 - dy / 2), crs
    raise ValueError("lat/lon is not a regular EASE-2 global grid")


def _h5_attrs(ds) -> dict:
    out = {}
    for k, v in ds.attrs.items():
        v = v.decode() if isinstance(v, bytes) else v
        if isinstance(v, np.ndarray) and v.size == 1:
            v = v.item()
        out[k] = v
    return out


def _h5_values(ds, sl) -> np.ndarray:
    a = ds[sl]
    at = _h5_attrs(ds)
    vr = None
    if "valid_min" in at or "valid_max" in at:
        vr = (at.get("valid_min"), at.get("valid_max"))
    elif "valid_range" in at:
        vr = tuple(np.asarray(at["valid_range"]).ravel()[:2])
    return _fill_to_nan(a, at.get("_FillValue"), vr, at.get("scale_factor"), at.get("add_offset"))


def _first_valid_pass(a: np.ndarray, fill: float = -9999.0) -> np.ndarray:
    """(y, x) passes through unchanged; (passes, y, x) → (y, x) taking, per
    cell, the first pass whose value is not fill."""
    if a.ndim == 2:
        return a
    out = a[0].copy()
    for p in a[1:]:
        bad = out <= fill + 1
        out[bad] = p[bad]
    return out


def read_ease2(
    h5,
    grid: TargetGrid,
    variables: dict[str, str],
    lat: str,
    lon: str,
    *,
    buffer_px: int = 2,
    time: str | None = None,
    select: dict[str, int] | None = None,
) -> Raster:
    """SMAP HDF5 (h5py File / group) → Raster on the native EASE-2 grid.

    `variables` maps output name → dataset path; `lat`/`lon` are the 2-D
    coordinate datasets on the same grid. `select` picks a slice along a
    leading axis (e.g. {"freeze_thaw": 0} for the AM pass of SPL3FTP)."""
    from affine import Affine
    from rasterio.warp import transform_bounds

    la, lo = h5[lat][()], h5[lon][()]  # 2 × ~25 MB for the 9 km global grid
    # SPL3FTP v004 stores coordinates per pass: (am_pm, y, x), each with its
    # own fill cells — merge to one 2-D field (first valid pass per cell).
    la, lo = _first_valid_pass(la), _first_valid_pass(lo)
    ny, nx = la.shape
    tr, crs = ease2_transform(la, lo)
    b = transform_bounds(
        grid.crs, crs, grid.west, grid.south, grid.east, grid.north, densify_pts=21
    )
    c0 = max(int((b[0] - tr.c) // tr.a) - buffer_px, 0)
    c1 = min(int((b[2] - tr.c) // tr.a) + buffer_px + 1, nx)
    r0 = max(int((b[3] - tr.f) // tr.e) - buffer_px, 0)
    r1 = min(int((b[1] - tr.f) // tr.e) + buffer_px + 1, ny)
    if c1 <= c0 or r1 <= r0:
        raise NoOverlap("EASE-2 window empty")
    arrs, names = [], []
    for out, path in variables.items():
        d = h5[path]
        sl: tuple = (slice(r0, r1), slice(c0, c1))
        if d.ndim == 3:
            k = (select or {}).get(out, 0)
            # leading or trailing extra axis (SPL3FTP: (am_pm, y, x))
            sl = (k, *sl) if d.shape[1:] == (ny, nx) else (*sl, k)
        arrs.append(_h5_values(d, sl))
        names.append(out)
    wtr = Affine(tr.a, 0, tr.c + c0 * tr.a, 0, tr.e, tr.f + r0 * tr.e)
    return Raster(np.stack(arrs), wtr, crs, names, times=[time] * len(names))


# --- HDF-EOS5 sinusoidal (VIIRS) -----------------------------------------
_GRID_RE = re.compile(r"GROUP=GRID_\d+(.*?)END_GROUP=GRID_\d+", re.S)


def parse_struct_metadata(text: str) -> dict[str, dict[str, Any]]:
    """HDF-EOS StructMetadata.0 → {grid name: {xdim, ydim, ul, lr}}."""
    out = {}
    for block in _GRID_RE.findall(text):

        def g(k, block=block):
            return re.search(rf"{k}=([^\n]+)", block).group(1).strip()

        name = g("GridName").strip('"')

        def nums(s):
            return tuple(float(v) for v in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s))

        out[name] = {
            "xdim": int(g("XDim")),
            "ydim": int(g("YDim")),
            "ul": nums(g("UpperLeftPointMtrs")),
            "lr": nums(g("LowerRightMtrs")),
        }
    return out


def sinusoidal_transform(meta: dict[str, Any]):
    from affine import Affine

    (ulx, uly), (lrx, lry) = meta["ul"], meta["lr"]
    return Affine((lrx - ulx) / meta["xdim"], 0, ulx, 0, (lry - uly) / meta["ydim"], uly)


def read_sinusoidal_h5(
    h5,
    grid: TargetGrid,
    fields: Sequence[str] | str,
    *,
    qc: dict[str, Any] | None = None,
    time: str | None = None,
) -> list[Raster]:
    """VIIRS HDF-EOS5 tile → one Raster per HDF-EOS grid that holds a
    requested field. `fields` are dataset names or regexes; `qc` maps an
    output field to (qc_field, callable(qc_array) → bool mask of *bad*)."""
    meta = h5["HDFEOS INFORMATION/StructMetadata.0"][()]
    if isinstance(meta, np.ndarray):  # 0-d / 1-element string array
        meta = meta.ravel()[0]
    meta = parse_struct_metadata(meta.decode() if isinstance(meta, bytes) else str(meta))
    pats = [re.compile(f"^{f}$") for f in ([fields] if isinstance(fields, str) else fields)]
    rasters = []
    for gname, gm in meta.items():
        base = f"HDFEOS/GRIDS/{gname}/Data Fields"
        if base not in h5:
            continue
        names = [n for n in h5[base] if any(p.match(n) for p in pats)]
        if not names:
            continue
        arrs = []
        for n in names:
            a = _h5_values(h5[f"{base}/{n}"], ())
            if qc and n in qc:
                qn, bad = qc[n]
                a[bad(h5[f"{base}/{qn}"][()])] = np.nan
            arrs.append(a)
        rasters.append(
            Raster(
                np.stack(arrs),
                sinusoidal_transform(gm),
                SINUSOIDAL,
                names,
                times=[time] * len(names),
            )
        )
    if not rasters:
        raise KeyError(f"no field matching {fields}")
    return rasters


# --- GEDI footprints -------------------------------------------------------
def read_gedi(
    h5,
    grid: TargetGrid,
    fields: dict[str, tuple[str, int | None]],
    *,
    lat: str = "lat_lowestmode",
    lon: str = "lon_lowestmode",
    quality: str | None = "quality_flag",
    buffer_deg: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """All beams → (lon, lat, {name: values}) of quality shots in the AOI.

    `fields` maps output name → (dataset path inside the beam, column or
    None), e.g. {"rh98": ("rh", 98)}. Only the contiguous index span that
    falls in the AOI is read from each dataset."""
    b = grid.bbox_ll()
    LON, LAT, VAL = [], [], {k: [] for k in fields}
    beams = [k for k in h5 if k.startswith("BEAM")]
    if beams and not any(lat in h5[k] for k in beams):
        # a wrong path must not read as "no shots" (L2B V003 moved lat/lon)
        raise KeyError(f"GEDI: no beam has {lat!r}")
    if quality and beams and not any(quality in h5[k] for k in beams):
        raise KeyError(f"GEDI: no beam has quality flag {quality!r}")
    for beam in beams:
        g = h5[beam]
        if lat not in g:
            continue
        la, lo = g[lat][()], g[lon][()]
        m = (
            (lo >= b.west - buffer_deg)
            & (lo <= b.east + buffer_deg)
            & (la >= b.south - buffer_deg)
            & (la <= b.north + buffer_deg)
        )
        if not m.any():
            continue
        i0, i1 = np.flatnonzero(m)[[0, -1]]
        sl = slice(int(i0), int(i1) + 1)
        mm = m[sl].copy()
        if quality and quality in g:
            mm &= g[quality][sl] == 1
        for deg in ("degrade_flag", "geolocation/degrade_flag"):  # L2A/L4A · L2B
            if deg in g:
                mm &= g[deg][sl] == 0
                break
        LON.append(lo[sl][mm])
        LAT.append(la[sl][mm])
        for name, (path, col) in fields.items():
            d = g[path]
            a = d[sl] if col is None else d[sl, col]
            v = _fill_to_nan(a, _h5_attrs(d).get("_FillValue", -9999))
            VAL[name].append(v[mm])
    if not LON:
        raise NoOverlap("no GEDI shots in AOI")
    return (
        np.concatenate(LON),
        np.concatenate(LAT),
        {k: np.concatenate(v) for k, v in VAL.items()},
    )
