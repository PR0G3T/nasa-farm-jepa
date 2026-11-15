"""Product functions end-to-end on synthetic granules (no network).

The fetch layer is monkeypatched to hand each product function local
files / in-memory HDF5 stand-ins whose correct 1 km answer is known, so
the whole chain — read → QA mask → regrid → Dataset → NetCDF — runs as in
production. What this cannot check: live endpoints, credentials, and the
exact dataset names inside real granules (see the route confidence tags).
"""

from __future__ import annotations

import datetime as dt
from typing import ClassVar

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from affine import Affine  # noqa: E402
from pyproj import Transformer  # noqa: E402

from nasa_farm.grid import catalog as C  # noqa: E402
from nasa_farm.grid.fetch import Context  # noqa: E402
from nasa_farm.grid.output import Out  # noqa: E402
from nasa_farm.grid.target import TargetGrid  # noqa: E402

DATE = dt.date(2024, 7, 15)


# --- h5py stand-in ------------------------------------------------------------
class DS:
    def __init__(self, a, **attrs):
        self.a = np.asarray(a)
        self.attrs = attrs
        self.shape, self.ndim = self.a.shape, self.a.ndim

    def __getitem__(self, k):
        return self.a if k == () else self.a[k]


class G(dict):
    attrs: ClassVar[dict] = {}

    def _walk(self, path):
        node = self
        for part in path.strip("/").split("/"):
            node = dict.__getitem__(node, part)
        return node

    def __getitem__(self, path):
        return self._walk(path)

    def __contains__(self, path):
        try:
            self._walk(path)
        except KeyError, TypeError:
            return False
        else:
            return True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def grid():
    return TargetGrid.named("ames")


@pytest.fixture
def context(grid, tmp_path):
    return Context(grid, DATE, tmp_path / "raw", key="t")


def _tif(path, a, transform, crs, nodata=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=a.shape[0],
        width=a.shape[1],
        count=1,
        dtype=a.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as ds:
        ds.write(a, 1)
    return str(path)


def _utm_frame(grid, res, pad_km=5.0):
    t = Transformer.from_crs(grid.crs, "EPSG:32615", always_xy=True)
    xs, ys = t.transform(
        [grid.west, grid.east, grid.west, grid.east],
        [grid.south, grid.south, grid.north, grid.north],
    )
    x0, y1 = min(xs) - pad_km * 1e3, max(ys) + pad_km * 1e3
    nx = int((max(xs) - x0 + pad_km * 1e3) // res)
    ny = int((y1 - min(ys) + pad_km * 1e3) // res)
    return Affine(res, 0, x0, 0, -res, y1), (ny, nx)


def _dataset(out):
    ds = out.to_dataset()
    assert ds.sizes["y"] == out.grid.shape[0] and "spatial_ref" in ds.coords
    return ds


# --- products ------------------------------------------------------------------
def test_smap_l4_ease2_bilinear(monkeypatch, context):
    res, x0, y0 = 9008.055210146, -17367530.44516137, 7314540.830638504
    ax, ay = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform(-93.6, 42.0)
    ny, nx = 30, 40  # window centred on Ames
    r0 = int((y0 - ay) // res) - ny // 2
    c0 = int((ax - x0) // res) - nx // 2
    cx = x0 + res * (c0 + np.arange(nx) + 0.5)
    cy = y0 - res * (r0 + np.arange(ny) + 0.5)
    X, Y = np.meshgrid(cx, cy)
    lon, lat = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True).transform(X, Y)

    def f(x, y):
        return 0.2 + 1e-7 * (x - cx[0]) + 2e-7 * (y - cy[-1])

    h5 = G(
        cell_lat=DS(lat),
        cell_lon=DS(lon),
        Geophysical_Data=G(
            sm_rootzone=DS(f(X, Y).astype("f4"), _FillValue=-9999.0, units="m3 m-3"),
            sm_surface=DS(np.full(X.shape, -9999.0, "f4"), _FillValue=-9999.0),
        ),
    )
    monkeypatch.setattr(C, "earthdata_granules", lambda *a, **k: [None])
    monkeypatch.setattr(C, "earthdata_open", lambda g, **kw: ["file"])
    monkeypatch.setattr(C, "_h5", lambda f: h5)
    out = Out(context.grid)
    C.smap_l4(context, out)
    ds = _dataset(out)
    gx, gy = np.meshgrid(context.grid.x, context.grid.y)
    ex, ey = Transformer.from_crs(context.grid.crs, "EPSG:6933", always_xy=True).transform(gx, gy)
    np.testing.assert_allclose(ds["sm_rootzone"].values, f(ex, ey), atol=1e-4)
    assert ds["sm_rootzone"].attrs["method"] == "bilinear"
    assert np.isnan(ds["sm_surface"].values).all()  # fill → NaN, not −9999


def test_viirs_lst_qc_and_scale(monkeypatch, context):
    meta = b"""GROUP=GRID_1
\t\tGridName="VIIRS_Grid_Daily_1km_LST21"
\t\tXDim=1200
\t\tYDim=1200
\t\tUpperLeftPointMtrs=(-7783653.637667,5559752.598333)
\t\tLowerRightMtrs=(-6671703.118000,4447802.078667)
\tEND_GROUP=GRID_1"""
    lst = np.full((1200, 1200), 15000, np.uint16)  # × 0.02 = 300 K
    qc = np.zeros((1200, 1200), np.uint16)
    qc[:, :40] = 2  # cloud over the west part of the AOI (tile cols 19–87)
    fields = G(LST_1KM=DS(lst, scale_factor=0.02, add_offset=0.0, _FillValue=0), QC=DS(qc))
    h5 = G(
        {
            "HDFEOS INFORMATION": G({"StructMetadata.0": DS(np.bytes_(meta))}),
            "HDFEOS": G(GRIDS=G(VIIRS_Grid_Daily_1km_LST21=G({"Data Fields": fields}))),
        }
    )
    monkeypatch.setattr(C, "earthdata_granules", lambda *a, **k: [None])
    monkeypatch.setattr(C, "earthdata_download", lambda g, context, s=None: ["f.h5"])
    monkeypatch.setattr(C, "_h5", lambda f: h5)
    out = Out(context.grid)
    C.viirs_lst_day(context, out)
    ds = _dataset(out)
    v = ds["LST_1KM"].values
    assert np.nanmax(np.abs(v - 300.0)) < 1e-3  # 15000 × 0.02 K
    from nasa_farm.grid.readers import SINUSOIDAL

    lon, lat = context.grid.lonlat()
    sx, _ = Transformer.from_crs("EPSG:4326", SINUSOIDAL, always_xy=True).transform(lon, lat)
    col = (sx + 7783653.637667) / 926.625433  # tile column of each cell
    assert np.isnan(v[col < 39]).all() and np.isfinite(v[col > 41]).all()  # QC mask
    assert ds["LST_1KM"].attrs["method"] == "average"


def test_ecostress_cloud_mask_and_valid_frac(monkeypatch, context, tmp_path):
    tr, (ny, nx) = _utm_frame(context.grid, 70.0)
    stem = tmp_path / "ECOv002_L2T_LSTE_33000_004_15TVG_20240715T180000_0712_01"
    lst = np.full((ny, nx), 305.0, "f4")
    cloud = np.zeros((ny, nx), "u1")
    cloud[:, : nx // 2] = 1
    files = [
        _tif(f"{stem}_LST.tif", lst, tr, "EPSG:32615", np.nan),
        _tif(f"{stem}_cloud.tif", cloud, tr, "EPSG:32615", 255),
    ]
    monkeypatch.setattr(C, "earthdata_granules", lambda *a, **k: [None])
    from pathlib import Path

    monkeypatch.setattr(
        C, "earthdata_download", lambda g, context, s=None: [Path(f) for f in files]
    )
    out = Out(context.grid)
    C.ecostress_lst(context, out)
    ds = _dataset(out)
    v, w = ds["LST"].values, ds["valid_frac"].values
    assert np.allclose(v[np.isfinite(v)], 305.0)
    assert np.nanmin(w) == 0.0 and np.nanmax(w) == 1.0
    assert np.isnan(v[w == 0]).all()


class _Asset:
    def __init__(self, href):
        self.href, self.extra_fields = href, {}


class _Item:
    def __init__(self, identifier, assets, **props):
        self.id, self.properties = identifier, props
        self.assets = {k: _Asset(v) for k, v in assets.items()}
        self.datetime = dt.datetime(2024, 7, 15, tzinfo=dt.UTC)


def test_sentinel2_offset_scl_mask_and_fractions(monkeypatch, context, tmp_path):
    tr10, s10 = _utm_frame(context.grid, 10.0, pad_km=1)
    tr20, s20 = _utm_frame(context.grid, 20.0, pad_km=1)
    assets = {
        b: _tif(tmp_path / f"{b}.tif", np.full(s10, 2000, "u2"), tr10, "EPSG:32615", 0)
        for b in C.S2_10M
    }
    assets |= {
        b: _tif(tmp_path / f"{b}.tif", np.full(s20, 3000, "u2"), tr20, "EPSG:32615", 0)
        for b in C.S2_20M
    }
    scl = np.full(s20, 4, "u1")
    scl[: s20[0] // 3] = 9  # north third: cloud
    assets["SCL"] = _tif(tmp_path / "SCL.tif", scl, tr20, "EPSG:32615", 0)
    item = _Item("S2B_TEST", assets, **{"s2:processing_baseline": "05.10"})
    monkeypatch.setattr(C, "planetary_items", lambda *a, **k: [item])
    out = Out(context.grid)
    C.s2_l2a(context, out)
    ds = _dataset(out)
    b4 = ds["B04"].values
    assert np.allclose(b4[np.isfinite(b4)], 0.1, atol=1e-6)  # (2000 − 1000) / 1e4
    assert np.allclose(ds["B11"].values[np.isfinite(ds["B11"].values)], 0.2, atol=1e-6)
    assert np.isnan(b4[0]).all() and np.isfinite(b4[-1]).all()
    fr = ds["scl_frac"]
    np.testing.assert_allclose(fr.sum("scl_class").values, 1, atol=1e-5)
    veg = fr.sel(scl_class="4:vegetation").values
    assert veg[-1].min() == pytest.approx(1) and veg[0].max() == pytest.approx(0)


def test_cdl_and_crop_frequency(monkeypatch, context):
    xr = pytest.importorskip("xarray")
    g = context.grid.buffered(3)
    ny, nx = int((g.north - g.south) // 30), int((g.east - g.west) // 30)
    cols = np.arange(nx)[None, :].repeat(ny, 0)
    years = np.arange(2008, 2025)
    # corn/soy alternation: even years corn on the west half, soy east
    a = np.stack([np.where((cols < nx // 2) == (y % 2 == 0), 1, 5) for y in years]).astype("u1")
    store = xr.Dataset(
        {"crop_type": (("year", "y", "x"), a)},
        coords={
            "year": years,  # pixel centres, y down
            "y": g.north - 15.0 - 30.0 * np.arange(ny),
            "x": g.west + 15.0 + 30.0 * np.arange(nx),
        },
    )
    monkeypatch.setattr(C, "_cdl_store", lambda: store)
    out = Out(context.grid)
    C.cdl(context, out)
    ds = _dataset(out)
    corn = ds["cdl_frac"].sel(cdl_class="1:corn").values
    assert set(np.unique(np.round(corn, 3))) <= {0.0, 1.0} or corn.mean() > 0
    np.testing.assert_allclose(ds["cdl_frac"].sum("cdl_class"), 1, atol=1e-5)
    out2 = Out(context.grid)
    C.crop_frequency(context, out2)
    ds2 = _dataset(out2)
    tot = ds2["years_corn"].values + ds2["years_soybeans"].values
    np.testing.assert_allclose(tot, 17, atol=1e-3)  # every year corn or soy
    assert ds2["years_wheat"].values.max() == 0


def test_latlon_granule_netcdf(monkeypatch, context, tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("scipy")
    lat = np.arange(40.125, 44, 0.25)
    lon = np.arange(-96.875, -90, 0.25)
    LON, LAT = np.meshgrid(lon, lat)
    ds = xr.Dataset(
        {"GWS_tavg": (("time", "lat", "lon"), (LON + LAT)[None].astype("f4"), {"units": "mm"})},
        coords={"time": np.array(["2024-07-15"], "datetime64[ns]"), "lat": lat, "lon": lon},
    )
    p = tmp_path / "g.nc"
    ds.to_netcdf(p, engine="scipy")
    monkeypatch.setattr(C, "earthdata_granules", lambda *a, **k: [None])
    monkeypatch.setattr(C, "earthdata_download", lambda g, context, s=None: [p])
    monkeypatch.setattr(C, "_xr", lambda f: xr.open_dataset(f, engine="scipy"))
    out = Out(context.grid)
    C.gldas_da(context, out)
    d = _dataset(out)
    glon, glat = context.grid.lonlat()
    np.testing.assert_allclose(d["GWS_tavg"].values, glon + glat, atol=1e-3)
    assert d["GWS_tavg"].attrs["time"].startswith("2024-07-15")


def test_gedi_binning(monkeypatch, context):
    lon, lat = context.grid.lonlat()
    n = 200
    plon = np.linspace(lon.min(), lon.max(), n)
    plat = np.full(n, lat.mean())
    rh = np.zeros((n, 101), "f4")
    rh[:, 98] = 12.0
    q = np.ones(n, "u1")
    q[::2] = 0  # half the shots bad
    beam = G(
        lat_lowestmode=DS(plat),
        lon_lowestmode=DS(plon),
        rh=DS(rh),
        elev_lowestmode=DS(np.full(n, 300.0)),
        l2a_quality_flag_rel3=DS(q),
        degrade_flag=DS(np.zeros(n, "u1")),
    )
    h5 = G(BEAM0000=beam, METADATA=G())
    monkeypatch.setattr(C, "earthdata_granules", lambda *a, **k: [None])
    monkeypatch.setattr(C, "earthdata_open", lambda g, **kw: ["f"])
    monkeypatch.setattr(C, "_h5", lambda f: h5)
    out = Out(context.grid)
    C.gedi_l2a(context, out)
    ds = _dataset(out)
    assert ds["n_shots"].values.sum() == pytest.approx(n / 2, abs=2)
    v = ds["rh98"].values
    assert np.allclose(v[np.isfinite(v)], 12.0)


def test_usgs_points(monkeypatch, context, tmp_path):
    lon, lat = context.grid.lonlat()
    feats = [
        {
            "properties": {
                "monitoring_location_id": "USGS-05470000",
                "value": "120",
                "time": "2024-07-15",
            },
            "geometry": {"coordinates": [float(lon[10, 10]), float(lat[10, 10])]},
        }
    ]
    monkeypatch.setattr(C, "usgs_items", lambda col, p: feats)
    out = Out(context.grid)
    C.usgs_discharge(context, out)
    assert out.points.loc[0, "row"] == 10 and out.points.loc[0, "col"] == 10
    assert out.vars["n_sites"][1].sum() == 1


def test_gssurgo_valu1_rasters(monkeypatch, context, tmp_path):
    tr, shape = _utm_frame(context.grid, 30.0)
    west = np.arange(shape[1])[None, :] < shape[1] // 2
    a = np.where(west, 0.5, -9999.0).repeat(shape[0], 0).astype("f4")  # east = nodata
    item = _Item(
        "gnatsgo", {c: _tif(tmp_path / f"{c}.tif", a, tr, "EPSG:32615", -9999.0) for c in C.VALU1}
    )
    monkeypatch.setattr(C, "planetary_items", lambda *a, **k: [item])
    out = Out(context.grid)
    C.gssurgo_valu1(context, out)
    assert set(C.VALU1) <= set(out.vars)
    v = out.vars["nccpi3all"][1]
    assert np.allclose(v[np.isfinite(v)], 0.5) and np.isnan(v).any()


def test_gssurgo_valu1_missing_asset(monkeypatch, context, tmp_path):
    tr, shape = _utm_frame(context.grid, 30.0)
    item = _Item(
        "gnatsgo",
        {"mukey": _tif(tmp_path / "mukey.tif", np.ones(shape, "u4"), tr, "EPSG:32615", 0)},
    )
    monkeypatch.setattr(C, "planetary_items", lambda *a, **k: [item])
    with pytest.raises(KeyError, match="Valu1"):
        C.gssurgo_valu1(context, Out(context.grid))


def test_run_one_writes_netcdf_and_stack(monkeypatch, grid, tmp_path):
    pytest.importorskip("scipy")
    from nasa_farm.grid import run as R

    def fake(context, out):
        out.add("a", np.ones(grid.shape))

    p = C.Product("fake", 99, "fake", "none", (), "V", fake)
    rec = R.run_one(p, grid, DATE, tmp_path, tmp_path / "raw", {})
    assert rec["status"] == "ok" and (tmp_path / "fake.nc").exists()
    p2 = C.Product("needy", 99, "x", "none", ("earthdata",), "V", fake)
    assert R.run_one(p2, grid, DATE, tmp_path, tmp_path, {})["status"] == "skipped"
    st = R.stack(tmp_path, [rec])
    import xarray as xr

    with xr.open_dataset(st) as s:
        assert "fake__a" in s
    R.report(tmp_path, [rec], grid, DATE)
    assert "1 ok" in (tmp_path / "report.md").read_text()


def _mrms_field(context, fill):
    """0.01° lat/lon field of 12.5 mm with `fill` (no coverage) on the
    western half of the target grid itself — so the masking is exercised."""
    b, g = context.grid.bbox_ll(30.0), context.grid.bbox_ll()
    res = 0.01
    nx, ny = int((b.east - b.west) / res) + 1, int((b.north - b.south) / res) + 1
    lon = b.west + res * (np.arange(nx) + 0.5)
    a = np.where(lon[None, :] < (g.west + g.east) / 2, fill, 12.5).repeat(ny, 0).astype("f8")
    return a, Affine(res, 0, b.west, 0, -res, b.north)


def _check_mrms(out, context):
    ds = _dataset(out)
    v = ds["qpe_24h"].values
    nx = context.grid.shape[1]
    assert ds["qpe_24h"].attrs["units"] == "mm"
    assert np.allclose(v[:, -nx // 4 :], 12.5, atol=1e-3)  # mm kept as is (east)
    assert np.isnan(v[:, : nx // 4]).all()  # no coverage → NaN (west)
    nxt = DATE + dt.timedelta(days=1)
    assert ds["qpe_24h"].attrs["time"] == f"{DATE:%Y-%m-%d}T00Z/{nxt:%Y-%m-%d}T00Z"


def test_mrms_qpe_24h_planetary(monkeypatch, context, tmp_path):
    a, tr = _mrms_field(context, -1.0)  # PC COG nodata
    item = _Item(
        "CONUS_MRMS_..._20240716-000000",
        {"cog": _tif(tmp_path / "q.tif", a, tr, "EPSG:4326", -1.0)},
    )
    calls = []
    monkeypatch.setattr(C, "planetary_items", lambda *a, **k: calls.append((a, k)) or [item])
    out = Out(context.grid)
    C.mrms_qpe_24h(context, out)
    ((args, kw),) = calls
    assert args[0] == "noaa-mrms-qpe-24h-pass2" and args[2] is None
    assert kw["datetime"] == "2024-07-16T00:00:00Z"  # end of the 24 h window
    assert kw["query"] == {"noaa_mrms_qpe:region": {"eq": "CONUS"}}
    _check_mrms(out, context)
    monkeypatch.setenv("MRMS_DAY_START_UTC", "12")
    C.mrms_qpe_24h(context, Out(context.grid))
    assert calls[-1][1]["datetime"] == "2024-07-16T12:00:00Z"  # 12Z → 12Z window


def test_mrms_qpe_24h_noaa_fallback(monkeypatch, context, tmp_path):
    import gzip
    import shutil

    from rasterio.shutil import copy as rio_copy

    a, tr = _mrms_field(context, -3.0)  # GRIB "no coverage"
    _tif(tmp_path / "q.tif", a, tr, "EPSG:4326")
    rio_copy(str(tmp_path / "q.tif"), str(tmp_path / "q.grib2"), driver="GRIB")
    gz = tmp_path / "q.grib2.gz"
    with (tmp_path / "q.grib2").open("rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    seen = []

    def no_pc(*a, **k):
        raise FileNotFoundError("PC: no item")  # date before the PC archive

    def fake_download(url, dest, **kw):
        seen.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(gz, dest)
        return dest

    monkeypatch.setattr(C, "planetary_items", no_pc)
    monkeypatch.setattr(C, "download", fake_download)
    out = Out(context.grid)
    C.mrms_qpe_24h(context, out)
    nxt = (DATE + dt.timedelta(days=1)).strftime("%Y%m%d")
    assert seen[0].endswith(f"/{nxt}/MRMS_MultiSensor_QPE_24H_Pass2_00.00_{nxt}-000000.grib2.gz")
    _check_mrms(out, context)
