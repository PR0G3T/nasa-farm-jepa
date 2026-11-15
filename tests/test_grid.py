"""Common 1 km grid — offline, synthetic-data tests of the regridding rules.

Each test builds a source raster whose correct 1 km answer is known
analytically, so the assertions check the numbers, not just shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rasterio")
from affine import Affine
from pyproj import Transformer

from nasa_farm.grid.regrid import (
    Raster,
    bin_points,
    method_for,
    native_res_m,
    rasterize_cover,
    rasterize_values,
    regrid_categorical,
    regrid_continuous,
)
from nasa_farm.grid.target import TargetGrid


@pytest.fixture
def grid() -> TargetGrid:
    return TargetGrid.named("ames")


def test_grid_is_snapped_and_covers_aoi(grid):
    assert grid.shape == (30, 30) or grid.shape == (31, 31)
    for v in (grid.west, grid.south, grid.east, grid.north):
        assert v % 1000 == 0
    b = grid.bbox_ll()
    assert b.west < -93.6 < b.east and b.south < 42.0 < b.north
    lon, lat = grid.lonlat()
    assert lon.shape == grid.shape and abs(lat.mean() - 42.0) < 0.02


def test_method_rules():
    assert method_for("continuous", 30, 1000) == "average"
    assert method_for("continuous", 9000, 1000) == "bilinear"
    assert method_for("categorical", 30, 1000) == "fractions"
    assert method_for("categorical", 9000, 1000) == "nearest"


def _utm_raster(grid: TargetGrid, res: float, fn, pad_km: float = 3.0, **kw) -> Raster:
    """Source on UTM 15N at `res` m, values fn(x5070, y5070) at pixel
    centres — a different CRS than the target on purpose."""
    t = Transformer.from_crs(grid.crs, "EPSG:32615", always_xy=True)
    xs, ys = t.transform(
        [grid.west, grid.east, grid.west, grid.east],
        [grid.south, grid.south, grid.north, grid.north],
    )
    pad = pad_km * 1000
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    nx, ny = int((x1 - x0) // res), int((y1 - y0) // res)
    tr = Affine(res, 0, x0, 0, -res, y1)
    cx = x0 + res * (np.arange(nx) + 0.5)
    cy = y1 - res * (np.arange(ny) + 0.5)
    X, Y = np.meshgrid(cx, cy)
    bx, by = Transformer.from_crs("EPSG:32615", grid.crs, always_xy=True).transform(X, Y)
    return Raster(fn(np.asarray(bx), np.asarray(by)), tr, "EPSG:32615", ["v"], **kw)


def test_fine_continuous_average_preserves_linear_field(grid):
    # A linear field's area mean over a cell equals its value at the centre.
    def f(x, y):
        return (1e-3 * (x - grid.west) + 2e-3 * (y - grid.south)).astype(np.float32)

    r = _utm_raster(grid, 30.0, f)
    out, valid, how = regrid_continuous(r, grid)
    assert how == "average"
    X, Y = np.meshgrid(grid.x, grid.y)
    np.testing.assert_allclose(out[0], f(X, Y), atol=0.05)  # ~1.5 m of 1 km
    np.testing.assert_allclose(valid, 1.0, atol=1e-6)


def test_nan_pixels_excluded_and_valid_frac(grid):
    r = _utm_raster(grid, 30.0, lambda x, y: np.full(x.shape, 5.0, np.float32))
    r.data[0, :, : r.data.shape[2] // 2] = np.nan  # west half "cloudy"
    r.data[0, :, r.data.shape[2] // 2 :] = 7.0
    out, valid, _ = regrid_continuous(r, grid)
    ok = np.isfinite(out[0])
    assert np.allclose(out[0][ok], 7.0)  # clouds never mixed in
    assert valid.min() == 0.0 and valid.max() == 1.0
    assert np.isnan(out[0][valid == 0]).all()


def test_coarse_bilinear_interpolates_linear_field(grid):
    # 0.125° lat/lon source (NLDAS-2-like), linear in lon/lat → bilinear exact
    def f(lon, lat):
        return (10 * lon + 3 * lat).astype(np.float32)

    lon0, lat1, res = -94.5, 43.0, 0.125
    lons = lon0 + res * (np.arange(16) + 0.5)
    lats = lat1 - res * (np.arange(16) + 0.5)
    LON, LAT = np.meshgrid(lons, lats)
    r = Raster(f(LON, LAT), Affine(res, 0, lon0, 0, -res, lat1), "EPSG:4326", ["v"])
    assert 10_000 < native_res_m(r) < 13_000
    out, valid, how = regrid_continuous(r, grid)
    assert how == "bilinear" and valid is None
    lon, lat = grid.lonlat()
    np.testing.assert_allclose(out[0], f(lon, lat), atol=1e-3)


def test_categorical_fractions_sum_to_one_and_mode(grid):
    # vertical stripes: 70 % class 1, 30 % class 5 in every 1 km column band
    def f(x, y):
        u = ((x - grid.west) % 1000) / 1000
        return np.where(u < 0.7, 1, 5).astype(np.uint8)

    r = _utm_raster(grid, 30.0, f, kind="categorical", nodata=0, classes=[1, 5, 24])
    fr, mode, classes, how = regrid_categorical(r, grid)
    assert how == "fractions" and classes == [1, 5, 24]
    assert fr.shape == (4, *grid.shape)  # 3 classes + "other"
    np.testing.assert_allclose(fr.sum(0), 1.0, atol=1e-5)
    assert abs(np.nanmean(fr[0]) - 0.7) < 0.03
    assert abs(np.nanmean(fr[1]) - 0.3) < 0.03
    assert np.nanmax(fr[2]) == 0 and np.nanmax(fr[3]) == 0
    assert (mode == 1).mean() > 0.9


def test_bin_points_mean_and_count(grid):
    lon, lat = grid.lonlat()
    i, j = 4, 7
    pl = np.full(5, lon[i, j]) + np.linspace(-1e-4, 1e-4, 5)
    pa = np.full(5, lat[i, j])
    out = bin_points(pl, pa, {"h": np.array([1, 2, 3, 4, np.nan])}, grid)
    assert out["count"][i, j] == 5 and out["count"].sum() == 5
    assert out["h"][i, j] == pytest.approx(2.5)
    assert np.isnan(out["h"][0, 0])


def test_rasterize_cover_and_values(grid):
    pytest.importorskip("shapely")
    from shapely.geometry import box

    w, n, r = grid.west, grid.north, grid.res
    half = box(w, n - r, w + r / 2, n)  # half of cell (0, 0)
    full = box(w + r, n - r, w + 2 * r, n)  # all of cell (0, 1)
    cov = rasterize_cover([half, full], grid)
    assert cov[0, 0] == pytest.approx(0.5) and cov[0, 1] == pytest.approx(1)
    val = rasterize_values([half, full], [10.0, 20.0], grid)
    assert val[0, 0] == pytest.approx(10) and val[0, 1] == pytest.approx(20)
    assert np.isnan(val[5, 5])


# --- readers ------------------------------------------------------------------
def test_ease2_transform_recovers_9km_grid():
    from nasa_farm.grid.readers import ease2_transform

    # NSIDC EASE-2 global 9 km constants (cell edges)
    res, x0, y0 = 9008.055210146, -17367530.44516137, 7314540.830638504
    r0, c0, ny, nx = 500, 900, 40, 60  # a window over the US
    cx = x0 + res * (c0 + np.arange(nx) + 0.5)
    cy = y0 - res * (r0 + np.arange(ny) + 0.5)
    X, Y = np.meshgrid(cx, cy)
    lon, lat = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True).transform(X, Y)
    lat = np.asarray(lat).copy()
    lat[:5, :5] = -9999.0  # L3-style fill values
    tr, crs = ease2_transform(lat, np.asarray(lon))
    assert crs == "EPSG:6933"
    assert tr.a == pytest.approx(res, rel=1e-6)
    assert tr.c == pytest.approx(x0 + c0 * res, abs=1.0)
    assert tr.f == pytest.approx(y0 - r0 * res, abs=1.0)


def test_read_ease2_merges_per_pass_coordinates(grid):
    """SPL3FTP v004: lat/lon are (am_pm, y, x) with different fill cells per
    pass — the reader must merge them (regression: 'too many values to unpack')."""
    h5py = pytest.importorskip("h5py")
    from nasa_farm.grid.readers import read_ease2

    res, x0, y0 = 36032.220840584, -17367530.44516137, 7314540.830638504
    r0, c0, ny, nx = 55, 215, 20, 30  # window over Ames (row 67, col 231)
    cx = x0 + res * (c0 + np.arange(nx) + 0.5)
    cy = y0 - res * (r0 + np.arange(ny) + 0.5)
    X, Y = np.meshgrid(cx, cy)
    lon, lat = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True).transform(X, Y)
    la = np.stack([lat, lat]).astype(np.float32)
    lo = np.stack([lon, lon]).astype(np.float32)
    la[0, :, :3] = lo[0, :, :3] = -9999.0  # AM-only fill, PM valid
    la[1, :, -3:] = lo[1, :, -3:] = -9999.0  # PM-only fill, AM valid
    ft = np.stack([np.zeros((ny, nx)), np.ones((ny, nx))]).astype(np.uint8)
    import io

    with h5py.File(io.BytesIO(), "w") as h5:
        h5["latitude"], h5["longitude"], h5["freeze_thaw"] = la, lo, ft
        r = read_ease2(
            h5,
            grid,
            {"ft_am": "freeze_thaw", "ft_pm": "freeze_thaw"},
            "latitude",
            "longitude",
            select={"ft_am": 0, "ft_pm": 1},
        )
    assert r.bands == ["ft_am", "ft_pm"]
    assert r.crs == "EPSG:6933"
    assert np.all(r.data[0] == 0) and np.all(r.data[1] == 1)


def test_struct_metadata_sinusoidal_transform():
    from nasa_farm.grid.readers import parse_struct_metadata, sinusoidal_transform

    txt = """GROUP=GridStructure
	GROUP=GRID_1
		GridName="VIIRS_Grid_Daily_1km_LST21"
		XDim=1200
		YDim=1200
		UpperLeftPointMtrs=(-7783653.637667,5559752.598333)
		LowerRightMtrs=(-6671703.118000,4447802.078667)
		Projection=HE5_GCTP_SNSOID
	END_GROUP=GRID_1
END_GROUP=GridStructure"""
    m = parse_struct_metadata(txt)["VIIRS_Grid_Daily_1km_LST21"]
    tr = sinusoidal_transform(m)
    assert tr.a == pytest.approx(926.625433, rel=1e-6)  # 1 km MODIS/VIIRS
    assert tr.c == pytest.approx(-7783653.637667)  # tile h11v04


def test_read_latlon_subsets_flips_and_splits_time(grid):
    xr = pytest.importorskip("xarray")
    lat = np.arange(25.0625, 53, 0.125)  # ascending (NLDAS-2)
    lon = np.arange(-124.9375, -67, 0.125)
    t = np.array(["2024-07-15T12", "2024-07-15T13"], dtype="datetime64[ns]")
    LON, LAT = np.meshgrid(lon, lat)
    v = np.stack([10 * LON + 3 * LAT + k for k in range(2)]).astype("f4")
    ds = xr.Dataset(
        {"Tair": (("time", "lat", "lon"), v, {"units": "K"})},
        coords={"time": t, "lat": lat, "lon": lon},
    )
    from nasa_farm.grid.readers import read_latlon
    from nasa_farm.grid.regrid import regrid_continuous

    r = read_latlon(ds, grid)
    assert r.bands == ["Tair", "Tair"] and r.times[0].startswith("2024-07-15T12")
    assert r.data.shape[1] < 30  # subset, not CONUS
    out, _, _how = regrid_continuous(r, grid)
    glon, glat = grid.lonlat()
    np.testing.assert_allclose(out[1], 10 * glon + 3 * glat + 1, atol=2e-3)


def test_read_tif_window_scale_and_nodata(grid, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from nasa_farm.grid.readers import read_tif

    r = _utm_raster(grid, 30.0, lambda x, y: np.full(x.shape, 5000, np.int16), pad_km=20)
    a = r.data[0].copy()
    a[:100, :] = -9999
    p = tmp_path / "t.tif"
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=a.shape[0],
        width=a.shape[1],
        count=1,
        dtype="int16",
        crs="EPSG:32615",
        transform=r.transform,
        nodata=-9999,
    ) as ds:
        ds.write(a, 1)
    rr = read_tif(str(p), grid, name="ndvi", scale=1e-4)
    assert rr.data.shape[1] < a.shape[0]  # windowed read
    vals = rr.data[0][np.isfinite(rr.data[0])]
    assert np.allclose(vals, 0.5)


def test_retry_transient_retries_only_server_errors():
    aiohttp = pytest.importorskip("aiohttp")
    from nasa_farm.grid.fetch import is_transient, retry_transient

    def http(status):
        return aiohttp.ClientResponseError(None, (), status=status, message="x")

    assert is_transient(http(502)) and is_transient(http(503)) and is_transient(http(429))
    assert not is_transient(http(404)) and not is_transient(KeyError("lat"))
    calls = []

    def flaky():  # fails twice with a 502, then succeeds
        calls.append(1)
        if len(calls) < 3:
            raise http(502)
        return "ok"

    assert retry_transient(flaky, attempts=4, wait=0) == "ok" and len(calls) == 3
    calls.clear()

    def missing():
        calls.append(1)
        raise KeyError("lat_lowestmode")

    with pytest.raises(KeyError):  # a real error is not retried
        retry_transient(missing, attempts=4, wait=0)
    assert len(calls) == 1
    with pytest.raises(aiohttp.ClientResponseError):  # gives up after `attempts`
        retry_transient(lambda: (_ for _ in ()).throw(http(502)), attempts=2, wait=0)


class _FakeRangeServer:
    """requests-like session serving `blob` by Range, with scripted faults."""

    def __init__(self, blob: bytes, faults=(), ranges=True):
        self.blob, self.faults, self.ranges, self.gets = blob, list(faults), ranges, []

    def head(self, url, **kw):
        return _Resp(
            200,
            b"",
            {
                "Content-Length": str(len(self.blob)),
                "Accept-Ranges": "bytes" if self.ranges else "none",
            },
        )

    def get(self, url, headers=None, **kw):
        lo, hi = (int(v) for v in headers["Range"].split("=")[1].split("-"))
        self.gets.append(lo)
        fault = self.faults.pop(0) if self.faults else None
        if fault == "503":
            return _Resp(503, b"")
        body = self.blob[lo : hi + 1]
        if fault == "short":
            body = body[:-1]
        if fault == "ignore":
            return _Resp(200, self.blob)
        return _Resp(206, body)


class _Resp:
    def __init__(self, status, content, headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}", response=self)


def test_download_ranged_retries_and_resumes(tmp_path, monkeypatch):
    pytest.importorskip("requests")
    from nasa_farm.grid import fetch

    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    blob = bytes(range(256)) * 40  # 10 240 B → 3 chunks of 4 kB
    srv = _FakeRangeServer(blob, faults=[None, "503", "short"])
    p = fetch.download_ranged("https://x/f.zip", tmp_path / "f.zip", sess=srv, chunk=4096)
    assert p.read_bytes() == blob and not (tmp_path / "f.zip.part").exists()
    # resume: an existing .part is continued, not restarted
    (tmp_path / "g.zip.part").write_bytes(blob[:4096])
    srv = _FakeRangeServer(blob)
    fetch.download_ranged("https://x/g.zip", tmp_path / "g.zip", sess=srv, chunk=4096)
    assert (tmp_path / "g.zip").read_bytes() == blob and srv.gets[0] == 4096
    # a server that ignores Range is an error, not a silently wrong file
    with pytest.raises(RuntimeError, match="ignored the Range"):
        fetch.download_ranged(
            "https://x/h.zip",
            tmp_path / "h.zip",
            sess=_FakeRangeServer(blob, faults=["ignore"]),
            chunk=4096,
        )
