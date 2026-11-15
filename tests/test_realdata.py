"""Real-data remediation — product adapters, multi-channel lake loss,
COG windowing, mascon decoding, stream masks, CV splits."""

import datetime as dt
import io

import httpx2 as httpx
import numpy as np
import pytest
import torch

from nasa_farm.acquire import cog
from nasa_farm.acquire.quality import CLOUD_QC, FMASK_QC
from nasa_farm.acquire.products import SPECS, Specification
from nasa_farm.constants import DTYPE
from nasa_farm.lake import Lake


def _power_payload(year: int) -> dict:
    nd = (dt.date(year, 12, 31) - dt.date(year, 1, 1)).days + 1
    days = [(dt.date(year, 1, 1) + dt.timedelta(d)).strftime("%Y%m%d") for d in range(nd)]
    return {
        "properties": {
            "parameter": {
                p: dict.fromkeys(days, v)
                for p, v in [
                    ("T2M_MIN", 5.0), ("T2M_MAX", 20.0), ("PRECTOTCORR", 1.0),
                    ("ALLSKY_SFC_SW_DWN", 15.0), ("RH2M", 70.0), ("WS2M", 3.0),
                    ("GWETROOT", 0.6),
                ]
            }
        }
    }


def _lake(tmp_path, fid="F1", county="Story", kind="maize", years=(2023, 2024)):
    """Synthetic lake: POWER forcing + obs streams on some days."""
    from nasa_farm.acquire.crawl import crawl
    from nasa_farm.acquire.power import Power

    def handler(request):
        return httpx.Response(200, json=_power_payload(int(request.url.params["start"][:4])))

    client = Power(httpx.Client(transport=httpx.MockTransport(handler)))
    crawl(
        [{"id": fid, "lat": 42.0, "lon": -93.6, "county": county, "kind": kind}],
        tmp_path, list(years), sources=("power",), power=client, log=lambda m: None,
    )
    with Lake(tmp_path) as lake:
        rng = np.random.default_rng(0)
        for year in years:
            base = dt.date(year, 1, 1)
            dates = np.sort(
                np.array(
                    [int((base + dt.timedelta(int(o))).strftime("%Y%m%d"))
                     for o in np.arange(0, 340, 10)],
                    np.int32,
                )
            )
            for var, scale in (("sm", 0.3), ("ndvi", 0.7), ("tws", 900.0), ("et", 4.0)):
                lake.put(fid, "obs", var, dates, rng.normal(scale, 0.02, len(dates)).astype(np.float32))
                lake.put(fid, "obs", f"{var}_w", dates, np.ones(len(dates), np.float32))
        mdates = np.array([y * 10000 + m * 100 + 15 for y in years for m in range(1, 13)], np.int32)
        lake.put(fid, "obs", "mascon", mdates, rng.normal(0, 30, len(mdates)).astype(np.float32))
    return tmp_path


class TestCog:
    def test_band_of(self):
        assert cog.band_of("HLS.L30.T15T.2024154T165847.v2.0.NDVI.tif") == "NDVI"
        assert cog.band_of("ECO_L2T_LSTE_001_LST.tif") == "LST"

    def test_pick(self):
        mapping = {"NDVI": "a", "Fmask": "b"}
        assert cog.pick(mapping, ("NDVI",)) == "a"
        assert cog.pick(mapping, ("missing", "Fmask")) == "b"
        assert cog.pick(mapping, ("nope",)) is None

    def test_window_utm(self):
        """In-memory UTM raster → windowed read + reprojected grid."""
        import rasterio
        from rasterio.transform import from_origin

        rasterio = pytest.importorskip("rasterio")
        data = np.arange(100 * 100, dtype=np.float32).reshape(100, 100) * 100
        data[0, 0] = -9999
        transform = from_origin(500000, 4700000, 30, 30)  # UTM 15N
        memory = rasterio.MemoryFile()
        with memory.open(
            driver="GTiff", width=100, height=100, count=1, dtype="float32",
            crs="EPSG:32615", transform=transform, nodata=-9999,
        ) as dst:
            dst.scales = (1e-4,)
            dst.write(data, 1)
        handle = memory.open()
        from pyproj import Transformer

        wgs = Transformer.from_crs("EPSG:32615", 4326, always_xy=True)
        lon0, lat0 = wgs.transform(500100, 4699100)
        lon1, lat1 = wgs.transform(502900, 4700900)
        grid = cog.window(handle, (lon0, lat0, lon1, lat1))
        assert grid is not None
        values, glat, glon = grid
        assert values.shape == glat.shape == glon.shape
        assert np.isnan(values.ravel()[0]) or np.isfinite(values).any()
        assert abs(np.nanmean(glat) - (lat0 + lat1) / 2) < 0.02
        assert abs(np.nanmean(glon) - (lon0 + lon1) / 2) < 0.02
        assert cog.window(handle, (0, 0, 1, 1)) is None  # off-tile → skip

    def test_month_of(self):
        class G:
            id = "HLS.L30.T15T.2024154T165847.v2.0"

        assert cog.month_of(G()) == 6


class TestQuality:
    def test_fmask(self):
        conf = FMASK_QC.to_conf(np.array([[0, 2, 8, 4], [16, 1, 32, 0]]))
        assert conf[0, 0] == 1.0 and conf[0, 1] == 0.0  # clear / cloud
        assert conf[0, 2] == 0.4  # shadow
        assert conf[0, 3] == 0.4  # adjacent cloud
        assert conf[1, 0] == 0.4  # snow/ice bit
        assert conf[1, 2] == 1.0  # water/aerosol bits don't block

    def test_cloud(self):
        conf = CLOUD_QC.to_conf(np.array([0, 1, 2]))
        assert (conf == [1.0, 0.0, 0.0]).all()


class TestSpecs:
    def test_registry(self):
        for name in ("smap", "hls-l30", "hls-s30", "ecostress-et", "ecostress-lst",
                     "gpm", "gldas-tws", "gldas-smp"):
            assert name in SPECS
        assert SPECS["gpm"].group == "forcing"
        assert SPECS["hls-l30"].engine == "cog"
        assert SPECS["gldas-tws"].shared and SPECS["gldas-smp"].shared
        assert SPECS["gpm"].scale == 24.0
        assert SPECS["hls-l30"].scale == 1.0  # file carries its own 1e-4 tag
        assert SPECS["ecostress-lst"].offset == -273.15


class TestMascon:
    def test_dates_datetime64(self):
        import xarray as xarray

        ds = xarray.Dataset(coords={"time": np.array(["2002-04-15", "2002-05-15"], dtype="datetime64[D]")})
        from nasa_farm.acquire.mascon import _dates

        assert (_dates(ds) == [20020415, 20020515]).all()

    def test_dates_days_since_2002(self):
        import xarray as xarray

        ds = xarray.Dataset(coords={"time": np.array([0.0, 31.0])})
        from nasa_farm.acquire.mascon import _dates

        assert (_dates(ds) == [20020115, 20020215]).all()


class TestNass:
    def test_yields_filter(self, monkeypatch):
        from nasa_farm.acquire import nass

        class Client:
            def get(self, url, params=None, timeout=None):
                class R:
                    def json(self):
                        return {"data": [
                            {"year": "2023", "Value": "210.0", "unit_desc": "BU / ACRE"},
                            {"year": "2023", "Value": "12.0", "unit_desc": "TONS / ACRE"},
                        ]}
                return R()

        monkeypatch.setattr(nass.httpx, "get", Client().get)
        out = nass.yields("Story", "IA", "maize", (2020, 2024), key="k")
        assert out == {2023: pytest.approx(210.0 * 62.7725)}

    def test_no_key(self, monkeypatch):
        from nasa_farm.acquire import nass

        monkeypatch.delenv("QUICKSTATS_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            nass.yields("Story", "IA", "maize", key=None)


class TestLakeTrain:
    def test_multichannel_and_holdout(self, tmp_path):
        from nasa_farm.optimize.lake import train

        _lake(tmp_path)
        result = train(
            tmp_path, ["F1"], [2023, 2024], channels=("sm", "ndvi"),
            holdout_years=(2024,), epochs=2, workers=1,
        )
        assert result["runs"] == 1 and len(result["held"]) == 1
        assert "F1/2024" in result["held"]


class TestStreams:
    def test_frames_and_fill(self):
        from nasa_farm.experiments.realdata import _fill, _frames, _stream

        dates = np.array([20230105, 20230120, 20230315], np.int32)
        values = np.array([0.1, 0.2, 0.9])
        filled, conf = _frames(dates, values, None, 2023, 74, 5)
        assert np.isfinite(filled).sum() == 3
        assert conf.sum() == 3.0
        column = np.array([np.nan, 0.5, np.nan, np.nan, 0.9, np.nan])
        out = _fill(column)
        assert np.allclose(out, [0.5, 0.5, 0.5, 0.5, 0.9, 0.9])
        obs = {"ndvi": (dates, values, None)}
        targets, mask = _stream(obs, ("ndvi", "sm"), 2023, 5)
        assert targets.shape == (1, 74, 2)
        assert mask.sum() == 3
        assert np.isfinite(targets.numpy()[:, :, 0]).all()  # filled
        assert not np.isfinite(targets.numpy()[:, :, 1]).any()  # empty channel

    def test_loco_split(self, tmp_path):
        _lake(tmp_path, "A", "Story")
        with Lake(tmp_path) as lake:
            lake.field("B", 41.0, -95.0, "Ford 1")
            lake.attrs("B", county="Ford")
        from nasa_farm.experiments import run
        from nasa_farm.experiments.specification import Specification

        spec = Specification(
            lake=str(tmp_path), years=(2023, 2024), epochs=1, observed_channels=("sm",)
        )
        rep = run("loco", spec)
        assert "Story" in rep.results["counties"]
        assert rep.results["counties"]["Story"]["held_fields"] == 1

    def test_stream_masked_jepa(self, tmp_path):
        _lake(tmp_path)
        from nasa_farm.experiments import run
        from nasa_farm.experiments.specification import Specification

        spec = Specification(
            lake=str(tmp_path), years=(2023, 2024), epochs=2,
            latent_dimension=4, members=1, cadence=10,
        )
        rep = run("stream", spec)
        assert rep.results["fields_fit"] >= 0
        assert rep.results["channels"] == ["ndvi", "sm1", "et", "lst"]

    def test_tws_grid(self, tmp_path):
        _lake(tmp_path)
        from nasa_farm.experiments import run
        from nasa_farm.experiments.specification import Specification

        spec = Specification(lake=str(tmp_path), years=(2023, 2024))
        rep = run("tws", spec)
        assert rep.results["k_tws"] in (5e-4, 1e-3, 2e-3, 5e-3, 1e-2)
        assert rep.results["fields"] == 1

    def test_imerg(self, tmp_path):
        _lake(tmp_path)
        with Lake(tmp_path) as lake:
            rng = np.random.default_rng(1)
            dates = np.array(
                [2023 * 10000 + m * 100 + d for m in range(1, 13) for d in range(1, 29)],
                np.int32,
            )
            lake.put("F1", "forcing", "precip_gpm", dates,
                     rng.gamma(0.5, 2.0, len(dates)).astype(np.float32))
        from nasa_farm.experiments import run
        from nasa_farm.experiments.specification import Specification

        spec = Specification(lake=str(tmp_path), years=(2023,))
        rep = run("imerg", spec)
        assert rep.results["n"] >= 0


class TestModel:
    def test_k_tws_buffer(self, crops):
        from nasa_farm.dynamics.model import K_TWS, WorldModel

        model = WorldModel(crops, residual=False)
        assert torch.isclose(model.k_tws, torch.tensor(K_TWS, dtype=DTYPE))
        assert "k_tws" in model.state_dict()
        with torch.no_grad():
            model.k_tws.fill_(1e-3)
        assert float(model.k_tws) == pytest.approx(1e-3)
