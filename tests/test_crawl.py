"""Crawler — point sources land in the lake; granule path is resumable."""

import datetime as dt

import httpx2 as httpx
import numpy as np
import pytest
import torch
import xarray as xr

from nasa_farm.acquire.crawl import (
    _day_keys, _granule_date, crawl, sample,
)
from nasa_farm.acquire.power import PowerClient, series_to_forcing
from nasa_farm.lake import get
from nasa_farm.optimize.data import forcings, targets


def _payload(year: int) -> dict:
    nd = (dt.date(year, 12, 31) - dt.date(year, 1, 1)).days + 1
    days = [(dt.date(year, 1, 1) + dt.timedelta(d)).strftime("%Y%m%d")
            for d in range(nd)]
    return {"properties": {"parameter": {
        p: {d: v for d in days} for p, v in [
            ("T2M_MIN", 5.0), ("T2M_MAX", 20.0), ("PRECTOTCORR", 1.0),
            ("ALLSKY_SFC_SW_DWN", 15.0), ("RH2M", 70.0), ("WS2M", 3.0),
            ("GWETROOT", 0.6)]}}}


class TestHelpers:
    def test_day_keys_drops_leap(self):
        keys = _day_keys([2024, 2025])
        assert len(keys[2024]) == 365 and "20240229" not in keys[2024]
        assert len(keys[2025]) == 365

    def test_granule_date(self):
        assert _granule_date("SPL3SMP_E.007_20230115T120000.h5") == 20230115
        assert _granule_date("ECO_L3G_SEB_doy2023180") == 20230629

    def test_sample_1d(self):
        ds = xr.Dataset(
            {"v": (("lat", "lon"), np.arange(6.0).reshape(2, 3))},
            coords={"lat": [48.0, 49.0], "lon": [1.0, 2.0, 3.0]})
        assert sample(ds, "v", 48.1, 2.1) == pytest.approx(1.0)

    def test_sample_2d(self):
        lat = np.array([[48.0, 48.0], [49.0, 49.0]])
        lon = np.array([[1.0, 2.0], [1.0, 2.0]])
        ds = xr.Dataset(
            {"v": (("y", "x"), np.array([[0.1, 0.2], [0.3, 0.4]])),
             "lat": (("y", "x"), lat), "lon": (("y", "x"), lon)})
        assert sample(ds, "v", 49.01, 2.01) == pytest.approx(0.4)


class TestCrawl:
    def test_power_to_lake(self, tmp_path):
        def handler(request):
            year = int(request.url.params["start"][:4])
            return httpx.Response(200, json=_payload(year))
        client = PowerClient(httpx.Client(transport=httpx.MockTransport(handler)))
        f = {"id": "F1", "lat": 48.0, "lon": 1.5}
        crawl([f], tmp_path, [2024, 2025], sources=("power",),
              power=client, log=lambda m: None)
        d, v = get(tmp_path, "F1", "obs", "sm")
        assert len(d) == 730 and np.allclose(v, 0.6)
        d, v = get(tmp_path, "F1", "forcing", "PRECTOTCORR")
        assert len(d) == 730 and "20240229" not in d.astype(str)

    def test_lake_forcing_matches_live(self, tmp_path):
        """Crawled forcing rebuilds to the same Forcing as the live path."""
        year = 2025
        def handler(request):
            return httpx.Response(200, json=_payload(year))
        client = PowerClient(httpx.Client(transport=httpx.MockTransport(handler)))
        f = {"id": "F1", "lat": 48.0, "lon": 1.5}
        crawl([f], tmp_path, [year], sources=("power",), power=client,
              log=lambda m: None)
        (lake_f,) = forcings(tmp_path, "F1", [year])
        raw = client.data_year(48.0, 1.5, year)
        live_f = series_to_forcing(raw, year)
        assert torch.allclose(lake_f.tmin, live_f.tmin)
        assert torch.allclose(lake_f.precip, live_f.precip)

    def test_targets(self, tmp_path):
        d = np.array([20240101], np.int32)
        from nasa_farm.lake import put
        put(tmp_path, "F1", "obs", "et", d, np.array([2.5], np.float32))
        t = targets(tmp_path, "F1")
        assert "et" in t and t["et"][1][0] == pytest.approx(2.5)
