"""POWER client — request building and payload parsing (mocked transport)."""

import datetime as dt

import httpx2 as httpx
import pytest

from nasa_farm.acquire.base import Temporal
from nasa_farm.acquire.power import (
    Power,
    parse_daily,
    request_params,
    series_to_forcing,
    series_to_wet,
)


def _payload(year: int = 2020, ndays: int = 3) -> dict:
    days = [(dt.date(year, 1, 1) + dt.timedelta(d)).strftime("%Y%m%d") for d in range(ndays)]
    return {
        "properties": {
            "parameter": {
                "T2M_MIN": dict.fromkeys(days, 5.0),
                "T2M_MAX": dict.fromkeys(days, 20.0),
                "PRECTOTCORR": dict.fromkeys(days, 2.0),
                "ALLSKY_SFC_SW_DWN": dict.fromkeys(days, 15.0),
                "RH2M": dict.fromkeys(days, 70.0),
                "WS2M": dict.fromkeys(days, 3.0),
                "GWETROOT": dict.fromkeys(days, 0.6),
            }
        },
        "header": {"title": "POWER"},
    }


class TestRequest:
    def test_params(self):
        p = request_params(42.03, -93.63, Temporal("2020-01-01", "2020-12-31"))
        assert p["community"] == "AG"
        assert p["start"] == "20200101" and p["end"] == "20201231"
        assert p["latitude"] == 42.03
        assert "PRECTOTCORR" in p["parameters"]


class TestParse:
    def test_fill_values_become_nan(self):
        payload = _payload()
        payload["properties"]["parameter"]["T2M_MIN"]["20200102"] = -999.0
        output = parse_daily(payload)
        assert output["T2M_MIN"]["20200101"] == 5.0
        assert output["T2M_MIN"]["20200102"] != output["T2M_MIN"]["20200102"]  # NaN

    def test_series_to_forcing_shape(self):
        year = 2021
        days = [(dt.date(year, 1, 1) + dt.timedelta(d)).strftime("%Y%m%d") for d in range(365)]
        data = {
            p: dict.fromkeys(days, v)
            for p, v in [
                ("T2M_MIN", 5.0),
                ("T2M_MAX", 20.0),
                ("PRECTOTCORR", 1.0),
                ("ALLSKY_SFC_SW_DWN", 15.0),
                ("RH2M", 70.0),
                ("WS2M", 3.0),
            ]
        }
        f = series_to_forcing(data, year)
        assert f.tmin.shape == (1, 365)
        assert float(f.tmin.mean()) == pytest.approx(5.0)
        assert float(f.u2.mean()) == pytest.approx(3.0)

    def test_series_to_wet(self):
        year = 2021
        days = [(dt.date(year, 1, 1) + dt.timedelta(d)).strftime("%Y%m%d") for d in range(365)]
        wet = series_to_wet({"GWETROOT": dict.fromkeys(days, 0.6)}, year)
        assert wet.shape == (365,)
        assert float(wet.mean()) == pytest.approx(0.6)
        assert float(series_to_wet({}, year).mean()) == pytest.approx(0.5)

    def test_leap_year_drops_feb29(self):
        year = 2024  # leap year
        days = [(dt.date(year, 1, 1) + dt.timedelta(d)).strftime("%Y%m%d") for d in range(366)]
        data = {
            p: dict.fromkeys(days, v)
            for p, v in [
                ("T2M_MIN", 5.0),
                ("T2M_MAX", 20.0),
                ("PRECTOTCORR", 1.0),
                ("ALLSKY_SFC_SW_DWN", 15.0),
                ("RH2M", 70.0),
                ("WS2M", 3.0),
            ]
        }
        f = series_to_forcing(data, year)
        assert f.tmin.shape == (1, 365)


class TestClientMocked:
    def test_fetch_roundtrip(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "power.larc.nasa.gov" in str(request.url)
            return httpx.Response(200, json=_payload())

        client = Power(httpx.Client(transport=httpx.MockTransport(handler)))
        output = client.fetch(
            None, Temporal("2020-01-01", "2020-01-03"), latitude=42.0, longitude=-93.0
        )
        assert output["T2M_MAX"]["20200103"] == 20.0

    def test_http_error_raises(self):
        client = Power(httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))))
        with pytest.raises(httpx.HTTPStatusError):
            client.fetch(None, Temporal("2020-01-01", "2020-01-03"), latitude=0.0, longitude=0.0)
