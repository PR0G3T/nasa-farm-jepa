"""NLDAS-3 — domain guard, unit conversions, canonical Forcing build."""

import datetime as dt

import pytest

from nasa_farm.acquire.reanalysis import (
    _rh, covers, convert, day_url, year_point,
)


class TestDomain:
    def test_covers(self):
        assert covers(40.0, -100.0)          # Kansas
        assert not covers(48.2, 1.6)         # France
        assert not covers(0.0, 0.0)

    def test_year_outside_raises(self):
        with pytest.raises(ValueError):
            year_point(48.2, 1.6, 2020)

    def test_day_url(self):
        assert day_url(dt.date(2010, 7, 22)) == (
            "https://nasa-waterinsight.s3.amazonaws.com/NLDAS3/forcing/"
            "daily/201007/NLDAS_FOR0010_D.A20100722.030.beta.nc")


class TestConvert:
    def test_units(self):
        out = convert({
            "Tair": 293.15, "Tair_min": 283.15, "Tair_max": 303.15,
            "Qair": 0.008, "PSurf": 101325.0,
            "Wind_E": 3.0, "Wind_N": 4.0,
            "SWdown": 250.0, "Rainf": 5.0})
        assert out["tmin"] == pytest.approx(10.0)
        assert out["tmax"] == pytest.approx(30.0)
        assert out["srad"] == pytest.approx(21.6)      # W/m² → MJ/m²/d
        assert out["precip"] == pytest.approx(5.0)
        assert out["u2"] == pytest.approx(5.0 * 0.748)  # hypot × FAO-56
        assert 1.0 <= out["rh"] <= 100.0

    def test_rainf_rate_converts(self):
        out = convert({"Rainf": 5e-5, "SWdown": 0.0, "Wind_E": 0, "Wind_N": 0})
        assert out["precip"] == pytest.approx(4.32)

    def test_rh_bounds(self):
        assert _rh(0.050, 310.0, 101325.0) == pytest.approx(100.0)
        assert _rh(0.0, 293.0, 101325.0) == pytest.approx(1.0)
        sat = _rh(0.0105, 293.15, 101325.0)
        assert 60.0 < sat < 80.0
