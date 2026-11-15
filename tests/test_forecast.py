"""Forecast forcing — Open-Meteo clients, doy bridging, provider dispatch."""

import datetime as dt

import httpx2 as httpx
import pytest
import torch

from nasa_farm.acquire.openmeteo import (
    OpenMeteo,
    members_to_forcing,
)
from nasa_farm.forcing import ensemble, synthetic_year
from nasa_farm.forcing.seam import bridge, roll_to_ordinal, span
from nasa_farm.types import FORCING_VARS


def _payload(ndays: int = 20, members: int = 4, start: dt.date = dt.date(2026, 3, 1)) -> dict:
    days = [(start + dt.timedelta(d)).isoformat() for d in range(ndays)]
    daily = {"time": days}
    for m in range(1, members + 1):
        for v, base in [
            ("temperature_2m_min", 4.0),
            ("temperature_2m_max", 18.0),
            ("precipitation_sum", 1.0),
            ("shortwave_radiation_sum", 16.0),
            ("relative_humidity_2m_mean", 65.0),
            ("wind_speed_10m_mean", 3.0),
        ]:
            daily[f"{v}_member{m:02d}"] = [base + 0.1 * m] * ndays
    return {"daily": daily, "daily_units": {}}


class TestClient:
    def test_ens_roundtrip(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "ensemble-api.open-meteo.com" in str(request.url)
            assert request.url.params["models"] == "ecmwf_ifs025"
            assert request.url.params["wind_speed_unit"] == "ms"
            return httpx.Response(200, json=_payload())

        c = OpenMeteo(httpx.Client(transport=httpx.MockTransport(handler)))
        f = c.ensemble(42.0, -93.0, members=51)
        assert f.tmin.shape == (4, 20)
        # member 2 is warmer than member 1 by construction
        assert float(f.tmax[1, 0] - f.tmax[0, 0]) == pytest.approx(0.1)
        # 2026-03-01 is doy 60 on the 365-day axis
        assert int(f.doy[0, 0]) == 60

    def test_doy_leap_drop(self):
        # leap year: Mar 1 is yday 61 → doy 60 after Feb-29 removal
        p = _payload(start=dt.date(2028, 2, 28), ndays=3)

        def handler(request):
            return httpx.Response(200, json=p)

        c = OpenMeteo(httpx.Client(transport=httpx.MockTransport(handler)))
        f = c.ensemble(0.0, 0.0, members=1)
        # Feb 29 dropped: 2 days survive, Mar 1 lands on doy 60
        assert f.days == 2
        assert [int(d) for d in f.doy[0]] == [59, 60]

    def test_missing_member_key_raises(self):
        def handler(request):
            return httpx.Response(200, json={"daily": {"time": ["2026-01-01"]}})

        c = OpenMeteo(httpx.Client(transport=httpx.MockTransport(handler)))
        with pytest.raises(RuntimeError):
            c.ensemble(0.0, 0.0)

    def test_missing_var_fills(self):
        cols = {1: {"temperature_2m_min": [1.0, 2.0]}, 2: {"temperature_2m_min": [3.0, 4.0]}}
        f = members_to_forcing(["2026-06-01", "2026-06-02"], cols, 10)
        assert f.tmin.shape == (2, 2)
        assert float(f.precip.abs().max()) == 0.0
        assert float(f.rh.mean()) == pytest.approx(70.0)

    def test_tmax_floor(self):
        cols = {1: {"temperature_2m_min": [10.0], "temperature_2m_max": [5.0]}}
        f = members_to_forcing(["2026-01-15"], cols, 1)
        assert float(f.tmax[0, 0]) == pytest.approx(10.0)


class TestSeam:
    def test_roll_to_doy(self):
        y = synthetic_year(40.0, seed=0)
        r = roll_to_ordinal(y, 200)
        assert int(r.doy[0, 0]) == 200
        assert int(r.doy[0, -1]) == 199  # wraps
        assert float(r.tmax[0, 0]) == pytest.approx(float(y.tmax[0, 199]))

    def test_roll_requires_year(self):
        y = synthetic_year(40.0, seed=0)
        with pytest.raises(ValueError):
            roll_to_ordinal(span(y, 0, 100), 5)

    def test_bridge_lengths_and_head(self):
        fc = span(synthetic_year(40.0, seed=0), 0, 15).repeat(3)
        history = [synthetic_year(40.0, seed=s) for s in (1, 2)]
        output = bridge(fc, history, 400, seed=0)
        assert output.tmin.shape == (3, 400)
        for k in FORCING_VARS:
            assert torch.equal(getattr(output, k)[:, :15], getattr(fc, k))
        # doy axis stays consecutive on the 365 cycle past the seam
        assert int(output.doy[0, 15]) == 16

    def test_bridge_truncates(self):
        fc = synthetic_year(40.0, seed=0)
        output = bridge(fc, [fc], 100)
        assert output.days == 100

    def test_bridge_picks_matching_year(self):
        """A member identical to one historical year's window must
        continue with *that* year — analog conditioning works."""
        match = synthetic_year(40.0, seed=7)
        other = synthetic_year(40.0, tmean_annual=-20.0, seed=1)
        other2 = synthetic_year(40.0, tmean_annual=45.0, seed=2)
        fc = span(match, 0, 20)
        output = bridge(fc, [other, match, other2], 365, seed=0)
        assert torch.equal(output.tmin[0], match.tmin[0])

    def test_member_count_preserved(self):
        fc = span(synthetic_year(40.0, seed=0), 0, 30).repeat(8)
        output = bridge(fc, [synthetic_year(40.0, seed=s) for s in range(6)], 730, seed=3)
        assert output.batch == 8 and output.days == 730


class TestProviders:
    def _client(self):
        def handler(request):
            return httpx.Response(200, json=_payload())

        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_ens_dispatch(self):
        history = [synthetic_year(40.0, seed=s) for s in range(4)]
        output = ensemble(
            "ens", 40.0, -93.0, (), 2, 51, 0, client=OpenMeteo(self._client()), history=history
        )
        assert output.tmin.shape == (4, 730)
        assert int(output.doy[0, 0]) == 60

    def test_seas5_dispatch(self):
        history = [synthetic_year(40.0, seed=s) for s in range(3)]
        output = ensemble(
            "seas5", 40.0, -93.0, (), 1, 51, 0, client=OpenMeteo(self._client()), history=history
        )
        assert output.tmin.shape == (4, 365)

    def test_members_cap(self):
        c = OpenMeteo(self._client())
        assert c.ensemble(0.0, 0.0, members=2).batch == 2

    def test_unknown_source(self):
        with pytest.raises(KeyError):
            ensemble("bogus", 0.0, 0.0, (), 1, 8, 0)


class TestServiceSource:
    def test_offline_synth_source(self):
        from nasa_farm.service import PlanRequest, Point, plan

        request = PlanRequest(
            points=[Point(latitude=42.0, longitude=-93.0)],
            horizon=1,
            years=2,
            crops=["maize"],
            offline=True,
            source="synth",
            members=4,
        )
        result = plan(request).results[0]
        assert result.status == "ok"
        assert result.scenarios == 4
