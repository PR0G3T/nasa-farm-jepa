"""SoilGrids client — payload parsing into SoilProfile + uncertainty."""

import httpx2 as httpx
import pytest

from nasa_farm.acquire.soilgrids import (
    SoilGridsClient, parse_dist, parse_soilgrids, request_params,
)

PAYLOAD = {
    "properties": {"layers": [
        {"name": "clay", "depths": [{"values": {
            "mean": 220, "Q0.05": 150, "Q0.5": 210, "Q0.95": 300}}]},
        {"name": "sand", "depths": [{"values": {
            "mean": 420, "Q0.05": 350, "Q0.5": 410, "Q0.95": 500}}]},
        {"name": "soc", "depths": [{"values": {
            "mean": 12, "Q0.05": 6, "Q0.5": 10, "Q0.95": 20}}]},
    ]}
}


def test_request_params_multivalued():
    params = dict((k, v) for k, v in request_params(41.0, -93.0))
    assert params["lat"] == "41.0" and params["depth"] == "0-30cm"


def test_parse_units_to_percent():
    out = parse_soilgrids(PAYLOAD)
    assert out["clay"] == pytest.approx(22.0)
    assert out["sand"] == pytest.approx(42.0)
    assert out["soc"] == pytest.approx(1.2)


def test_profile_from_mocked_api():
    client = SoilGridsClient(httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=PAYLOAD))))
    p = client.profile(41.0, -93.0)
    assert float(p.clay[0, 0]) == pytest.approx(21.0)   # Q0.5 median
    assert float(p.layer(0).theta_fc) > float(p.layer(0).theta_wp)


def test_profile_fallback_on_error():
    client = SoilGridsClient(httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    p = client.profile(0.0, 0.0, fallback_texture="clay")
    assert float(p.clay[0, 0]) == pytest.approx(50.0)


def test_parse_dist_all_stats():
    d = parse_dist(PAYLOAD)
    assert d["Q0.05"]["clay"] == pytest.approx(15.0)
    assert d["Q0.5"]["clay"] == pytest.approx(21.0)
    assert d["Q0.95"]["sand"] == pytest.approx(50.0)


def test_bounds_ordered():
    client = SoilGridsClient(httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=PAYLOAD))))
    lo, mid, hi = client.bounds(41.0, -93.0)
    assert float(lo.clay[0, 0]) < float(mid.clay[0, 0]) \
        < float(hi.clay[0, 0])
    assert float(mid.clay[0, 0]) == pytest.approx(21.0)


def test_profiles_ensemble_within_bounds():
    client = SoilGridsClient(httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=PAYLOAD))))
    lo, _, hi = client.bounds(41.0, -93.0)
    s = client.profiles(41.0, -93.0, 16, seed=0)
    assert s.batch == 16
    assert (s.clay >= lo.clay.min() - 1e-9).all() \
        and (s.clay <= hi.clay.max() + 1e-9).all()


def test_fallback_bounds_degenerate():
    client = SoilGridsClient(httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    lo, mid, hi = client.bounds(0.0, 0.0, fallback_texture="clay")
    assert float(lo.clay[0, 0]) == float(mid.clay[0, 0]) \
        == float(hi.clay[0, 0]) == pytest.approx(50.0)
