"""SoilGrid client — payload parsing into flat dicts + SoilProfiles."""

import httpx2 as httpx
import pytest

from nasa_farm.acquire.soilgrid import (
    DEFAULTS, SoilGridClient, parse, request_params,
)

PAYLOAD = {
    "properties": {"layers": [
        {"name": "clay", "depths": [{"values": {
            "mean": 220, "Q0.05": 150, "Q0.5": 210, "Q0.95": 300}}]},
        {"name": "sand", "depths": [{"values": {
            "mean": 420, "Q0.05": 350, "Q0.5": 410, "Q0.95": 500}}]},
        {"name": "soc", "depths": [{"values": {
            "mean": 12, "Q0.05": 6, "Q0.5": 10, "Q0.95": 20}}]},
        {"name": "bdod", "depths": [{"values": {
            "mean": 140, "Q0.05": 110, "Q0.5": 135, "Q0.95": 160}}]},
        {"name": "phh2o", "depths": [{"values": {
            "mean": 68, "Q0.05": 55, "Q0.5": 65, "Q0.95": 78}}]},
    ]}
}


def _client(status: int = 200) -> SoilGridClient:
    return SoilGridClient(httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(status, json=PAYLOAD))))


def test_request_params_multivalued():
    params = dict(request_params(41.0, -93.0))
    assert params["lat"] == "41.0" and params["depth"] == "0-30cm"


def test_parse_units():
    out = parse(PAYLOAD)
    assert out["mean"]["clay"] == pytest.approx(22.0)
    assert out["mean"]["sand"] == pytest.approx(42.0)
    assert out["mean"]["soc"] == pytest.approx(1.2)      # dg/kg → g/kg
    assert out["mean"]["bdod"] == pytest.approx(1.4)     # cg/cm³ → g/cm³
    assert out["mean"]["phh2o"] == pytest.approx(6.8)    # pH×10 → pH
    assert out["Q0.05"]["clay"] == pytest.approx(15.0)
    assert out["Q0.95"]["sand"] == pytest.approx(50.0)


def test_sample_flat_dict():
    d = _client().sample(41.0, -93.0)
    assert d["clay"] == pytest.approx(21.0)              # Q0.5 median
    assert d["sand"] == pytest.approx(41.0)
    assert d["bd"] == pytest.approx(1.35)
    assert d["soc"] == pytest.approx(1.0)
    assert d["ph"] == pytest.approx(6.5)
    assert d["clay_lo"] == pytest.approx(15.0)
    assert d["clay_hi"] == pytest.approx(30.0)
    assert d["sand_lo"] == pytest.approx(35.0)
    assert d["bd_hi"] == pytest.approx(1.6)


def test_sample_fallback_on_error():
    d = _client(500).sample(0.0, 0.0)
    assert d["clay"] == DEFAULTS["clay"]
    assert d["clay_lo"] < d["clay"] < d["clay_hi"]


def test_profile_from_mocked_api():
    p = _client().profile(41.0, -93.0)
    assert float(p.clay[0, 0]) == pytest.approx(21.0)
    assert float(p.layer(0).theta_fc) > float(p.layer(0).theta_wp)


def test_profile_fallback():
    p = _client(500).profile(0.0, 0.0)
    assert float(p.clay[0, 0]) == pytest.approx(DEFAULTS["clay"])


def test_bounds_ordered():
    lo, mid, hi = _client().bounds(41.0, -93.0)
    assert float(lo.clay[0, 0]) < float(mid.clay[0, 0]) \
        < float(hi.clay[0, 0])
    assert float(mid.clay[0, 0]) == pytest.approx(21.0)


def test_profiles_ensemble_within_bounds():
    c = _client()
    lo, _, hi = c.bounds(41.0, -93.0)
    s = c.profiles(41.0, -93.0, 16, seed=0)
    assert s.batch == 16
    assert (s.clay >= lo.clay.min() - 1e-9).all() \
        and (s.clay <= hi.clay.max() + 1e-9).all()
