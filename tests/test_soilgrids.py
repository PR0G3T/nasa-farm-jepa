"""SoilGrids client — payload parsing into SoilProfile."""

import httpx
import pytest

from nasa_farm.acquire.soilgrids import (
    SoilGridsClient, parse_soilgrids, request_params,
)

PAYLOAD = {
    "properties": {"layers": [
        {"name": "clay", "depths": [{"values": {"mean": 220}}]},
        {"name": "sand", "depths": [{"values": {"mean": 420}}]},
        {"name": "soc", "depths": [{"values": {"mean": 12}}]},
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
    assert float(p.clay1[0]) == pytest.approx(22.0)
    assert float(p.layer1.theta_fc) > float(p.layer1.theta_wp)


def test_profile_fallback_on_error():
    client = SoilGridsClient(httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    p = client.profile(0.0, 0.0, fallback_texture="clay")
    assert float(p.clay1[0]) == pytest.approx(50.0)
