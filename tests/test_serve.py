"""Serving layer — service.plan contract, HTTP API, CLI export/serve.

All compute runs `offline=True` (synthetic forcing, deterministic) so the
suite never touches the network; fastapi-dependent tests skip cleanly when
the `serve` extra is absent."""

import json

import pytest
from pydantic import ValidationError

from nasa_farm.commands import main
from nasa_farm.service import PlanRequest, Point, plan

REQUEST = {"points": [{"lat": 48.2, "lon": 1.61}],
       "horizon": 2, "crops": ["maize", "fallow"], "years": 2,
       "offline": True, "top_k": 2}


def test_request_validation_bounds():
    with pytest.raises(ValidationError):
        PlanRequest(points=[Point(latitude=200.0, longitude=0.0)])
    with pytest.raises(ValidationError):
        PlanRequest(points=[])
    with pytest.raises(ValidationError):
        PlanRequest(points=[Point(latitude=0.0, longitude=0.0)], horizon=0)


def test_plan_single_site():
    result = plan(PlanRequest(**REQUEST)).results[0]
    assert result.status == "ok"
    assert result.plans_evaluated == 4          # 2 crops ^ 2y
    assert result.scenarios == 2               # years=2 → 2 scenarios
    assert len(result.top) == 2 and len(result.pareto) >= 1
    assert all("rotation" in r for r in result.top)
    assert all("yield_mean" in r for r in result.pareto)


def test_plan_batch_aligned_and_independent():
    request = PlanRequest(**{**REQUEST, "points": [
        {"id": "a", "lat": 48.2, "lon": 1.61},
        {"id": "b", "lat": 42.0, "lon": -93.6}]})
    result = plan(request)
    assert [r.point.id for r in result.results] == ["a", "b"]
    assert all(r.status == "ok" for r in result.results)


def test_plan_unknown_crop_is_request_error():
    with pytest.raises(ValueError, match="unknown crops"):
        plan(PlanRequest(**{**REQUEST, "crops": ["maize", "kudzu"]}))


def test_plan_site_error_degrades_not_raises():
    result = plan(PlanRequest(**{**REQUEST, "soil": "not_a_texture"})).results[0]
    assert result.status == "error" and result.detail


def test_cli_export_writes_scenarios(tmp_path):
    source = tmp_path / "f.json"
    source.write_text(json.dumps(
        [{"id": "P1", "name": "P1", "lat": 48.2, "lon": 1.61}]))
    output = tmp_path / "plan.json"
    code = main(["export", "--fields", str(source), "--out", str(output),
               "--horizon", "1", "--crops", "maize,fallow", "--years", "1",
               "--offline", "--top-k", "1",
               "--scenario", "baseline", "--scenario", "warm:delta_t=2"])
    assert code == 0
    data = json.loads(output.read_text())
    assert set(data["scenarios"]) == {"baseline", "warm"}
    assert data["scenarios"]["warm"]["params"] == {"delta_t": 2.0}
    hit = data["scenarios"]["baseline"]["fields"]["P1"]
    assert hit["status"] == "ok" and hit["plans_evaluated"] == 2


def test_cli_export_rejects_unknown_scenario_key(tmp_path):
    source = tmp_path / "f.json"
    source.write_text("[]")
    with pytest.raises(SystemExit):
        main(["export", "--fields", str(source), "--out",
              str(tmp_path / "o.json"), "--offline",
              "--scenario", "bad:notakey=1"])


def test_cli_plan_unchanged_shape(capsys):
    code = main(["plan", "--lat", "48.2", "--lon", "1.61", "--horizon", "2",
               "--crops", "maize,fallow", "--years", "2", "--offline",
               "--top-k", "2"])
    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert set(output) == {"plans_evaluated", "scenarios", "pareto", "top"}
    assert output["plans_evaluated"] == 4


class TestAPI:
    fastapi = pytest.importorskip("fastapi")

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from nasa_farm.server import create_app
        return TestClient(create_app())

    def test_healthz(self, client):
        body = client.get("/healthz").json()
        assert body["status"] == "ok" and body["version"]

    def test_meta_vocab(self, client):
        body = client.get("/v1/meta").json()
        assert "maize" in body["crops"]
        assert body["objectives"]["yield_mean"] == 1
        assert "yield_kg" in body["metrics"]

    def test_plan_endpoint(self, client):
        body = client.post("/v1/plan", json=REQUEST).json()
        result = body["results"][0]
        assert result["status"] == "ok" and result["plans_evaluated"] == 4
        assert result["point"]["lat"] == 48.2

    def test_plan_cache_hit(self, client):
        r1 = client.post("/v1/plan", json=REQUEST)
        r2 = client.post("/v1/plan", json=REQUEST)
        assert "x-cache" not in r1.headers
        assert r2.headers.get("x-cache") == "hit"
        assert r1.json() == r2.json()

    def test_plan_bad_input_is_422(self, client):
        result = client.post("/v1/plan", json={**REQUEST, "points": [
            {"lat": 200.0, "lon": 0.0}]})
        assert result.status_code == 422

    def test_plan_unknown_crop_is_400(self, client):
        result = client.post("/v1/plan", json={**REQUEST, "crops": ["kudzu"]})
        assert result.status_code == 400
        assert "unknown crops" in result.json()["detail"]


def test_tunnel_url_regex():
    from nasa_farm.server import TUNNEL_URL
    line = ("INF |  Your quick Tunnel has been created! "
            "https://abc-def-123.trycloudflare.com")
    assert TUNNEL_URL.search(line).group(0) == \
        "https://abc-def-123.trycloudflare.com"
    assert TUNNEL_URL.search("https://example.com") is None
