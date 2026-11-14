"""Full-pipeline integration — state → scenarios → enumerate → Pareto → CLI."""

import json

import pytest
import torch

from nasa_farm.commands import main
from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.forcing import climatology_ensemble, synthetic_year
from nasa_farm.plan import enumerate_rotations, evaluate, recommend
from nasa_farm.state.crops import CropLibrary
from nasa_farm.state.field import Field, initial_state
from nasa_farm.state.soil import SoilProfile


@pytest.fixture
def pipeline():
    crops = CropLibrary.default()
    soil = SoilProfile.preset("silt_loam")
    field = Field(initial_state(1, soil), soil, torch.tensor([41.9], dtype=DTYPE))
    ensemble = climatology_ensemble(
        [synthetic_year(41.9, tmean_annual=13.0, seed=s) for s in range(4)], 3)
    return crops, field, ensemble


def test_bare_fallow_leaches_most(pipeline):
    """No crop uptake on fallow → mineralized N accumulates → worst leaching.
    This is the documented nitrate-leaching risk of bare fallow."""
    crops, field, ensemble = pipeline
    model = WorldModel(crops, residual=False)
    mz = crops.index("maize")
    plans = torch.tensor([
        [mz, mz, mz],
        [crops.index("maize"), crops.index("soybean"), crops.index("clover")],
        [crops.index("fallow")] * 3,
    ])
    event = evaluate(model, field, plans, ensemble)
    leach = event.objectives["n_leached"]
    assert float(leach[2]) > float(leach[0])
    assert float(leach[2]) > float(leach[1])
    assert float(event.objectives["yield_mean"][0]) > 0.0
    assert float(event.objectives["yield_mean"][2]) == 0.0


def test_pareto_nonempty_and_finite(pipeline):
    crops, field, ensemble = pipeline
    model = WorldModel(crops, residual=False)
    index = [crops.index(n) for n in ("maize", "soybean", "wheat", "clover")]
    plans = enumerate_rotations(index, 3)
    record = recommend(evaluate(model, field, plans, ensemble), top_k=5)
    assert 1 <= len(record["pareto_idx"]) <= len(plans)
    for i in record["ranking"]:
        for k, v in record["objectives"].items():
            assert torch.isfinite(v[i])


def test_cli_plan_offline(capsys):
    code = main(["plan", "--lat", "42.0", "--lon", "-93.6", "--horizon", "2",
               "--crops", "maize,soybean,fallow", "--years", "2", "--offline",
               "--top-k", "3"])
    assert code == 0
    output = json.loads(capsys.readouterr().output)
    assert output["plans_evaluated"] == 9
    assert output["scenarios"] == 2
    assert len(output["top"]) == 3
    assert all("rotation" in p for p in output["pareto"] + output["top"])


def test_cli_config_toml(tmp_path, capsys):
    """TOML supplies defaults; explicit CLI flags still win (later in argv)."""
    config = tmp_path / "p.toml"
    config.write_text(
        '[plan]\nhorizon = 2\ncrops = "maize,fallow"\nyears = 2\n'
        'offline = true\n')
    code = main(["plan", "--lat", "42.0", "--lon", "-93.6",
               "--config", str(config), "--top-k", "2"])
    assert code == 0
    output = json.loads(capsys.readouterr().output)
    assert output["plans_evaluated"] == 4   # 2 crops ^ 2y from TOML
    assert output["scenarios"] == 2
    assert len(output["top"]) == 2          # CLI flag won over nothing → applied
