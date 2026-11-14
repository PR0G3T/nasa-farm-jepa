"""Experiment harness — spec roundtrips, registry, determinism, and the
scientific invariants every runner must satisfy."""

import json

import pytest
import torch

from nasa_farm.commands import main
from nasa_farm.experiments import Report, Specification, experiments, run


def specification(**keywords) -> Specification:
    base = dict(crop="maize", horizon=1, members=6, cadence=20,
                noise=0.05, perturb={"rue": 0.8}, epochs=10, seed=0,
                channels=("yield_kg",))
    return Specification(**(base | keywords))


class TestSpec:
    def test_json_roundtrip(self):
        s = specification(years=(2020, 2021), alphas=(0.1, 0.5))
        assert Specification.from_json(s.to_json()) == s

    def test_from_dict_ignores_unknown(self):
        d = specification().as_dict() | {"future_field": 1}
        assert Specification.from_dict(d) == specification()


class TestRegistry:
    def test_experiments_registered(self):
        assert {"twin", "bench", "assim"} <= set(experiments())

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            run("nope", specification())


class TestTwin:
    def test_structure_and_identifiability(self):
        r = run("twin", specification(epochs=15))
        p = r.results["calibration"]["params"]["rue"]
        assert set(p) == {"start", "truth", "fitted", "recovery_pct"}
        # fit must move the trait toward the perturbed truth
        assert abs(p["fitted"] - p["truth"]) < abs(p["start"] - p["truth"])
        ch = r.results["channels"]["yield_kg"]
        assert ch["cal"]["rmse"] < ch["prior"]["rmse"]
        assert ch["rmse_reduction_pct"] > 0

    def test_deterministic(self):
        assert run("twin", specification(epochs=5)).results == \
            run("twin", specification(epochs=5)).results


class TestBench:
    def test_predictors_and_calibration_gain(self):
        r = run("bench", specification(epochs=15, members=8))
        ch = r.results["channels"]["yield_kg"]
        assert {"prior", "cal", "oracle", "summary"} <= set(ch)
        for name in ("prior", "cal", "oracle"):
            assert "crps" in ch[name] and "picp_0.1" in ch[name]
        assert ch["cal"]["rmse"] < ch["prior"]["rmse"]

    def test_multi_channel(self):
        r = run("bench", specification(epochs=5, members=4,
                              channels=("yield_kg", "water_used")))
        assert set(r.results["channels"]) == {"yield_kg", "water_used"}


class TestAssim:
    def test_variants_and_enkf_gain(self):
        r = run("assim", specification(members=8, cadence=20))
        result = r.results
        assert {"open", "enkf", "ensf", "reduction_pct"} <= set(result)
        for name in ("open", "enkf", "ensf"):
            v = result[name]
            assert {"obs_rmse", "obs_crps", "spread", "state_rmse",
                    "channels"} <= set(v)
            assert set(v["state_rmse"]) == \
                {"sw0", "swr", "n_min", "lai", "biomass"}
        # filtering must beat open-loop on end-of-season yield
        assert (result["enkf"]["channels"]["yield_kg"]["rmse"]
                < result["open"]["channels"]["yield_kg"]["rmse"])

    def test_deterministic(self):
        assert run("assim", specification(members=6)).results == \
            run("assim", specification(members=6)).results


class TestReport:
    def test_save_load_roundtrip(self, tmp_path):
        r = run("twin", specification(epochs=5))
        p = r.save(tmp_path / "r.json")
        r2 = Report.load(p)
        assert r2 == r and r2.specification["perturb"] == {"rue": 0.8}

    def test_flat_and_latex(self):
        r = run("twin", specification(epochs=5))
        keys = dict(r.flat())
        assert "calibration.params.rue.recovery_pct" in keys
        tex = r.latex()
        assert "\\toprule" in tex and "\\bottomrule" in tex


class TestCli:
    def test_cli_exp_json(self, capsys):
        code = main(["exp", "twin", "--perturb", "rue:0.8", "--epochs", "5",
                   "--members", "4"])
        assert code == 0
        output = json.loads(capsys.readouterr().output)
        assert output["experiment"] == "twin"
        assert output["results"]["calibration"]["params"]["rue"]["truth"] \
            == pytest.approx(3.9 * 0.8)

    def test_cli_exp_out(self, tmp_path, capsys):
        f = tmp_path / "rep.json"
        code = main(["exp", "twin", "--epochs", "5", "--perturb", "rue:0.9",
                   "--out", str(f)])
        assert code == 0 and f.exists()
        assert Report.load(f).name == "twin"
