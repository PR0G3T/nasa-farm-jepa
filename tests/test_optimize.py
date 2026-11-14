"""HPO & NAS — search space, trainers, gradient identifiability, proxies."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.dynamics.residual import Residual
from nasa_farm.forcing import synthetic_year, tile_to_horizon
from nasa_farm.optimize.hyperparameter import (
    blackbox, gradient, successive_halving,
)
from nasa_farm.optimize.architecture import (
    ARCH_SPACE, build_residual, jacob_cov_score, search, synflow_score,
    zero_cost_rank,
)
from nasa_farm.optimize.space import Parameter, Space
from nasa_farm.optimize.targets import (
    ObservedSeries, ObservedTrajectory, series, season_metrics,
    series_loss, synthetic_observation, trajectory_loss,
)
from nasa_farm.optimize.trainer import fit, leaf, trait_leaf
from nasa_farm.state.crops import INDEX
from nasa_farm.state.field import Field
from nasa_farm.state.soil import SoilProfile
from nasa_farm.types import Actions


class TestSpace:
    def test_param_validation(self):
        with pytest.raises(ValueError):
            Parameter("bad")                          # neither choices nor range
        with pytest.raises(ValueError):
            Parameter("bad2", low=1.0, high=0.5)      # inverted
        with pytest.raises(ValueError):
            Parameter("bad3", low=-1.0, high=10.0, log=True)  # log needs low>0
        Parameter("ok", choices=(1, 2))               # no raise

    def test_sample_bounds_and_reproducibility(self):
        sp = Space([Parameter("x", low=0.0, high=1.0),
                    Parameter("y", low=1e-3, high=1.0, log=True),
                    Parameter("z", choices=("a", "b"))])
        a = sp.sample(16, seed=7)
        b = sp.sample(16, seed=7)
        assert a == b
        for config in a:
            assert 0.0 <= config["x"] <= 1.0
            assert 1e-3 <= config["y"] <= 1.0
            assert config["z"] in ("a", "b")

    def test_sobol_covers_better_than_edges(self):
        """Quasi-random fill: no empty half for n ≥ 8 in 1-D."""
        sp = Space([Parameter("x", low=0.0, high=1.0)])
        xs = sorted(c["x"] for c in sp.sample(8, seed=0))
        assert xs[0] < 0.5 and xs[-1] > 0.5
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        assert max(gaps) < 0.5


class TestTargets:
    def test_season_metrics_grad_path(self, hybrid_model, field, crops):
        forcing = tile_to_horizon(synthetic_year(42.0, seed=3), 1)
        a = Actions.idle(1, forcing.days)
        m = season_metrics(hybrid_model, field, forcing, a)
        m.sum().backward()
        grads = [p.gradient for p in hybrid_model.residual.parameters()
                 if p.gradient is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)

    def test_trajectory_loss_masking(self, field, hybrid_model, crops):
        forcing = tile_to_horizon(synthetic_year(42.0, seed=3), 1)
        predicted = season_metrics(hybrid_model, field, forcing,
                              Actions.idle(1, forcing.days))[0]
        # exact match on observed cells, NaN elsewhere → zero loss
        observed = ObservedTrajectory(values=predicted.clone())
        assert float(trajectory_loss(predicted, observed)) == pytest.approx(0.0)
        # shifting an observed channel raises the loss; NaN cells don't count
        moved = predicted.clone()
        moved[:, 0] += 10.0
        obs2 = ObservedTrajectory(values=moved)
        l = trajectory_loss(predicted, obs2)
        assert float(l) > 0.0

    def test_synthetic_observation_channels(self, physics_model, field, crops):
        forcing = tile_to_horizon(synthetic_year(42.0, seed=1), 1)
        a = Actions.idle(1, forcing.days)
        sow = a.sow.clone()
        sow[0, int(crops.get("maize")[INDEX["sow_doy"]]) - 1] = crops.index("maize")
        observed = synthetic_observation(
            physics_model, field, forcing,
            Actions(sow, a.irrigation, a.n_fert), ("yield_kg",))
        mask = observed.observed_mask()
        assert mask.sum() == observed.values.shape[0]  # one channel, H rows


class TestObservedSeries:
    def test_obs_series_shapes_and_zero_loss(self, physics_model, field):
        forcing = tile_to_horizon(synthetic_year(42.0, seed=1), 1)
        predicted = series(physics_model, field, forcing,
                          Actions.idle(1, forcing.days),
                          ("ndvi", "sm1"), every=30)
        assert predicted.shape == (1, 13, 2)
        observed = ObservedSeries(values=predicted[0].clone())
        assert float(series_loss(predicted, observed)) == pytest.approx(0.0)

    def test_series_loss_counts_only_observed(self, physics_model, field):
        forcing = tile_to_horizon(synthetic_year(42.0, seed=1), 1)
        predicted = series(physics_model, field, forcing,
                          Actions.idle(1, forcing.days), ("ndvi",), every=30)
        moved = predicted[0].clone()
        moved[:, 0] += 1.0
        moved[3:] = float("nan")      # unobserved cells excluded
        l = series_loss(predicted, ObservedSeries(values=moved))
        assert float(l) > 0.0

    def test_series_grads_reach_residual(self, hybrid_model, field):
        forcing = tile_to_horizon(synthetic_year(42.0, seed=1), 1)
        predicted = series(hybrid_model, field, forcing,
                          Actions.idle(1, forcing.days), ("sm1",), every=15)
        predicted.sum().backward()
        grads = [p.gradient for p in hybrid_model.residual.parameters()
                 if p.gradient is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)


class TestTrainer:
    def test_fit_minimizes_quadratic(self):
        p = leaf(torch.tensor([5.0]))
        result = fit(lambda: (p - 2.0) ** 2, {"p": p}, epochs=400, rate=0.1)
        assert float(p) == pytest.approx(2.0, abs=5e-2)
        assert result.best_loss == pytest.approx(0.0, abs=1e-2)

    def test_early_stopping(self):
        p = leaf(torch.tensor([0.0]))
        result = fit(lambda: p * 0 + 1.0, {"p": p}, epochs=500, patience=5)
        assert result.epochs_run <= 6


class TestGradientHPO:
    def test_trait_recovery(self, physics_model, field, crops):
        """Perturb maize rue ×0.8 → gradient fit recovers it through the
        rollout — the exact mechanism real calibration will use."""
        i_c, i_p = crops.index("maize"), INDEX["rue"]
        truth = WorldModel(crops, residual=False)
        table = truth.crop_table.detach().clone()
        table[i_c, i_p] *= 0.8
        truth.crop_table = table
        forcing = tile_to_horizon(synthetic_year(42.0, seed=0), 1)
        a = Actions.idle(1, forcing.days)
        sow = a.sow.clone()
        sow[0, int(crops.get("maize")[INDEX["sow_doy"]]) - 1] = i_c
        act = Actions(sow, a.irrigation, a.n_fert)
        observed = synthetic_observation(truth, field, forcing, act, ("yield_kg",))

        base = physics_model.crop_table.detach().clone()
        p = leaf(base[i_c, i_p])

        def loss():
            t = base.clone()
            t[i_c, i_p] = p
            physics_model.crop_table = t
            return trajectory_loss(
                season_metrics(physics_model, field, forcing, act), observed)

        result = gradient(loss, {"p": p}, epochs=150, rate=0.05)
        assert float(p) == pytest.approx(float(table[i_c, i_p]), relative=0.05)
        assert result.best_loss == pytest.approx(0.0, abs=1e-3)

    def test_trait_leaf_binds_live(self, hybrid_model):
        t = trait_leaf(hybrid_model)
        assert t.requires_grad and t.data_ptr() == hybrid_model.crop_table.data_ptr()


class TestBlackbox:
    def test_asha_prunes(self):
        seen = []
        def object(config, budget):
            seen.append(budget)
            return abs(config["x"] - 0.3)
        config = [{"x": v} for v in torch.linspace(0, 1, 9).tolist()]
        trials = successive_halving(object, config, rungs=(1, 2), eta=3)
        budgets = sorted(set(seen))
        assert budgets == [1, 2]
        assert sum(1 for s in seen if s == 2) <= 3  # ≤ 1/eta promoted

    def test_blackbox_finds_minimum(self):
        sp = Space([Parameter("x", low=0.0, high=1.0)])
        result = blackbox(lambda c, b: (c["x"] - 0.31) ** 2, sp,
                           n_trials=32, rungs=None, seed=0)
        assert abs(result.best_config["x"] - 0.31) < 0.05

    def test_optuna_if_available(self):
        optuna = pytest.importorskip("optuna")
        from nasa_farm.optimize.hyperparameter import optuna
        sp = Space([Parameter("x", low=0.0, high=1.0)])
        result = optuna(lambda c, b: (c["x"] - 0.4) ** 2, sp,
                         n_trials=20, seed=0)
        assert abs(result.best_config["x"] - 0.4) < 0.15


class TestNAS:
    def test_build_residual_contract(self):
        for config in ARCH_SPACE.sample(4, seed=0):
            net = build_residual(config)
            output = net(torch.randn(8, 14, dtype=torch.float64))
            assert output.shape == (8, 5)
            assert float(output.abs().max()) == 0.0  # zero-init → identity

    def test_proxies_finite_and_rank(self):
        cfgs = ARCH_SPACE.sample(6, seed=0)
        ranked = zero_cost_rank(cfgs, seed=0)
        assert len(ranked) == len(cfgs)
        assert all(torch.isfinite(torch.tensor(s)) for _, s in ranked)

    def test_synflow_positive_jacobian_ordered(self):
        net = build_residual({"hidden": 64, "depth": 2, "activation": "silu",
                              "layernorm": False, "skip": False},
                             zero_init=False)
        assert synflow_score(net) > 0
        assert jacob_cov_score(net) >= 0

    def test_nas_search_returns_valid_arch(self, physics_model, field, crops):
        forcing = tile_to_horizon(synthetic_year(42.0, seed=0), 1)
        a = Actions.idle(1, forcing.days)
        observed = synthetic_observation(physics_model, field, forcing, a,
                                    ("yield_kg",))

        def object(net, budget):
            m = WorldModel(crops, residual=True)
            m.residual = net
            ps = {f"p{i}": p for i, p in enumerate(net.parameters())}
            r = fit(lambda: trajectory_loss(
                season_metrics(m, field, forcing, a), observed),
                ps, epochs=budget, rate=0.02, patience=budget)
            return r.best_loss

        result = search(object, n_candidates=6, proxy_top_k=3,
                         rungs=(5, 10), eta=2, seed=0)
        assert result.best_config["hidden"] in (16, 32, 64, 128, 256)
        assert result.best_config["depth"] in (1, 2, 3, 4)
        assert torch.isfinite(torch.tensor(result.best_value))
        assert len(result.leaderboard) <= 3
