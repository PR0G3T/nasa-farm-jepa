"""Planning — enumeration, action expansion, Pareto exactness, ranking."""

import pytest
import torch

from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.forcing import climatology_ensemble, synthetic_year
from nasa_farm.plan import (
    enumerate_rotations, evaluate, expand_actions, objective_matrix,
    pareto_front, recommend, scalarize,
)
from nasa_farm.state.crops import INDEX


class TestEnumeration:
    def test_count_and_uniqueness(self):
        plans = enumerate_rotations([0, 1, 2], 3)
        assert plans.shape == (27, 3)
        assert plans.unique(dim=0).shape[0] == 27

    def test_single_year(self):
        plans = enumerate_rotations([0, 4], 1)
        assert plans.tolist() == [[0], [4]]


class TestExpandActions:
    def test_sow_days_and_values(self, crops):
        plans = torch.tensor([[crops.index("maize"), crops.index("soybean")]])
        a = expand_actions(plans, crops.table)
        assert a.days == 2 * DAYS_PER_YEAR
        maize_doy = int(crops.get("maize")[INDEX["sow_doy"]])
        soy_doy = int(crops.get("soybean")[INDEX["sow_doy"]])
        assert int(a.sow[0, maize_doy - 1]) == crops.index("maize")
        assert int(a.sow[0, DAYS_PER_YEAR + soy_doy - 1]) == crops.index("soybean")
        assert int((a.sow >= 0).sum()) == 2

    def test_irrigation_window_bounded(self, crops):
        plans = torch.tensor([[crops.index("maize")]])
        a = expand_actions(plans, crops.table, irr_mm_day=5.0, irr_window=100)
        assert float(a.irrigation.max()) <= 5.0
        assert int((a.irrigation > 0).sum()) <= 100
        assert float(a.irrigation.sum()) == pytest.approx(
            5.0 * int((a.irrigation > 0).sum()))

    def test_fert_single_shot(self, crops):
        plans = torch.tensor([[crops.index("maize")]])
        a = expand_actions(plans, crops.table, n_fert_kg=80.0, fert_delay=30)
        assert float(a.n_fert.sum()) == pytest.approx(80.0)


class TestPareto:
    def test_known_front(self):
        points = torch.tensor([[1.0, 1.0], [2.0, 2.0], [1.5, 1.0], [0.5, 0.5]],
                           dtype=DTYPE)
        signs = torch.ones(2, dtype=DTYPE)  # maximize both
        mask = pareto_front(points, signs)
        assert mask.tolist() == [False, True, False, False]

    def test_knee_points_kept(self):
        """Trade-off extremes survive; the middle dominated point drops."""
        points = torch.tensor([[2.0, 0.5], [1.5, 1.5], [0.5, 2.0], [1.0, 1.0]],
                           dtype=DTYPE)
        mask = pareto_front(points, torch.ones(2, dtype=DTYPE))
        assert mask.tolist() == [True, True, True, False]

    def test_signs_flip_direction(self):
        points = torch.tensor([[1.0, 10.0], [5.0, 1.0]], dtype=DTYPE)
        both_max = pareto_front(points, torch.ones(2, dtype=DTYPE))
        assert both_max.all()
        # maximize col0, minimize col1 → point 0 dominates? z=(p0*1, p1*-1):
        # p0: (1,-10), p1:(5,-1) → p1 dominates → front = [False, True]
        signs = torch.tensor([1.0, -1.0], dtype=DTYPE)
        assert pareto_front(points, signs).tolist() == [False, True]

    def test_all_equal_all_nondominated(self):
        points = torch.ones(6, 3, dtype=DTYPE)
        assert pareto_front(points, torch.ones(3, dtype=DTYPE)).all()

    def test_single_point_is_front(self):
        points = torch.tensor([[1.0, 2.0]], dtype=DTYPE)
        assert pareto_front(points, torch.ones(2, dtype=DTYPE)).tolist() == [True]


class TestScalarize:
    def test_weighted_choice(self):
        points = torch.tensor([[10.0, 0.0], [0.0, 10.0], [6.0, 6.0]], dtype=DTYPE)
        signs = torch.ones(2, dtype=DTYPE)
        w_first = torch.tensor([1.0, 0.0], dtype=DTYPE)
        w_even = torch.tensor([0.5, 0.5], dtype=DTYPE)
        assert int(scalarize(points, signs, w_first).argmax()) == 0
        assert int(scalarize(points, signs, w_even).argmax()) == 2


class TestEvaluate:
    def test_end_to_end_shapes(self, physics_model, field, crops):
        index = [crops.index(n) for n in ("maize", "soybean", "fallow")]
        plans = enumerate_rotations(index, 2)  # 9 plans
        ensemble = climatology_ensemble([synthetic_year(42.0, seed=s) for s in range(3)], 2)
        event = evaluate(physics_model, field, plans, ensemble)
        assert event.metrics.shape[:3] == (9, 3, 2)
        for v in event.objectives.values():
            assert v.shape == (9,) and torch.isfinite(v).all()

    def test_evaluate_deterministic(self, physics_model, field, crops):
        plans = enumerate_rotations([crops.index("maize")], 1)
        ensemble = climatology_ensemble([synthetic_year(42.0, seed=1)], 1)
        a = evaluate(physics_model, field, plans, ensemble)
        b = evaluate(physics_model, field, plans, ensemble)
        for k in a.objectives:
            assert torch.equal(a.objectives[k], b.objectives[k])

    def test_maize_beats_fallow_on_yield(self, physics_model, field, crops):
        plans = enumerate_rotations([crops.index("maize"), crops.index("fallow")], 1)
        ensemble = climatology_ensemble([synthetic_year(42.0, seed=1)], 1)
        event = evaluate(physics_model, field, plans, ensemble)
        mz, fa = event.objectives["yield_mean"]
        assert float(mz) > float(fa) >= 0.0

    def test_recommend_structure(self, physics_model, field, crops):
        index = [crops.index(n) for n in ("maize", "soybean", "wheat")]
        plans = enumerate_rotations(index, 2)
        ensemble = climatology_ensemble([synthetic_year(42.0, seed=s) for s in range(2)], 2)
        record = recommend(evaluate(physics_model, field, plans, ensemble), top_k=3)
        assert len(record["pareto_idx"]) >= 1
        assert len(record["ranking"]) == 3
        assert (record["scores"][record["ranking"][0]] >=
                record["scores"][record["ranking"][-1]])


class TestWorkers:
    def test_auto_workers_positive(self):
        from nasa_farm.hardware.capabilities import processor_workers
        assert processor_workers() >= 1

    def test_sharded_evaluate_bit_identical(self, physics_model, field, crops):
        """Thread-sharded rollout = serial rollout, bit for bit: rows are
        independent, contiguous chunks preserve order, no shared mutation."""
        index = [crops.index(n) for n in ("maize", "soybean", "wheat")]
        plans = enumerate_rotations(index, 2)  # 9 plans
        ensemble = climatology_ensemble(
            [synthetic_year(42.0, seed=s) for s in range(3)], 2)
        serial = evaluate(physics_model, field, plans, ensemble, workers=1,
                          device="cpu")
        sharded = evaluate(physics_model, field, plans, ensemble, workers=4,
                           device="cpu")
        assert torch.equal(serial.metrics, sharded.metrics)
        for k in serial.objectives:
            assert torch.equal(serial.objectives[k], sharded.objectives[k])

    def test_workers_rejects_batched_field(self, physics_model, field, crops):
        plans = enumerate_rotations([crops.index("maize"), crops.index("soybean")], 1)
        ensemble = climatology_ensemble([synthetic_year(42.0, seed=1)], 1)
        batched = field.repeat(2)  # batch == r*s → valid size, unshardable
        with pytest.raises(ValueError, match="workers"):
            evaluate(physics_model, batched, plans, ensemble, workers=2)

    def test_workers_rejects_wrong_batch(self, physics_model, field, crops):
        plans = enumerate_rotations([crops.index("maize"), crops.index("soybean")], 1)
        ensemble = climatology_ensemble([synthetic_year(42.0, seed=1)], 1)
        with pytest.raises(ValueError, match="matches neither"):
            evaluate(physics_model, field.repeat(3), plans, ensemble)


class TestAdaptiveHardware:
    def test_streamed_blocks_bit_identical(self, physics_model, field, crops):
        """mem_budget→0 forces plan-by-plan streaming; must equal one pass."""
        index = [crops.index(n) for n in ("maize", "soybean", "wheat")]
        plans = enumerate_rotations(index, 2)
        ensemble = climatology_ensemble(
            [synthetic_year(42.0, seed=s) for s in range(2)], 2)
        full = evaluate(physics_model, field, plans, ensemble,
                        workers=1, device="cpu", mem_budget=1.0)
        streamed = evaluate(physics_model, field, plans, ensemble,
                            workers=1, device="cpu", mem_budget=1e-9)
        assert torch.equal(full.metrics, streamed.metrics)

    def test_oom_ladder_retries_then_succeeds(self, physics_model, field, crops,
                                            monkeypatch):
        """A transient OOM halves the lane's pull and retries — never
        crashes, results bit-identical to an unperturbed run."""
        import nasa_farm.plan as plan_mod
        orig = plan_mod._rollout_metrics
        calls = {"n": 0}

        def flaky(*a, **keywords):
            calls["n"] += 1
            if calls["n"] == 1:
                raise MemoryError("simulated transient OOM")
            return orig(*a, **keywords)

        monkeypatch.setattr(plan_mod, "_rollout_metrics", flaky)
        index = [crops.index(n) for n in ("maize", "soybean", "wheat")]
        plans = enumerate_rotations(index, 2)
        ensemble = climatology_ensemble(
            [synthetic_year(42.0, seed=s) for s in range(2)], 2)
        event = evaluate(physics_model, field, plans, ensemble,
                      workers=1, device="cpu", mem_budget=1.0)
        reference = evaluate(physics_model, field, plans, ensemble,
                       workers=1, device="cpu", mem_budget=1.0)
        assert calls["n"] >= 2 and torch.equal(event.metrics, reference.metrics)

    def test_non_oom_errors_propagate(self, physics_model, field, crops,
                                     monkeypatch):
        """Bugs are never masked: non-OOM failures abort the sweep."""
        import nasa_farm.plan as plan_mod

        def broken(*a, **keywords):
            raise RuntimeError("a real bug, not OOM")

        monkeypatch.setattr(plan_mod, "_rollout_metrics", broken)
        plans = enumerate_rotations([crops.index("maize")], 1)
        ensemble = climatology_ensemble([synthetic_year(42.0, seed=1)], 1)
        with pytest.raises(RuntimeError, match="a real bug"):
            evaluate(physics_model, field, plans, ensemble, device="cpu")

    def test_probe_short_circuits_small_workloads(self, physics_model, field,
                                                  crops):
        """<500k field-days: probing costs more than the run — skipped."""
        from nasa_farm.hardware import probe
        plans = enumerate_rotations([crops.index("maize")], 1)
        ensemble = climatology_ensemble([synthetic_year(42.0, seed=1)], 1)
        r, s, days = plans.shape[0], ensemble.batch, ensemble.days
        assert not probe.worth(r, s, days)

    def test_physical_cores_detected(self):
        from nasa_farm.hardware.topology import physical_cores
        assert physical_cores() >= 1
