"""Planning — enumeration, action expansion, Pareto exactness, ranking."""

import pytest
import torch

from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.forcing import climatology_ensemble, synthetic_year
from nasa_farm.plan import (
    enumerate_rotations, evaluate, expand_actions, objective_matrix,
    pareto_front, recommend, scalarize,
)
from nasa_farm.state.crops import IDX


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
        maize_doy = int(crops.get("maize")[IDX["sow_doy"]])
        soy_doy = int(crops.get("soybean")[IDX["sow_doy"]])
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
        pts = torch.tensor([[1.0, 1.0], [2.0, 2.0], [1.5, 1.0], [0.5, 0.5]],
                           dtype=DTYPE)
        signs = torch.ones(2, dtype=DTYPE)  # maximize both
        mask = pareto_front(pts, signs)
        assert mask.tolist() == [False, True, False, False]

    def test_knee_points_kept(self):
        """Trade-off extremes survive; the middle dominated point drops."""
        pts = torch.tensor([[2.0, 0.5], [1.5, 1.5], [0.5, 2.0], [1.0, 1.0]],
                           dtype=DTYPE)
        mask = pareto_front(pts, torch.ones(2, dtype=DTYPE))
        assert mask.tolist() == [True, True, True, False]

    def test_signs_flip_direction(self):
        pts = torch.tensor([[1.0, 10.0], [5.0, 1.0]], dtype=DTYPE)
        both_max = pareto_front(pts, torch.ones(2, dtype=DTYPE))
        assert both_max.all()
        # maximize col0, minimize col1 → point 0 dominates? z=(p0*1, p1*-1):
        # p0: (1,-10), p1:(5,-1) → p1 dominates → front = [False, True]
        signs = torch.tensor([1.0, -1.0], dtype=DTYPE)
        assert pareto_front(pts, signs).tolist() == [False, True]

    def test_all_equal_all_nondominated(self):
        pts = torch.ones(6, 3, dtype=DTYPE)
        assert pareto_front(pts, torch.ones(3, dtype=DTYPE)).all()

    def test_single_point_is_front(self):
        pts = torch.tensor([[1.0, 2.0]], dtype=DTYPE)
        assert pareto_front(pts, torch.ones(2, dtype=DTYPE)).tolist() == [True]


class TestScalarize:
    def test_weighted_choice(self):
        pts = torch.tensor([[10.0, 0.0], [0.0, 10.0], [6.0, 6.0]], dtype=DTYPE)
        signs = torch.ones(2, dtype=DTYPE)
        w_first = torch.tensor([1.0, 0.0], dtype=DTYPE)
        w_even = torch.tensor([0.5, 0.5], dtype=DTYPE)
        assert int(scalarize(pts, signs, w_first).argmax()) == 0
        assert int(scalarize(pts, signs, w_even).argmax()) == 2


class TestEvaluate:
    def test_end_to_end_shapes(self, physics_model, field, crops):
        idx = [crops.index(n) for n in ("maize", "soybean", "fallow")]
        plans = enumerate_rotations(idx, 2)  # 9 plans
        ens = climatology_ensemble([synthetic_year(42.0, seed=s) for s in range(3)], 2)
        ev = evaluate(physics_model, field, plans, ens)
        assert ev.metrics.shape[:3] == (9, 3, 2)
        for v in ev.objectives.values():
            assert v.shape == (9,) and torch.isfinite(v).all()

    def test_evaluate_deterministic(self, physics_model, field, crops):
        plans = enumerate_rotations([crops.index("maize")], 1)
        ens = climatology_ensemble([synthetic_year(42.0, seed=1)], 1)
        a = evaluate(physics_model, field, plans, ens)
        b = evaluate(physics_model, field, plans, ens)
        for k in a.objectives:
            assert torch.equal(a.objectives[k], b.objectives[k])

    def test_maize_beats_fallow_on_yield(self, physics_model, field, crops):
        plans = enumerate_rotations([crops.index("maize"), crops.index("fallow")], 1)
        ens = climatology_ensemble([synthetic_year(42.0, seed=1)], 1)
        ev = evaluate(physics_model, field, plans, ens)
        mz, fa = ev.objectives["yield_mean"]
        assert float(mz) > float(fa) >= 0.0

    def test_recommend_structure(self, physics_model, field, crops):
        idx = [crops.index(n) for n in ("maize", "soybean", "wheat")]
        plans = enumerate_rotations(idx, 2)
        ens = climatology_ensemble([synthetic_year(42.0, seed=s) for s in range(2)], 2)
        rec = recommend(evaluate(physics_model, field, plans, ens), top_k=3)
        assert len(rec["pareto_idx"]) >= 1
        assert len(rec["ranking"]) == 3
        assert (rec["scores"][rec["ranking"][0]] >=
                rec["scores"][rec["ranking"][-1]])


class TestWorkers:
    def test_auto_workers_positive(self):
        from nasa_farm.plan import auto_workers
        assert auto_workers() >= 1

    def test_sharded_evaluate_bit_identical(self, physics_model, field, crops):
        """Thread-sharded rollout = serial rollout, bit for bit: rows are
        independent, contiguous chunks preserve order, no shared mutation."""
        idx = [crops.index(n) for n in ("maize", "soybean", "wheat")]
        plans = enumerate_rotations(idx, 2)  # 9 plans
        ens = climatology_ensemble(
            [synthetic_year(42.0, seed=s) for s in range(3)], 2)
        serial = evaluate(physics_model, field, plans, ens, workers=1,
                          device="cpu")
        sharded = evaluate(physics_model, field, plans, ens, workers=4,
                           device="cpu")
        assert torch.equal(serial.metrics, sharded.metrics)
        for k in serial.objectives:
            assert torch.equal(serial.objectives[k], sharded.objectives[k])

    def test_workers_rejects_batched_field(self, physics_model, field, crops):
        plans = enumerate_rotations([crops.index("maize"), crops.index("soybean")], 1)
        ens = climatology_ensemble([synthetic_year(42.0, seed=1)], 1)
        batched = field.repeat(2)  # batch == r*s → valid size, unshardable
        with pytest.raises(ValueError, match="workers"):
            evaluate(physics_model, batched, plans, ens, workers=2)

    def test_workers_rejects_wrong_batch(self, physics_model, field, crops):
        plans = enumerate_rotations([crops.index("maize"), crops.index("soybean")], 1)
        ens = climatology_ensemble([synthetic_year(42.0, seed=1)], 1)
        with pytest.raises(ValueError, match="matches neither"):
            evaluate(physics_model, field.repeat(3), plans, ens)


class TestAdaptiveHardware:
    def test_streamed_blocks_bit_identical(self, physics_model, field, crops):
        """mem_budget→0 forces plan-by-plan streaming; must equal one pass."""
        idx = [crops.index(n) for n in ("maize", "soybean", "wheat")]
        plans = enumerate_rotations(idx, 2)
        ens = climatology_ensemble(
            [synthetic_year(42.0, seed=s) for s in range(2)], 2)
        full = evaluate(physics_model, field, plans, ens,
                        workers=1, device="cpu", mem_budget=1.0)
        streamed = evaluate(physics_model, field, plans, ens,
                            workers=1, device="cpu", mem_budget=1e-9)
        assert torch.equal(full.metrics, streamed.metrics)

    def test_oom_ladder_retries_then_succeeds(self, physics_model, field, crops,
                                            monkeypatch):
        """A transient OOM halves the block and retries — never crashes."""
        import nasa_farm.plan as plan_mod
        orig = plan_mod._eval_block
        calls = {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise MemoryError("simulated transient OOM")
            return orig(*a, **kw)

        monkeypatch.setattr(plan_mod, "_eval_block", flaky)
        idx = [crops.index(n) for n in ("maize", "soybean", "wheat")]
        plans = enumerate_rotations(idx, 2)
        ens = climatology_ensemble(
            [synthetic_year(42.0, seed=s) for s in range(2)], 2)
        ev = evaluate(physics_model, field, plans, ens,
                      workers=1, device="cpu", mem_budget=1.0)
        ref = evaluate(physics_model, field, plans, ens,
                       workers=1, device="cpu", mem_budget=1.0)
        assert calls["n"] >= 2 and torch.equal(ev.metrics, ref.metrics)

    def test_non_oom_errors_propagate(self, physics_model, field, crops,
                                     monkeypatch):
        """Bugs are never masked: non-OOM failures raise immediately."""
        import nasa_farm.plan as plan_mod

        def broken(*a, **kw):
            raise RuntimeError("a real bug, not OOM")

        monkeypatch.setattr(plan_mod, "_eval_block", broken)
        plans = enumerate_rotations([crops.index("maize")], 1)
        ens = climatology_ensemble([synthetic_year(42.0, seed=1)], 1)
        with pytest.raises(RuntimeError, match="a real bug"):
            evaluate(physics_model, field, plans, ens, device="cpu")

    def test_probe_short_circuits_small_workloads(self, physics_model, field,
                                                  crops):
        """<500k field-days: CPU directly, zero probe cost — deterministic."""
        import nasa_farm.plan as plan_mod
        plans = enumerate_rotations([crops.index("maize")], 1)
        ens = climatology_ensemble([synthetic_year(42.0, seed=1)], 1)
        dev = plan_mod._pick_device(physics_model, field, plans, ens, 1)
        assert dev.type == "cpu"

    def test_physical_cores_detected(self):
        from nasa_farm.plan import _physical_cores
        assert _physical_cores() >= 1
