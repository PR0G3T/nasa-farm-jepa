"""Interpretable heads — aggregation semantics and objective folding."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.heads import METRICS, OBJECTIVE_DIRECTION, SeasonAccumulator, objectives
from nasa_farm.state.field import initial_state
from nasa_farm.state.soil import SoilProfile


def _state(batch: int, **kw) -> object:
    soil = SoilProfile.preset("loam", batch=batch)
    s = initial_state(batch, soil)
    return s.replace(**kw)


def _fluxes(batch: int, **kw) -> dict:
    z = torch.zeros(batch, dtype=DTYPE)
    base = {"yield_event": z, "es": z, "t1": z, "t2": z, "runoff": z,
            "deep": z, "leached": z, "water_fac": torch.ones(batch, dtype=DTYPE)}
    return {**base, **kw}


class TestAccumulator:
    def test_aggregates_into_years(self):
        acc = SeasonAccumulator(1, 2)
        growing = torch.tensor([True])
        for t in range(365):
            s = _state(1, growing=growing)
            acc(t, s, _fluxes(1, yield_event=torch.tensor([10.0]),
                              es=torch.tensor([1.0]), water_fac=torch.tensor([1.0])))
        for t in range(365, 730):
            s = _state(1, growing=torch.tensor([False]))
            acc(t, s, _fluxes(1))
        m = acc.result()
        i = {n: k for k, n in enumerate(METRICS)}
        assert m.shape == (1, 2, len(METRICS))
        assert float(m[0, 0, i["yield_kg"]]) == pytest.approx(3650.0)
        assert float(m[0, 0, i["cover_days"]]) == pytest.approx(365.0)
        assert float(m[0, 1, i["cover_days"]]) == 0.0
        assert float(m[0, 0, i["water_used"]]) == pytest.approx(365.0)

    def test_stress_days_only_when_growing(self):
        acc = SeasonAccumulator(1, 1)
        acc(0, _state(1, growing=torch.tensor([True])),
            _fluxes(1, water_fac=torch.tensor([0.2])))
        acc(1, _state(1, growing=torch.tensor([False])),
            _fluxes(1, water_fac=torch.tensor([0.2])))
        i = {n: k for k, n in enumerate(METRICS)}
        assert float(acc.result()[0, 0, i["stress_days"]]) == 1.0


class TestObjectives:
    def test_folds_plan_scenario(self):
        r, s, h = 4, 3, 2
        m = torch.zeros(r * s, h, len(METRICS), dtype=DTYPE)
        i = {n: k for k, n in enumerate(METRICS)}
        m[..., i["yield_kg"]] = 100.0
        m[..., i["soc_end"]] = torch.linspace(0, 1, r * s).unsqueeze(1).expand(r * s, h)
        obj = objectives(m, r, s)
        assert set(obj) == set(OBJECTIVE_DIRECTION)
        for v in obj.values():
            assert v.shape == (r,) and torch.isfinite(v).all()
        assert float(obj["yield_mean"][0]) == pytest.approx(200.0)

    def test_risk_quantile_below_mean(self):
        r, s, h = 1, 100, 1
        m = torch.zeros(s, h, len(METRICS), dtype=DTYPE)
        i = {n: k for k, n in enumerate(METRICS)}
        m[:, 0, i["yield_kg"]] = torch.linspace(0, 1000, s, dtype=DTYPE)
        obj = objectives(m, r, s)
        assert float(obj["yield_risk"][0]) < float(obj["yield_mean"][0])

    def test_soc_gain_signed(self):
        r, s, h = 1, 2, 3
        m = torch.zeros(r * s, h, len(METRICS), dtype=DTYPE)
        i = {n: k for k, n in enumerate(METRICS)}
        m[:, 0, i["soc_end"]] = 100.0
        m[:, -1, i["soc_end"]] = 160.0
        obj = objectives(m, r, s)
        assert float(obj["soc_gain"][0]) == pytest.approx(60.0)
