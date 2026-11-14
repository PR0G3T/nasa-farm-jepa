"""Interpretable heads — aggregation semantics and objective folding."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.heads import METRICS, OBJECTIVE_DIRECTION, SeasonAccumulator, objectives
from nasa_farm.state.field import initial_state
from nasa_farm.state.soil import SoilProfile


def _state(batch: int, **keywords) -> object:
    soil = SoilProfile.preset("loam", batch=batch)
    s = initial_state(batch, soil)
    return s.replace(**keywords)


def _fluxes(batch: int, **keywords) -> dict:
    z = torch.zeros(batch, dtype=DTYPE)
    base = {"yield_event": z, "es": z, "t": torch.zeros(batch, 2, dtype=DTYPE),
            "runoff": z, "deep": z, "leached": z, "n2o": z, "heat_dd": z,
            "frost": z, "water_fac": torch.ones(batch, dtype=DTYPE)}
    return {**base, **keywords}


class TestAccumulator:
    def test_aggregates_into_years(self):
        accumulate = SeasonAccumulator(1, 2)
        growing = torch.tensor([True])
        for t in range(365):
            s = _state(1, growing=growing)
            accumulate(t, s, _fluxes(1, yield_event=torch.tensor([10.0]),
                              es=torch.tensor([1.0]), water_fac=torch.tensor([1.0])))
        for t in range(365, 730):
            s = _state(1, growing=torch.tensor([False]))
            accumulate(t, s, _fluxes(1))
        m = accumulate.result()
        i = {n: k for k, n in enumerate(METRICS)}
        assert m.shape == (1, 2, len(METRICS))
        assert float(m[0, 0, i["yield_kg"]]) == pytest.approx(3650.0)
        assert float(m[0, 0, i["cover_days"]]) == pytest.approx(365.0)
        assert float(m[0, 1, i["cover_days"]]) == 0.0
        assert float(m[0, 0, i["water_used"]]) == pytest.approx(365.0)

    def test_stress_days_only_when_growing(self):
        accumulate = SeasonAccumulator(1, 1)
        accumulate(0, _state(1, growing=torch.tensor([True])),
            _fluxes(1, water_fac=torch.tensor([0.2])))
        accumulate(1, _state(1, growing=torch.tensor([False])),
            _fluxes(1, water_fac=torch.tensor([0.2])))
        i = {n: k for k, n in enumerate(METRICS)}
        assert float(accumulate.result()[0, 0, i["stress_days"]]) == 1.0


class TestObjectives:
    def test_folds_plan_scenario(self):
        r, s, h = 4, 3, 2
        m = torch.zeros(r * s, h, len(METRICS), dtype=DTYPE)
        i = {n: k for k, n in enumerate(METRICS)}
        m[..., i["yield_kg"]] = 100.0
        m[..., i["soc_end"]] = torch.linspace(0, 1, r * s).unsqueeze(1).expand(r * s, h)
        object = objectives(m, r, s)
        assert set(object) == set(OBJECTIVE_DIRECTION)
        for v in object.values():
            assert v.shape == (r,) and torch.isfinite(v).all()
        assert float(object["yield_mean"][0]) == pytest.approx(200.0)

    def test_risk_quantile_below_mean(self):
        r, s, h = 1, 100, 1
        m = torch.zeros(s, h, len(METRICS), dtype=DTYPE)
        i = {n: k for k, n in enumerate(METRICS)}
        m[:, 0, i["yield_kg"]] = torch.linspace(0, 1000, s, dtype=DTYPE)
        object = objectives(m, r, s)
        assert float(object["yield_risk"][0]) < float(object["yield_mean"][0])

    def test_soc_gain_signed(self):
        r, s, h = 1, 2, 3
        m = torch.zeros(r * s, h, len(METRICS), dtype=DTYPE)
        i = {n: k for k, n in enumerate(METRICS)}
        m[:, 0, i["soc_end"]] = 100.0
        m[:, -1, i["soc_end"]] = 160.0
        object = objectives(m, r, s)
        assert float(object["soc_gain"][0]) == pytest.approx(60.0)
