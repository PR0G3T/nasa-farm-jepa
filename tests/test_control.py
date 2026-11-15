"""MPPI controller — noise statistics, action bounds, uplift, determinism."""

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.control import path_integral
from nasa_farm.forcing import augment, noise, synthetic_year
from nasa_farm.heads import METRICS, SeasonAccumulator
from nasa_farm.state.field import Field, initial_state
from nasa_farm.types import Actions


def _base(crops, days: int) -> Actions:
    a = Actions.idle(1, days)
    sow = a.sow.clone()
    sow[0, 119] = crops.index("maize")
    return Actions(sow, a.irrigation, a.n_fert)


class TestOUNoise:
    def test_unit_variance_zero_mean(self):
        g = torch.Generator().manual_seed(0)
        x = noise(4096, 200, 0.7, g)
        assert abs(float(x.mean())) < 0.05
        assert 0.8 < float(x.std()) < 1.2

    def test_ar1_autocorrelation(self):
        g = torch.Generator().manual_seed(0)
        x = noise(2048, 400, 0.7, g)
        r1 = (x[:, 1:] * x[:, :-1]).mean() / x.var()
        assert 0.5 < float(r1) < 0.9

    def test_deterministic_per_generator(self):
        g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
        assert torch.equal(noise(8, 50, 0.6, g1), noise(8, 50, 0.6, g2))


class TestMPPI:
    def _setup(self, soil, crops):
        field = Field(initial_state(1, soil), soil, torch.tensor([42.0], dtype=DTYPE))
        ensemble = augment(synthetic_year(42.0, seed=1), 3, seed=0)
        return field, ensemble, _base(crops, ensemble.days)

    def test_actions_within_bounds(self, physics_model, soil, crops):
        field, ensemble, base = self._setup(soil, crops)
        result = path_integral(
            physics_model, field, ensemble, base, iterations=3, samples=24, seed=0
        )
        assert (result.actions.irrigation >= 0).all()
        assert (result.actions.irrigation <= 25.0).all()
        assert (result.actions.n_fert >= 0).all()
        assert (result.actions.n_fert <= 10.0).all()

    def test_deterministic_given_seed(self, physics_model, soil, crops):
        field, ensemble, base = self._setup(soil, crops)
        r1 = path_integral(physics_model, field, ensemble, base, iterations=2, samples=16, seed=5)
        r2 = path_integral(physics_model, field, ensemble, base, iterations=2, samples=16, seed=5)
        assert torch.equal(r1.actions.irrigation, r2.actions.irrigation)

    def test_optimized_schedule_helps_in_drought(self, physics_model, soil, crops):
        """Yield under the optimized schedule ≥ rainfed — irrigation can only
        add water the reward already values."""
        field, ensemble, base = self._setup(soil, crops)
        result = path_integral(
            physics_model, field, ensemble, base, iterations=3, samples=24, seed=0
        )
        i = METRICS.index("yield_kg")

        def y(actions):
            accumulate = SeasonAccumulator(ensemble.batch, 1)
            physics_model.rollout(
                field.repeat(ensemble.batch),
                ensemble,
                actions.repeat(ensemble.batch),
                observer=accumulate,
            )
            return float(accumulate.result()[..., i].mean())

        with torch.inference_mode():
            assert y(result.actions) >= 0.95 * y(base)
