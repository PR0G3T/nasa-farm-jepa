"""EnKF assimilation — analysis statistics, online schedule, truth pull."""

import pytest
import torch

from nasa_farm.assimilation import Observation, assimilate, enkf_update
from nasa_farm.constants import DTYPE
from nasa_farm.forcing import synthetic_year, tile_to_horizon
from nasa_farm.state.field import Field, initial_state
from nasa_farm.types import Actions


class TestUpdate:
    def _with_sw0(self, s, v):
        sw = s.sw.clone()
        sw[:, 0] = v
        return s.replace(sw=sw)

    def test_analysis_shrinks_spread_and_recenters(self, soil):
        s = self._with_sw0(initial_state(16, soil),
                           torch.linspace(50.0, 90.0, 16, dtype=DTYPE))
        pred = s.sw[:, 0].unsqueeze(-1)           # observe sw0 directly
        g = torch.Generator().manual_seed(0)
        s2 = enkf_update(s, pred, torch.tensor([80.0], dtype=DTYPE), 1.0,
                         gen=g)
        assert float(s2.sw[:, 0].var()) < float(s.sw[:, 0].var())
        assert abs(float(s2.sw[:, 0].mean()) - 80.0) \
            < abs(float(s.sw[:, 0].mean()) - 80.0)

    def test_cross_covariance_updates_unobserved(self, soil):
        """Observing sw0 moves the root zone when the ensemble links them."""
        s = initial_state(16, soil)
        sw0 = torch.linspace(60.0, 100.0, 16, dtype=DTYPE)
        sw = s.sw.clone()
        sw[:, 0], sw[:, 1] = sw0, 4.0 * sw0 + 20.0
        s = s.replace(sw=sw)
        s2 = enkf_update(s, s.sw[:, 0].unsqueeze(-1),
                         torch.tensor([80.0], dtype=DTYPE), 1.0,
                         gen=torch.Generator().manual_seed(0))
        prior = float((s.sw[:, 1] - (4.0 * 80.0 + 20.0)).abs().mean())
        post = float((s2.sw[:, 1] - (4.0 * 80.0 + 20.0)).abs().mean())
        assert post < prior

    def test_states_stay_nonnegative(self, soil):
        s = self._with_sw0(initial_state(16, soil),
                           torch.full((16,), 5.0, dtype=DTYPE))
        s2 = enkf_update(s, s.sw[:, 0].unsqueeze(-1),
                         torch.tensor([300.0], dtype=DTYPE), 0.01,
                         gen=torch.Generator().manual_seed(0))
        assert (s2.sw >= 0).all() and (s2.n_min >= 0).all() \
            and (s2.lai >= 0).all() and (s2.biomass >= 0).all()

    def test_deterministic_given_seed(self, soil):
        s = initial_state(16, soil)
        pred = s.sw[:, 0].unsqueeze(-1)
        a = enkf_update(s, pred, torch.tensor([80.0]), 1.0,
                        gen=torch.Generator().manual_seed(3))
        b = enkf_update(s, pred, torch.tensor([80.0]), 1.0,
                        gen=torch.Generator().manual_seed(3))
        assert torch.equal(a.sw[:, 0], b.sw[:, 0])


class TestAssimilate:
    def _truth_obs(self, physics_model, soil, crops,
                   days=(200, 250, 300)):
        """Truth rollout (maize sown day 120) → vegetation-water-content
        measurements (biomass-proxy: stress-dependent and cumulative)."""
        from nasa_farm.observe import observables
        field = Field(initial_state(1, soil), soil,
                      torch.tensor([42.0], dtype=DTYPE))
        forcing = tile_to_horizon(synthetic_year(42.0, seed=1), 1)
        a = Actions.idle(1, forcing.days)
        sow = a.sow.clone()
        sow[0, 119] = crops.index("maize")
        rec = {}

        def obs(t, state, fx):
            if t in days:
                rec[t] = float(observables(state, ("vwc",))[0, 0])
        physics_model.rollout(
            field, forcing,
            Actions(sow, a.irrigation, a.n_fert), observer=obs)
        return forcing, [Observation(t, ("vwc",),
                                     torch.tensor([v], dtype=DTYPE), 0.05)
                         for t, v in rec.items()]

    def test_pulls_ensemble_toward_truth(self, physics_model, soil, crops):
        """Same biased start: assimilated yield ends closer to the truth
        than the free-running biased ensemble (cumulative channels keep
        mid-season errors alive — the EnKF use case)."""
        forcing, schedule = self._truth_obs(physics_model, soil, crops)
        a = Actions.idle(1, forcing.days)
        sow = a.sow.clone()
        sow[0, 119] = crops.index("maize")
        truth = physics_model.rollout(
            Field(initial_state(1, soil), soil,
                  torch.tensor([42.0], dtype=DTYPE)), forcing,
            Actions(sow, a.irrigation, a.n_fert))
        biased = initial_state(16, soil)
        bsw = biased.sw.clone()
        bsw[:, 0] = (initial_state(1, soil).sw[0, 0] - 25.0
                     + torch.linspace(-8.0, 8.0, 16, dtype=DTYPE))
        biased = biased.replace(sw=bsw)
        acts = Actions(sow.repeat(16, 1), a.irrigation.repeat(16, 1),
                       a.n_fert.repeat(16, 1))
        free = physics_model.rollout(
            Field(biased.clone(), soil,
                  torch.tensor([42.0] * 16, dtype=DTYPE)), forcing, acts)
        out = assimilate(physics_model,
                         Field(biased, soil,
                               torch.tensor([42.0] * 16, dtype=DTYPE)),
                         forcing, acts, schedule, seed=0)
        err_free = float((free.state.cum_yield.mean()
                          - truth.state.cum_yield).abs())
        err_post = float((out.state.cum_yield.mean()
                          - truth.state.cum_yield).abs())
        assert err_post < err_free

    def test_full_horizon_simulated(self, physics_model, soil, crops):
        """Every forcing day is stepped exactly once around analyses:
        gdd (forcing-only, never EnKF-updated) matches the plain rollout
        bitwise, while the analysis measurably moved the water state."""
        forcing, schedule = self._truth_obs(physics_model, soil, crops,
                                            days=(200,))
        a = Actions.idle(4, forcing.days)
        sow = a.sow.clone()
        sow[:, 119] = crops.index("maize")
        actions = Actions(sow, a.irrigation, a.n_fert)
        start = initial_state(4, soil).replace(
            n_min=torch.linspace(20.0, 60.0, 4, dtype=DTYPE))
        field = Field(start, soil, torch.tensor([42.0] * 4, dtype=DTYPE))
        ref = physics_model.rollout(
            Field(start.clone(), soil, torch.tensor([42.0] * 4, dtype=DTYPE)),
            forcing, actions)
        out = assimilate(physics_model, field, forcing, actions,
                         schedule, seed=0)
        assert torch.equal(out.state.gdd, ref.state.gdd)
        assert not torch.equal(out.state.cum_yield, ref.state.cum_yield)

    def test_unknown_channel_raises(self):
        with pytest.raises(KeyError):
            Observation(10, ("bogus",), torch.tensor([0.0]), 0.1)
