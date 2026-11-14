"""Autonomy — drift detection, champion-challenger gate, realizable
subgoals, SBC calibration and goal-conditioned inverse dynamics."""

import pytest
import torch
import torch.nn as nn

from nasa_farm.constants import DTYPE
from nasa_farm.forcing import synthetic_year
from nasa_farm.types import Actions

G = torch.Generator().manual_seed(0)


class TestDrift:
    def test_psi_stable(self):
        a = torch.randn(400, 5, generator=G, dtype=DTYPE)
        b = a + 0.01 * torch.randn(400, 5, generator=G, dtype=DTYPE)
        from nasa_farm.observe.drift import psi
        assert float(psi(a, b).max()) < 0.25

    def test_psi_shift(self):
        from nasa_farm.observe.drift import psi
        a = torch.randn(400, 3, generator=G, dtype=DTYPE)
        b = a + 3.0
        assert float(psi(a, b).min()) > 0.25

    def test_energy_zero_same(self):
        from nasa_farm.observe.drift import energy
        a = torch.randn(300, 4, generator=G, dtype=DTYPE)
        assert float(energy(a, a)) == pytest.approx(0.0, abs=1e-9)

    def test_energy_positive_shift(self):
        from nasa_farm.observe.drift import energy
        a = torch.randn(300, 4, generator=G, dtype=DTYPE)
        assert float(energy(a, a + 2.0)) > 1.0

    def test_drifted(self):
        from nasa_farm.observe.drift import drifted
        a = torch.randn(300, 3, generator=G, dtype=DTYPE)
        assert not drifted(a, a + 0.01 * torch.randn(
            300, 3, generator=G, dtype=DTYPE))
        assert drifted(a, a + 4.0)


class TestGate:
    def test_promote_better(self):
        from nasa_farm.optimize.gate import gate
        m = nn.Linear(2, 1, dtype=DTYPE)
        score = {"v": 1.0}
        rep = gate(m, lambda: score.update(v=0.5), lambda: score["v"])
        assert rep == {"promoted": True, "champion": 1.0, "challenger": 0.5}

    def test_reject_worse_restores(self):
        from nasa_farm.optimize.gate import gate
        m = nn.Linear(2, 1, dtype=DTYPE)
        before = {k: v.clone() for k, v in m.state_dict().items()}
        score = {"v": 1.0}
        def bad_fit():
            score["v"] = 5.0
            with torch.no_grad():
                m.weight.fill_(99.0)
        rep = gate(m, bad_fit, lambda: score["v"])
        assert not rep["promoted"]
        for k, v in m.state_dict().items():
            assert torch.equal(v, before[k])


class TestLibrary:
    def test_nearest_returns_realized(self, physics_model, field, crops):
        from nasa_farm.retrieve import StateLibrary, record
        forcing = synthetic_year(42.0, seed=3)
        sow = torch.zeros(1, 365, dtype=torch.long)
        sow[:, 100] = crops.index("maize")
        acts = Actions(sow=sow, irrigation=torch.zeros(1, 365, dtype=DTYPE),
                       n_fert=torch.zeros(1, 365, dtype=DTYPE))
        lib = StateLibrary()
        record(lib, physics_model, field, forcing, acts, every=30)
        assert len(lib) > 0
        goal = lib.rows[0]
        out = lib.nearest(goal)
        # exact retrieval: a stored state is its own nearest neighbor
        assert torch.equal(out[0], goal)
        # every returned row is realized by construction
        assert (out[0] == lib.rows).all(-1).any()

    def test_empty_library_raises(self):
        from nasa_farm.retrieve import StateLibrary
        with pytest.raises(ValueError):
            StateLibrary().nearest(torch.zeros(4, dtype=DTYPE))


class TestSbc:
    def test_ranks_uniform_gaussian(self):
        """Conjugate Gaussian: trained posterior must produce ~uniform
        ranks — the SBC calibration certificate."""
        from nasa_farm.optimize.sbi import fit_posterior, sbc_ranks
        g = torch.Generator().manual_seed(1)
        n = 400
        theta = torch.randn(n, 2, generator=g, dtype=DTYPE)
        x = theta + 0.5 * torch.randn(n, 2, generator=g, dtype=DTYPE)
        post = fit_posterior(theta, x, epochs=120, hidden=64, seed=0)
        g2 = torch.Generator().manual_seed(9)
        m = 32
        th = torch.randn(m, 2, generator=g2, dtype=DTYPE)
        xo = th + 0.5 * torch.randn(m, 2, generator=g2, dtype=DTYPE)
        ranks = sbc_ranks(post, th, xo, n=32, seed=0)
        assert ranks.shape == (m, 2)
        u = ranks.to(DTYPE) / 32.0
        assert 0.3 < float(u.mean()) < 0.7
        # degenerate posteriors pile ranks at 0 or n — reject that
        assert 0.2 < float(((u > 0.25) & (u < 0.75)).to(DTYPE).mean()) < 0.9


class TestIdm:
    def _demo(self, field, irr, crops, seed):
        from nasa_farm.forcing import span
        days = 60
        forcing = span(synthetic_year(42.0, seed=seed), 0, days)
        sow = torch.zeros(1, days, dtype=torch.long)
        sow[:, 0] = crops.index("maize")
        return field, forcing, Actions(
            sow=sow, irrigation=torch.full((1, days), irr, dtype=DTYPE),
            n_fert=torch.zeros(1, days, dtype=DTYPE))

    def test_goal_selectivity(self, physics_model, field, crops):
        """Two demos identical except the irrigation rate: on the *same*
        state the trained net must answer the goal-specific action —
        conditioning on the realized outcome is what makes it inverse
        dynamics rather than imitation."""
        from nasa_farm.policy import _day_feats, fit_idm
        gfn = lambda s: s.sw.sum(-1, keepdim=True)  # terminal soil water
        demos = [self._demo(field, 0.0, crops, 3),
                 self._demo(field, 10.0, crops, 3)]
        idm = fit_idm(physics_model, demos, goal_fn=gfn,
                      epochs=300, seed=0)
        fld, frc, acts = demos[0]
        for t in range(frc.days):
            with torch.inference_mode():
                fld, _ = physics_model.step(fld, acts.day(t), frc.day(t), t)
        g_dry = gfn(fld.state)
        fld, frc, acts = demos[1]
        for t in range(frc.days):
            with torch.inference_mode():
                fld, _ = physics_model.step(fld, acts.day(t), frc.day(t), t)
        g_wet = gfn(fld.state)
        assert float(g_wet - g_dry) > 0.0
        f0 = _day_feats(physics_model, field.state,
                        demos[0][1].day(0), demos[0][2].day(0))
        rem = torch.ones(1, 1, dtype=DTYPE)
        a_dry = float(idm(f0, g_dry, rem).detach()[0, 0])
        a_wet = float(idm(f0, g_wet, rem).detach()[0, 0])
        # same state, different goal → the wet-goal action is larger and
        # tracks its teacher (10 mm) better than the dry one does
        assert a_wet > a_dry
        assert abs(a_wet - 10.0) < abs(a_dry - 10.0)

    def test_idm_actions_shape(self, physics_model, field, crops):
        from nasa_farm.policy import fit_idm, idm_actions
        f_, frc, acts = self._demo(field, 5.0, crops, 3)
        idm = fit_idm(physics_model, [(f_, frc, acts)], epochs=5, seed=0)
        out = idm_actions(idm, physics_model, f_, frc, acts,
                          torch.zeros(idm.goal_dim, dtype=DTYPE))
        assert out.irrigation.shape == acts.irrigation.shape
        assert (out.irrigation >= 0).all()
