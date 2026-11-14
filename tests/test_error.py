"""Observation-error machinery — DBCP identity, AOEI, full-R EnKF."""

import pytest
import torch

from nasa_farm.assim import Observation, assimilate, enkf_update
from nasa_farm.assim.err import RStat, aoei
from nasa_farm.constants import DTYPE


class TestRStat:
    def test_dbcp_recovers_true_r(self):
        """Linear-Gaussian cycle: E[(y−x_b)(y−x_a)] = R exactly.
        bg = x + b, y = x + r, analysis BLUE with K = B/(B+R) gives
        d_a = (R/(B+R))·d_b ⇒ E[d_b·d_a] = R.
        """
        g = torch.Generator().manual_seed(0)
        b_var, r_true = 4.0, 2.0
        n = 60_000
        b = torch.randn(2, n, generator=g, dtype=DTYPE) * b_var ** 0.5
        r = torch.randn(2, n, generator=g, dtype=DTYPE) * r_true ** 0.5
        d_bg = r - b
        d_an = d_bg * r_true / (b_var + r_true)
        rs = RStat()
        for i in range(n):
            rs.update(d_bg[:, i], d_an[:, i])
        est = rs.estimate
        assert float(est.diagonal().mean()) == pytest.approx(r_true,
                                                           rel=0.05)
        assert est.shape == (2, 2)

    def test_decay_forgets_old(self):
        rs = RStat(decay=0.5)
        for v in (torch.tensor([4.0]), torch.tensor([1.0])):
            rs.update(v, v)
        e = rs.estimate
        # weighted mean of 16 and 1 with the recent sample weighting
        # double: (0.5·16 + 1)/1.5 = 6 — below the unweighted midpoint.
        assert 1.0 < float(e[0, 0]) < 8.5
        assert float(e[0, 0]) == pytest.approx(6.0)

    def test_var_is_diagonal(self):
        rs = RStat()
        rs.update(torch.tensor([2.0, 0.0]), torch.tensor([1.0, 0.0]))
        assert rs.var.tolist() == pytest.approx([2.0, 0.0], abs=1e-12)


class TestAoei:
    def test_small_innovation_keeps_r(self):
        r = torch.tensor([1.0, 1.0], dtype=DTYPE)
        out = aoei(torch.tensor([0.5, 0.8], dtype=DTYPE),
                   torch.tensor([0.3, 0.3], dtype=DTYPE), r)
        assert torch.equal(out, r)

    def test_large_innovation_inflates(self):
        out = aoei(torch.tensor([4.0], dtype=DTYPE),
                   torch.tensor([1.0], dtype=DTYPE), 2.0)
        assert float(out[0]) == pytest.approx(15.0)   # d² − var_ens

    def test_monotone_in_innovation(self):
        a = aoei(torch.tensor([1.0]), torch.tensor([0.5]), 1.0)
        b = aoei(torch.tensor([5.0]), torch.tensor([0.5]), 1.0)
        assert float(b[0]) >= float(a[0])


class TestFullR:
    def test_whitening_equivalence(self, soil):
        """Analysis with correlated R equals the whitened problem:
        L⁻¹whitening of obs+pred and R=I gives the same update."""
        from nasa_farm.state.field import initial_state
        s = initial_state(16, soil)
        pred = torch.stack([s.sw[:, 0],
                            s.sw[:, 0] * 0.5 + 1.0], -1)
        r = torch.tensor([[2.0, 1.2], [1.2, 2.0]], dtype=DTYPE)
        obs = torch.tensor([80.0, 45.0], dtype=DTYPE)
        l = torch.linalg.cholesky(r)
        a = enkf_update(s, pred, obs, r, perturb=False)
        b = enkf_update(s, torch.linalg.solve(l, pred.t()).t(),
                        torch.linalg.solve(l, obs),
                        torch.eye(2, dtype=DTYPE), perturb=False)
        assert torch.allclose(a.sw, b.sw, atol=1e-10)
        assert torch.allclose(a.lai, b.lai, atol=1e-10)

    def test_correlated_noise_shape(self, soil):
        """Full R path runs the stochastic update and stays in bounds."""
        from nasa_farm.state.field import initial_state
        s = initial_state(16, soil)
        pred = torch.stack([s.sw[:, 0], s.lai], -1)
        r = torch.tensor([[4.0, 1.0], [1.0, 0.5]], dtype=DTYPE)
        s2 = enkf_update(s, pred, torch.tensor([70.0, 3.0], dtype=DTYPE),
                         r, gen=torch.Generator().manual_seed(0))
        assert (s2.sw >= 0).all() and (s2.lai >= 0).all()

    def test_r_stat_fills_during_assimilate(self, physics_model, soil,
                                            crops):
        """The driver accumulates the diagnostic at each analysis day."""
        from nasa_farm.forcing import synthetic_year, tile_to_horizon
        from nasa_farm.state.field import Field, initial_state
        from nasa_farm.types import Actions
        field = Field(initial_state(8, soil), soil,
                      torch.tensor([42.0] * 8, dtype=DTYPE))
        forcing = tile_to_horizon(synthetic_year(42.0, seed=1), 1)
        a = Actions.idle(8, forcing.days)
        sow = a.sow.clone()
        sow[:, 119] = crops.index("maize")
        acts = Actions(sow, a.irrigation, a.n_fert)
        rs = RStat()
        sched = [Observation(200, ("vwc",), torch.tensor([0.4], dtype=DTYPE),
                            0.05),
                 Observation(250, ("vwc",), torch.tensor([0.6], dtype=DTYPE),
                             0.05)]
        assimilate(physics_model, field, forcing, acts, sched,
                   r_stat=rs, seed=0)
        assert rs.estimate is not None and rs.estimate.shape == (1, 1)
        assert float(rs.estimate[0, 0]) >= 0.0
