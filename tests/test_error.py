"""Observation-error machinery — DBCP identity, AOEI, full-R EnKF."""

import pytest
import torch

from nasa_farm.assimilation import Observation, assimilate, kalman_update
from nasa_farm.assimilation.error import Statistic, inflate
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
        statistic = Statistic()
        for i in range(n):
            statistic.update(d_bg[:, i], d_an[:, i])
        est = statistic.estimate
        assert float(est.diagonal().mean()) == pytest.approx(r_true,
                                                           relative=0.05)
        assert est.shape == (2, 2)

    def test_decay_forgets_old(self):
        statistic = Statistic(decay=0.5)
        for v in (torch.tensor([4.0]), torch.tensor([1.0])):
            statistic.update(v, v)
        e = statistic.estimate
        # weighted mean of 16 and 1 with the recent sample weighting
        # double: (0.5·16 + 1)/1.5 = 6 — below the unweighted midpoint.
        assert 1.0 < float(e[0, 0]) < 8.5
        assert float(e[0, 0]) == pytest.approx(6.0)

    def test_var_is_diagonal(self):
        statistic = Statistic()
        statistic.update(torch.tensor([2.0, 0.0]), torch.tensor([1.0, 0.0]))
        assert statistic.var.tolist() == pytest.approx([2.0, 0.0], abs=1e-12)


class TestAoei:
    def test_small_innovation_keeps_r(self):
        r = torch.tensor([1.0, 1.0], dtype=DTYPE)
        output = inflate(torch.tensor([0.5, 0.8], dtype=DTYPE),
                   torch.tensor([0.3, 0.3], dtype=DTYPE), r)
        assert torch.equal(output, r)

    def test_large_innovation_inflates(self):
        output = inflate(torch.tensor([4.0], dtype=DTYPE),
                   torch.tensor([1.0], dtype=DTYPE), 2.0)
        assert float(output[0]) == pytest.approx(15.0)   # d² − var_ens

    def test_monotone_in_innovation(self):
        a = inflate(torch.tensor([1.0]), torch.tensor([0.5]), 1.0)
        b = inflate(torch.tensor([5.0]), torch.tensor([0.5]), 1.0)
        assert float(b[0]) >= float(a[0])


class TestFullR:
    def test_whitening_equivalence(self, soil):
        """Analysis with correlated R equals the whitened problem:
        L⁻¹whitening of obs+pred and R=I gives the same update."""
        from nasa_farm.state.field import initial_state
        s = initial_state(16, soil)
        predicted = torch.stack([s.sw[:, 0],
                            s.sw[:, 0] * 0.5 + 1.0], -1)
        r = torch.tensor([[2.0, 1.2], [1.2, 2.0]], dtype=DTYPE)
        observed = torch.tensor([80.0, 45.0], dtype=DTYPE)
        l = torch.linalg.cholesky(r)
        a = kalman_update(s, predicted, observed, r, perturb=False)
        b = kalman_update(s, torch.linalg.solve(l, predicted.t()).t(),
                        torch.linalg.solve(l, observed),
                        torch.eye(2, dtype=DTYPE), perturb=False)
        assert torch.allclose(a.sw, b.sw, atol=1e-10)
        assert torch.allclose(a.lai, b.lai, atol=1e-10)

    def test_correlated_noise_shape(self, soil):
        """Full R path runs the stochastic update and stays in bounds."""
        from nasa_farm.state.field import initial_state
        s = initial_state(16, soil)
        predicted = torch.stack([s.sw[:, 0], s.lai], -1)
        r = torch.tensor([[4.0, 1.0], [1.0, 0.5]], dtype=DTYPE)
        s2 = kalman_update(s, predicted, torch.tensor([70.0, 3.0], dtype=DTYPE),
                         r, generation=torch.Generator().manual_seed(0))
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
        statistic = Statistic()
        schedule = [Observation(200, ("vwc",), torch.tensor([0.4], dtype=DTYPE),
                            0.05),
                 Observation(250, ("vwc",), torch.tensor([0.6], dtype=DTYPE),
                             0.05)]
        assimilate(physics_model, field, forcing, acts, schedule,
                   r_stat=statistic, seed=0)
        assert statistic.estimate is not None and statistic.estimate.shape == (1, 1)
        assert float(statistic.estimate[0, 0]) >= 0.0
