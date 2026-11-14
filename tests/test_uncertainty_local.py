"""Localized + panel conformal — conditional coverage behavior."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.uncertainty import (
    LocalConformal, PanelConformal, SplitConformal, coverage,
)


class TestLocal:
    def _hetero(self, n=2000, seed=0):
        g = torch.Generator().manual_seed(seed)
        x = torch.rand(n, 1, generator=g, dtype=DTYPE) * 2 - 1
        y = torch.randn(n, generator=g, dtype=DTYPE) * (0.1 + x[:, 0].abs())
        return x, y

    def test_constant_features_match_split(self):
        """No local structure → same interval as split conformal."""
        g = torch.Generator().manual_seed(0)
        y = torch.randn(400, 1, generator=g, dtype=DTYPE)
        x = torch.zeros(400, 1, dtype=DTYPE)
        cal, te = slice(0, 300), slice(300, 400)
        z = torch.zeros(300, 1, dtype=DTYPE)
        lc = LocalConformal(alpha=0.1).fit(z, y[cal], x[cal])
        sc = SplitConformal(alpha=0.1).fit(z, y[cal])
        lo_l, hi_l = lc.bound(torch.zeros(100, 1, dtype=DTYPE), x[te])
        lo_s, hi_s = sc.bound(torch.zeros(100, 1, dtype=DTYPE))
        assert float((hi_l - lo_l).mean()) == pytest.approx(
            float((hi_s - lo_s).mean()), rel=0.35)

    def test_width_tracks_heteroscedasticity(self):
        """Intervals widen where |x| is large — conditional behavior."""
        x, y = self._hetero()
        cal, te = slice(0, 1500), slice(1500, 2000)
        lc = LocalConformal(alpha=0.1, bw=0.3).fit(
            torch.zeros(1500, dtype=DTYPE), y[cal], x[cal])
        lo, hi = lc.bound(torch.zeros(500, dtype=DTYPE), x[te])
        w = hi - lo
        far = (x[te, 0].abs() > 0.7)
        near = ~far
        assert float(w[far].mean()) > 1.4 * float(w[near].mean())
        assert coverage(y[te], lo, hi) >= 0.8

    def test_conditional_coverage_improves(self):
        """Local bands cover the hard region better than global split."""
        x, y = self._hetero(seed=3)
        cal, te = slice(0, 1500), slice(1500, 2000)
        z = torch.zeros(1500, dtype=DTYPE)
        lc = LocalConformal(alpha=0.1, bw=0.3).fit(z, y[cal], x[cal])
        sc = SplitConformal(alpha=0.1).fit(
            torch.zeros(1500, 1, dtype=DTYPE), y[cal].unsqueeze(1))
        lo_l, hi_l = lc.bound(torch.zeros(500, dtype=DTYPE), x[te])
        lo_s, hi_s = sc.bound(torch.zeros(500, 1, dtype=DTYPE))
        lo_s, hi_s = lo_s.squeeze(1), hi_s.squeeze(1)
        hard = x[te, 0].abs() > 0.8
        assert coverage(y[te][hard], lo_l[hard], hi_l[hard]) >= \
            coverage(y[te][hard], lo_s[hard], hi_s[hard]) - 1e-9


class TestPanel:
    def test_uniform_sim_is_conformal_like(self):
        g = torch.Generator().manual_seed(0)
        peers = torch.randn(200, generator=g, dtype=DTYPE).abs()
        sim = torch.zeros(200, dtype=DTYPE)
        pc = PanelConformal(alpha=0.1)
        lo, hi = pc.bound(torch.tensor(0.0, dtype=DTYPE), peers, sim)
        assert hi > 0 and hi - lo < peers.quantile(0.95) * 2 + 0.01

    def test_similar_peers_dominate(self):
        """Weights concentrate on similar peers → their level rules."""
        peers = torch.cat([torch.full((20,), 0.1, dtype=DTYPE),
                           torch.full((20,), 5.0, dtype=DTYPE)])
        sim = torch.cat([torch.ones(20, dtype=DTYPE),
                         torch.full((20,), -1.0, dtype=DTYPE)])
        pc = PanelConformal(alpha=0.1, eta=5.0)
        lo, hi = pc.bound(torch.tensor(0.0, dtype=DTYPE), peers, sim)
        assert float(hi) < 1.0               # similar (small) peers win

    def test_alpha_adapts(self):
        pc = PanelConformal(alpha=0.1, gamma=0.05)
        a0 = pc.alpha_t
        pc.update(err=True)
        assert pc.alpha_t < a0               # miss → widen next time
        pc.update(err=False)
        assert pc.alpha_t > a0 - 0.05
