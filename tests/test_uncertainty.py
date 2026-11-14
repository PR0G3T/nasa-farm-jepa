"""Conformal UQ — quantile correctness, finite-sample marginal coverage."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.uncertainty import (
    SplitConformal, conf_level, coverage, interval, weighted_quantile,
)


class TestWeightedQuantile:
    def test_unweighted_matches_sorted(self):
        x = torch.arange(100, dtype=DTYPE)
        assert float(weighted_quantile(x, 0.9)) == pytest.approx(90.0, abs=1.0)

    def test_weights_shift_quantile(self):
        x = torch.arange(10, dtype=DTYPE)
        w = torch.zeros(10, dtype=DTYPE)
        w[8] = 0.9
        w[1] = 0.1
        assert float(weighted_quantile(x, 0.5, w)) == pytest.approx(8.0)


class TestConformalLevel:
    def test_exact_finite_sample_rank(self):
        scores = torch.arange(1, 101, dtype=DTYPE)  # 1..100
        q = conf_level(scores, alpha=0.1)
        # ceil(101·0.9)/100 = 0.91 quantile → ≈ 91
        assert 89.0 <= float(q) <= 93.0

    def test_alpha_zero_gives_max(self):
        scores = torch.arange(1, 51, dtype=DTYPE)
        assert float(conf_level(scores, alpha=0.0)) == float(scores.max())


class TestIntervalCoverage:
    def test_marginal_coverage_iid(self):
        """Exchangeable calibration ⇒ empirical coverage ≈ 1−α."""
        g = torch.Generator().manual_seed(0)
        y = torch.randn(1200, generator=g, dtype=DTYPE)
        cal, test = y[:200], y[200:]
        scores = cal.abs()
        lo, hi = interval(torch.zeros(test.shape, dtype=DTYPE), scores, 0.1)
        c = coverage(test, lo, hi)
        assert 0.85 <= c <= 0.96

    def test_split_conformal_per_channel(self):
        g = torch.Generator().manual_seed(0)
        y = torch.randn(400, 2, generator=g, dtype=DTYPE) \
            * torch.tensor([1.0, 5.0], dtype=DTYPE)
        cal, test = y[:200], y[200:]
        sc = SplitConformal(alpha=0.1).fit(torch.zeros_like(cal), cal)
        lo, hi = sc.bound(torch.zeros_like(test))
        assert lo.shape == test.shape
        for c in range(2):
            assert 0.85 <= coverage(test[:, c], lo[:, c], hi[:, c]) <= 0.97
        assert (hi - lo)[:, 1].mean() > (hi - lo)[:, 0].mean() * 3.0
