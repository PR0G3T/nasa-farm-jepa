"""Scoring-rule correctness — analytic values, edge cases, and the
cross-checks that make the metrics trustworthy enough to publish."""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from nasa_farm import metrics as mx
from nasa_farm.constants import DTYPE


def t(*v):
    return torch.tensor(v, dtype=DTYPE)


def test_perfect_prediction():
    o = t(100.0, 200.0, 300.0, 400.0)
    p = o.clone()
    assert mx.rmse(p, o) == 0 and mx.mae(p, o) == 0 and mx.mbe(p, o) == 0
    assert mx.nrmse(p, o) == 0 and mx.nse(p, o) == 1 and mx.r2(p, o) == 1
    assert mx.kge(p, o, "2009") == 1 and mx.kge(p, o) == 1
    assert mx.d_index(p, o) == 1 and mx.ccc(p, o) == 1


def test_known_values():
    o, p = t(1.0, 2.0, 3.0), t(2.0, 3.0, 4.0)
    assert mx.rmse(p, o) == pytest.approx(1.0)
    assert mx.mae(p, o) == pytest.approx(1.0)
    assert mx.mbe(p, o) == pytest.approx(1.0)
    # sse = 3, sst = 2 → NSE = 1 − 3/2 = −0.5 (a pure bias kills EF)
    assert mx.nse(p, o) == pytest.approx(-0.5)
    # ... yet Pearson R² stays 1 — r² is blind to bias, why we report both
    assert mx.r2(p, o) == pytest.approx(1.0)
    # Willmott d: num=3, den = Σ(|p−2|+|o−2|)² = 1²+1²+3² = 11 → 8/11
    assert mx.d_index(p, o) == pytest.approx(8 / 11)
    # Lin CCC (population moments): 2·(2/3) / (2/3+2/3+1) = 0.5714…
    assert mx.ccc(p, o) == pytest.approx(4 / 7)


def test_kge_versions():
    # p = 2·o: perfect correlation, bias factor 2.
    o, p = t(1.0, 2.0, 3.0, 4.0), t(2.0, 4.0, 6.0, 8.0)
    # '09: variability = σ ratio = 2 → 1−sqrt(0+1+1) = 1−sqrt(2)
    assert mx.kge(p, o, "2009") == pytest.approx(1 - math.sqrt(2))
    # '12: variability = CV ratio = 1 → 1−sqrt(0+0+1) = 0
    assert mx.kge(p, o, "2012") == pytest.approx(0.0)


def test_nse_below_zero_when_worse_than_mean():
    o = t(1.0, 2.0, 3.0, 4.0)
    p = torch.full_like(o, 10.0)
    assert mx.nse(p, o) < 0


def test_nrmse_norms():
    o, p = t(10.0, 20.0, 30.0), t(11.0, 19.0, 30.0)
    r = mx.rmse(p, o)
    assert mx.nrmse(p, o, "mean") == pytest.approx(float(r / o.mean()))
    assert mx.nrmse(p, o, "sd") == pytest.approx(float(r / o.std()))
    assert mx.nrmse(p, o, "range") == pytest.approx(
        float(r / (o.max() - o.min())))
    # auto: mean→0 falls back to sd without producing inf/nan
    o2, p2 = t(-1.0, 0.0, 1.0), t(-0.9, 0.1, 1.2)
    output = mx.nrmse(p2, o2, "auto")
    assert torch.isfinite(output)


def test_nan_awareness():
    o = t(1.0, float("nan"), 3.0)
    p = t(1.5, 9.0, 3.5)          # the NaN row must be dropped entirely
    assert mx.mae(p, o) == pytest.approx(0.5)
    assert mx.score(p, o)["n"] == 2
    assert math.isnan(float(mx.rmse(p, torch.full_like(o, float("nan")))))


def test_broadcast():
    o = t(1.0, 2.0, 3.0)
    assert mx.mae(torch.tensor(2.0, dtype=DTYPE), o) \
        == pytest.approx(2.0 / 3.0)


def test_crps_single_member_is_mae():
    e = t(2.0, 5.0).unsqueeze(-1)          # 2 obs, 1 member each
    o = t(1.0, 3.0)
    assert mx.crps(e, o) == pytest.approx((1.0 + 2.0) / 2)


def test_crps_constant_ensemble_is_mae():
    e = torch.full((3, 8), 4.0, dtype=DTYPE)
    o = t(1.0, 4.0, 6.0)
    assert mx.crps(e, o) == pytest.approx((3.0 + 0.0 + 2.0) / 3)


def test_crps_hand_case():
    # members {0, 1} around y=0.5 → symmetric, fair CRPS = 0
    e = t(0.0, 1.0).unsqueeze(0)
    assert mx.crps(e, t(0.5)) == pytest.approx(0.0)


def test_crps_matches_quadratic_form():
    """The O(m log m) sorted form must equal the textbook O(m²) pairwise
    definition — cross-validation of the two implementations."""
    g = torch.Generator().manual_seed(0)
    e = torch.randn(50, 9, generator=g, dtype=DTYPE)
    o = torch.randn(50, generator=g, dtype=DTYPE)
    m = e.shape[-1]
    ae = (e - o.unsqueeze(-1)).abs().mean(-1)
    pair = (e.unsqueeze(-1) - e.unsqueeze(-2)).abs()
    # textbook fair CRPS: mean|x−y| − Σ_{i≠j}|xᵢ−xⱼ| / (2·m·(m−1))
    brute = (ae - pair.sum((-1, -2)) / (2 * m * (m - 1))).mean()
    assert mx.crps(e, o) == pytest.approx(float(brute), rel=1e-10)


@given(st.integers(min_value=2, max_value=64))
@settings(max_examples=16)
def test_crps_nonnegative(m):
    g = torch.Generator().manual_seed(m)
    e = torch.randn(20, m, generator=g, dtype=DTYPE)
    o = torch.randn(20, generator=g, dtype=DTYPE)
    assert float(mx.crps(e, o)) >= -1e-9


def test_pit_uniform_when_calibrated():
    """Obs drawn from the same distribution as members → PIT ≈ uniform."""
    g = torch.Generator().manual_seed(1)
    e = torch.randn(4000, 31, generator=g, dtype=DTYPE)
    o = torch.randn(4000, generator=g, dtype=DTYPE)
    h = mx.pit_hist(e, o, bins=10, generation=torch.Generator().manual_seed(2))
    frequency = h / h.sum()
    assert bool((frequency > 0.05).all()) and bool((frequency < 0.15).all())


def test_pit_detects_bias():
    g = torch.Generator().manual_seed(3)
    e = torch.randn(2000, 21, generator=g, dtype=DTYPE)
    o = torch.randn(2000, generator=g, dtype=DTYPE) + 3.0   # obs way high
    h = mx.pit_hist(e, o, bins=10)
    assert h[-1] > 0.9 * h.sum()     # mass piles at the right edge
    assert h[:3].sum() == 0          # nothing left below the ensemble


def test_picp_oracle_band():
    """μ ± 1.645σ on N(0,1) samples → ≈90 % coverage at α=0.1."""
    g = torch.Generator().manual_seed(4)
    o = torch.randn(5000, generator=g, dtype=DTYPE)
    z = 1.6448536269514722
    c = mx.picp(torch.full_like(o, -z), torch.full_like(o, z), o)
    assert float(c) == pytest.approx(0.9, abs=0.02)


def test_interval_score():
    # inside → score = width
    assert mx.interval_score(t(0.0), t(2.0), t(1.0), 0.1) == pytest.approx(2.0)
    # below lo by 1 → width + (2/α)·1 = 2 + 20 = 22
    assert mx.interval_score(t(2.0), t(4.0), t(1.0), 0.1) == pytest.approx(22.0)


def test_mean_width_and_pinball():
    assert mx.mean_width(t(0.0, 1.0), t(2.0, 3.0)) == pytest.approx(2.0)
    # τ = 0.5 → pinball = 0.5·MAE
    p, o = t(1.0, 2.0, 5.0), t(0.0, 3.0, 4.0)
    assert mx.pinball(p, o, 0.5) == pytest.approx(0.5 * (1 + 1 + 1) / 3)


def test_spread_skill_calibrated():
    """Members and obs iid from the same distribution → rmse ≈ spread."""
    g = torch.Generator().manual_seed(5)
    e = torch.randn(4000, 32, generator=g, dtype=DTYPE) * 2.0
    o = torch.randn(4000, generator=g, dtype=DTYPE) * 2.0
    assert float(mx.spread_skill(e, o)) == pytest.approx(1.0, abs=0.1)


def test_score_bundle_keys_and_null():
    s = mx.score(t(1.0, 2.0), t(1.0, 2.5))
    for k in ("rmse", "nrmse", "mae", "mbe", "r2", "nse", "kge09",
              "kge12", "d", "ccc", "n"):
        assert k in s
    single = mx.score(t(1.0), t(2.0))
    assert single["r2"] is None and single["nse"] is None  # undefined @ n=1
    assert single["rmse"] == 1.0


def test_score_ens_bundle():
    g = torch.Generator().manual_seed(6)
    e = torch.randn(30, 12, generator=g, dtype=DTYPE)
    o = torch.randn(30, generator=g, dtype=DTYPE)
    s = mx.score_ens(e, o, alphas=(0.1, 0.5))
    for k in ("rmse", "crps", "spread", "spread_skill", "pit",
              "picp_0.1", "is_0.1", "width_0.1",
              "picp_0.5", "is_0.5", "width_0.5"):
        assert k in s
    assert len(s["pit"]) == 10 and sum(s["pit"]) == 30
    # α = miscoverage: the 90 % band covers more than the 50 % band
    assert s["picp_0.1"] >= s["picp_0.5"]


def test_dtype_float64():
    o = t(1.0, 2.0, 3.0)
    assert mx.rmse(o, o).dtype == torch.float64
