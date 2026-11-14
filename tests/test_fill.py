"""Whittaker gap-filling — interpolation, smoothness, weights contract."""

import numpy as numpy
import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.observe.fill import fill_series, whittaker


class TestWhittaker:
    def test_zero_lambda_interpolates(self):
        y = torch.arange(50, dtype=DTYPE) ** 1.3
        z = whittaker(y, lam=1e-9)
        assert torch.allclose(z, y, atol=1e-3)

    def test_linear_is_fixed_point(self):
        """D² annihilates linear signals: any λ reproduces them exactly."""
        y = 3.0 * torch.arange(60, dtype=DTYPE) - 10.0
        for lam in (0.1, 100.0, 1e5):
            assert torch.allclose(whittaker(y, lam=lam), y, atol=1e-6)

    def test_constant_is_fixed_point(self):
        y = torch.full((40,), 7.5, dtype=DTYPE)
        assert torch.allclose(whittaker(y, lam=1e4), y, atol=1e-6)

    def test_gap_is_filled(self):
        """w=0 cells get the smooth continuation, not NaN."""
        x = torch.linspace(0, 2 * numpy.pi, 90, dtype=DTYPE)
        y = x.sin()
        w = torch.ones_like(y)
        w[30:60] = 0.0
        z = whittaker(y, w, lam=1.0)
        assert torch.isfinite(z).all()
        assert (z[30:60] - y[30:60]).abs().max() < 0.35

    def test_low_weight_downweights_outlier(self):
        y = torch.zeros(60, dtype=DTYPE)
        y[30] = 5.0
        w = torch.ones_like(y)
        w[30] = 0.05
        z = whittaker(y, w, lam=10.0)
        assert abs(float(z[30])) < 0.5          # outlier mostly ignored

    def test_all_missing_is_nan(self):
        y = torch.full((20,), float("nan"), dtype=DTYPE)
        assert torch.isnan(whittaker(y)).all()


def D(start: str, days: int, step: int = 5) -> numpy.ndarray:
    """yyyymmdd ints, `step`-day spaced, month-safe."""
    import datetime as dt
    d0 = dt.date(int(start[:4]), int(start[4:6]), int(start[6:]))
    return numpy.array(
        [int((d0 + dt.timedelta(i * step)).strftime("%Y%m%d"))
         for i in range(days)], numpy.int32)


class TestFillSeries:
    def test_daily_grid_and_weights(self):
        dates = D("20230101", 20)
        values = numpy.linspace(0.2, 0.8, len(dates))
        index, z, w = fill_series(dates, values)
        assert index[-1] - index[0] + 1 == len(z)
        assert (w == 1.0).sum() == len(dates)
        assert (w == 0.0).any()                 # gaps between obs days

    def test_observed_days_preserved_low_lam(self):
        dates = D("20230101", 12)
        values = numpy.arange(len(dates), dtype=float)
        index, z, w = fill_series(dates, values, lam=1e-4)
        assert numpy.allclose(z[::5][: len(dates)], values, atol=0.1)

    def test_duplicate_dates_merge(self):
        dates = numpy.array([20230101, 20230101, 20230106], numpy.int32)
        values = numpy.array([0.0, 1.0, 0.9])
        index, z, w = fill_series(dates, values, lam=0.01)
        assert z[0] == pytest.approx(0.5, abs=0.05)

    def test_obs_weights_propagate(self):
        dates = D("20230101", 10)
        values = numpy.ones(len(dates))
        wt = numpy.full(len(dates), 0.3)
        index, z, w = fill_series(dates, values, weights=wt)
        assert w.max() <= 0.3

    def test_envelope_lifts_fit(self):
        """Down-biased noise should lose weight toward the upper edge."""
        dates = D("20230101", 20)
        t = numpy.linspace(0, numpy.pi, len(dates))
        values = 0.2 + 0.6 * numpy.sin(t)
        values[3] -= 0.4                          # cloud-dimmed point
        _, z0, w = fill_series(dates, values, lam=100.0)
        _, z1, _ = fill_series(dates, values, lam=100.0, envelope=3)
        obs_pos = int(numpy.nonzero(w > 0)[0][3])
        assert z1[obs_pos] > z0[obs_pos]
