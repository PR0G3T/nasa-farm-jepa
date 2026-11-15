"""Scenario ensemble builders."""

import torch

from nasa_farm.forcing import (
    augment,
    climatology_ensemble,
    ensemble_of,
    synthetic_year,
    tile_to_horizon,
)
from nasa_farm.types import Forcing


class TestSyntheticYear:
    def test_shape_and_bounds(self):
        f = synthetic_year(42.0, seed=1)
        assert f.tmin.shape == (1, 365)
        assert (f.precip >= 0).all() and (f.srad > 0).all()
        assert (f.tmax >= f.tmin).all()
        assert torch.equal(f.doy[0], torch.arange(1, 366, dtype=f.doy.dtype))

    def test_deterministic_given_seed(self):
        a, b = synthetic_year(42.0, seed=5), synthetic_year(42.0, seed=5)
        assert torch.equal(a.tmin, b.tmin) and torch.equal(a.precip, b.precip)

    def test_different_seeds_differ(self):
        a, b = synthetic_year(42.0, seed=5), synthetic_year(42.0, seed=6)
        assert not torch.equal(a.tmin, b.tmin)

    def test_southern_hemisphere_phase_shift(self):
        nh = synthetic_year(42.0, seed=3)
        sh = synthetic_year(-42.0, seed=3)
        # July (doy ~200): cold in SH, warm in NH
        assert float(nh.tmin[0, 200]) > float(sh.tmin[0, 200])
        assert float(sh.tmin[0, 20]) > float(nh.tmin[0, 20])


class TestEnsemble:
    def test_tile_horizon(self):
        f = tile_to_horizon(synthetic_year(40.0, seed=1), 3)
        assert f.days == 3 * 365
        assert torch.equal(f.doy[0, :365], f.doy[0, 365:730])

    def test_ensemble_batch(self):
        ensemble = ensemble_of([synthetic_year(40.0, seed=s) for s in range(4)])
        assert ensemble.batch == 4 and ensemble.days == 365

    def test_climatology_shift(self):
        years = [synthetic_year(40.0, seed=s) for s in range(3)]
        base = climatology_ensemble(years, 2)
        hot = climatology_ensemble(years, 2, delta_t=2.5, precip_scale=0.7)
        assert base.batch == 3 and base.days == 730
        assert torch.allclose(hot.tmin, base.tmin + 2.5)
        assert torch.allclose(hot.precip, base.precip * 0.7)
        assert torch.equal(hot.doy, base.doy)

    def test_day_slice(self):
        f = synthetic_year(40.0, seed=1).repeat(3)
        d = f.day(10)
        assert d["tmin"].shape == (3,)
        assert float(d["doy"][0]) == 11.0


class TestCO2Channel:
    def test_synthetic_default_420(self):
        f = synthetic_year(42.0, seed=1)
        assert (f.co2 == 420.0).all()

    def test_synthetic_ppm_param(self):
        f = synthetic_year(42.0, co2_ppm=560.0, seed=1)
        assert (f.co2 == 560.0).all()

    def test_none_fills_reference(self):
        v = torch.zeros(1, 5, dtype=torch.float64)
        f = Forcing(v, v, v, v, v, v, v)
        assert (f.co2 == 420.0).all()

    def test_scalar_ppm_expands(self):
        v = torch.zeros(1, 5, dtype=torch.float64)
        f = Forcing(v, v, v, v, v, v, v, 550.0)
        assert f.co2.shape == (1, 5) and (f.co2 == 550.0).all()

    def test_day_slice_carries_co2(self):
        d = synthetic_year(42.0, seed=1).day(10)
        assert "co2" in d and float(d["co2"][0]) == 420.0


class TestAugment:
    def test_batch_grows_by_members(self):
        a = augment(synthetic_year(42.0, seed=1), 4, seed=0)
        assert a.batch == 5 and a.days == 365

    def test_members_differ(self):
        a = augment(synthetic_year(42.0, seed=1), 3, seed=0)
        assert not torch.equal(a.tmin[1], a.tmin[2])

    def test_original_preserved_first(self):
        f = synthetic_year(42.0, seed=1)
        a = augment(f, 2, seed=0)
        assert torch.equal(a.tmin[0], f.tmin[0])
        assert torch.equal(a.precip[0], f.precip[0])

    def test_physical_bounds(self):
        a = augment(synthetic_year(42.0, seed=1), 4, seed=0)
        assert (a.precip >= 0).all() and (a.srad > 0).all()
        assert (a.tmax >= a.tmin).all() and (a.rh <= 100).all()

    def test_shared_temperature_innovation(self):
        """tmin/tmax share their AR(1) anomaly → diurnal range stays put."""
        f = synthetic_year(42.0, seed=1)
        a = augment(f, 8, seed=0)
        spread = (a.tmax - a.tmin)[1:] - (f.tmax - f.tmin)
        assert float(spread.abs().mean()) < 1.0

    def test_deterministic_given_seed(self):
        f = synthetic_year(42.0, seed=1)
        a, b = augment(f, 3, seed=7), augment(f, 3, seed=7)
        assert torch.equal(a.tmin, b.tmin) and torch.equal(a.precip, b.precip)

    def test_temporal_correlation(self):
        """AR(1) perturbations are day-to-day correlated, not white noise."""
        a = augment(synthetic_year(42.0, seed=1), 64, rho=0.7, seed=0)
        e = a.tmin[1:] - a.tmin[0]  # (64, T) member anomalies
        e = e - e.mean(0, keepdim=True)
        r1 = (e[:, 1:] * e[:, :-1]).mean() / e.var()
        assert 0.4 < float(r1) < 0.9
