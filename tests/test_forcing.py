"""Scenario ensemble builders."""

import pytest
import torch

from nasa_farm.forcing import (
    climatology_ensemble, ensemble_of, synthetic_year, tile_to_horizon,
)


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
        ens = ensemble_of([synthetic_year(40.0, seed=s) for s in range(4)])
        assert ens.batch == 4 and ens.days == 365

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
