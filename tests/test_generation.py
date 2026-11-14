"""CFM weather generator — contract, seeding, distributional sanity."""

import pytest
import torch

from nasa_farm.forcing import fit_gen, sample, synthetic_year


@pytest.fixture(scope="module")
def trained():
    years = [synthetic_year(42.0, seed=s) for s in range(6)]
    return years, fit_gen(years, epochs=150, hidden=64, seed=0)


class TestFlowGen:
    def test_shapes(self, trained):
        _, m = trained
        a = sample(m, 60, 8, seed=1)
        assert a.tmin.shape == (8, 60) and a.doy.shape == (8, 60)

    def test_physical_bounds(self, trained):
        _, m = trained
        a = sample(m, 90, 16, seed=2)
        assert (a.tmax >= a.tmin).all()
        assert (a.precip >= 0).all() and (a.srad > 0).all()
        assert (a.rh >= 0).all() and (a.rh <= 100).all()
        assert (a.u2 >= 0).all()
        assert torch.isfinite(a.tmin).all()

    def test_deterministic_given_seed(self, trained):
        _, m = trained
        a, b = (sample(m, 30, 4, seed=s) for s in (5, 5))
        assert torch.equal(a.tmin, b.tmin) and torch.equal(a.precip, b.precip)

    def test_seeds_differ(self, trained):
        _, m = trained
        a, b = (sample(m, 30, 4, seed=s) for s in (5, 6))
        assert not torch.equal(a.tmin, b.tmin)

    def test_members_differ(self, trained):
        _, m = trained
        a = sample(m, 60, 8, seed=3)
        assert float(a.tmin.std(0).mean()) > 0.0

    def test_distribution_near_training(self, trained):
        """Seasonal statistics land near the training years', not noise."""
        years, m = trained
        a = sample(m, 365, 32, seed=4)
        t_tmin = torch.stack([y.tmin[0] for y in years]).mean()
        t_tmax = torch.stack([y.tmax[0] for y in years]).mean()
        assert float(a.tmin.mean()) == pytest.approx(float(t_tmin), abs=8.0)
        assert float(a.tmax.mean()) == pytest.approx(float(t_tmax), abs=8.0)

    def test_seasonal_phase_carried(self, trained):
        _, m = trained
        a = sample(m, 365, 16, seed=6)
        summer = a.tmax[:, 170:240].mean()
        winter = a.tmax[:, :60].mean()
        assert float(summer) > float(winter)
