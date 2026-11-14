"""ECC reordering — marginals preserved, dependence transferred."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.forcing import ecc, ranks, shuffle_forcing, synthetic_year


class TestRanks:
    def test_rank_range(self):
        x = torch.rand(6, 40, dtype=DTYPE,
                       generator=torch.Generator().manual_seed(0))
        r = ranks(x, 0)
        assert int(r.min()) == 0 and int(r.max()) == 5
        assert (r.sort(0).values == torch.arange(6, dtype=DTYPE)
                .unsqueeze(1)).all()

    def test_rank_matches_order(self):
        x = torch.tensor([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]], dtype=DTYPE)
        r = ranks(x, 0)
        assert r[:, 0].tolist() == [2.0, 0.0, 1.0]
        assert r[:, 1].tolist() == [0.0, 2.0, 1.0]


class TestEcc:
    def test_column_multisets_preserved(self):
        g = torch.Generator().manual_seed(0)
        fc = torch.rand(8, 60, generator=g, dtype=DTYPE)
        tm = torch.rand(8, 60, generator=g, dtype=DTYPE)
        out = ecc(fc, tm)
        for t in range(60):
            assert sorted(out[:, t].tolist()) == pytest.approx(
                sorted(fc[:, t].tolist()))

    def test_template_dependence_transferred(self):
        """A template member always ranked lowest pulls that member's
        column to the forecast minimum at every day."""
        g = torch.Generator().manual_seed(1)
        fc = torch.rand(8, 50, generator=g, dtype=DTYPE)
        tm = torch.rand(8, 50, generator=g, dtype=DTYPE)
        tm[0] = -1.0                             # member 0: rank 0 everywhere
        out = ecc(fc, tm)
        assert (out[0] == fc.min(0).values).all()

    def test_lag1_correlation_follows_template(self):
        """AR(1) template ranks transfer their autocorrelation: members
        come out temporally smooth even though raw forecast is white."""
        g = torch.Generator().manual_seed(2)
        eps = torch.randn(8, 300, generator=g, dtype=DTYPE)
        tm = torch.zeros(8, 300, dtype=DTYPE)
        tm[:, 0] = eps[:, 0]
        for t in range(1, 300):
            tm[:, t] = 0.98 * tm[:, t - 1] + eps[:, t]
        fc = torch.rand(8, 300, generator=g, dtype=DTYPE)
        out = ecc(fc, tm)

        def lag1(x):
            a, b = x[:, 1:], x[:, :-1]
            a = a - a.mean(1, keepdim=True)
            b = b - b.mean(1, keepdim=True)
            return float(((a * b).mean(1)
                          / (a.std(1) * b.std(1) + 1e-12)).mean())
        assert lag1(out) > 0.6
        assert lag1(fc) < 0.2

    def test_unequal_sizes(self):
        g = torch.Generator().manual_seed(3)
        out = ecc(torch.rand(6, 30, generator=g, dtype=DTYPE),
                  torch.rand(10, 30, generator=g, dtype=DTYPE))
        assert out.shape == (6, 30)

    def test_deterministic(self):
        g = torch.Generator().manual_seed(4)
        fc = torch.rand(8, 40, generator=g, dtype=DTYPE)
        tm = torch.rand(8, 40, generator=g, dtype=DTYPE)
        assert torch.equal(ecc(fc, tm), ecc(fc, tm))


class TestShuffleForcing:
    def test_marginals_and_contract(self):
        fc = torch.cat([synthetic_year(42.0, seed=s).tmin
                        for s in range(6)])
        tm = fc.clone()
        f = synthetic_year(42.0, seed=9).repeat(6)
        out = shuffle_forcing(
            f.__class__(fc, f.tmax, f.srad, f.precip, f.rh, f.u2, f.doy),
            f.__class__(tm, f.tmax, f.srad, f.precip, f.rh, f.u2, f.doy))
        for t in range(f.days):
            assert sorted(out.tmin[:, t].tolist()) == pytest.approx(
                sorted(fc[:, t].tolist()))
        assert torch.equal(out.doy, f.doy)
