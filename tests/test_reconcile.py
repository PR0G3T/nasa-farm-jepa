"""Multi-source forcing consensus — precision weighting, bounds."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.forcing import consensus, synthetic_year
from nasa_farm.types import Forcing


def _make(tmin_off=0.0, precip_scale=1.0, n=1):
    f = synthetic_year(42.0, seed=1)
    return Forcing(f.tmin + tmin_off, f.tmax + tmin_off, f.srad,
                   f.precip * precip_scale, f.rh, f.u2, f.doy)


class TestConsensus:
    def test_identical_sources_recovered(self):
        f = _make()
        c, spread = consensus({"a": f, "b": f, "c": f})
        assert torch.allclose(c.tmin, f.tmin, atol=1e-12)
        assert float(spread["tmin"].max()) < 1e-9

    def test_uniform_weights_give_mean(self):
        c, _ = consensus({"a": _make(-2.0), "b": _make(4.0)})
        assert float(c.tmin.mean()) == pytest.approx(
            float((_make(-2.0).tmin.mean() + _make(4.0).tmin.mean()) / 2))

    def test_weights_favor_precise(self):
        c, _ = consensus({"a": _make(-2.0), "b": _make(4.0)},
                         prec={"a": 9.0, "b": 1.0})
        d_a = float((c.tmin - _make(-2.0).tmin).abs().mean())
        d_b = float((c.tmin - _make(4.0).tmin).abs().mean())
        assert d_a < d_b

    def test_var_prec_override(self):
        a, b = _make(), _make(precip_scale=3.0)
        c, _ = consensus({"a": a, "b": b},
                         var_prec={"precip": {"a": 1.0, "b": 0.0}})
        assert torch.equal(c.precip, a.precip)

    def test_spread_captures_disagreement(self):
        _, spread = consensus({"a": _make(), "b": _make(tmin_off=6.0)})
        assert float(spread["tmin"].mean()) == pytest.approx(3.0)
        assert float(spread["precip"].max()) == 0.0

    def test_physical_contract(self):
        c, _ = consensus({"a": _make(), "b": _make()})
        assert (c.tmax >= c.tmin).all()
        assert (c.precip >= 0).all() and (c.rh <= 100).all()

    def test_member_mean_used(self):
        f = _make().repeat(4)
        c, _ = consensus({"a": f, "b": f})
        assert c.batch == 1 and c.days == f.days
