"""Pedotransfer physics — hydraulic ordering across textures."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.state.soil import SoilProfile, theta_fc, theta_sat, theta_wp

TEXTURES = [
    (92.0, 3.0), (82.0, 8.0), (65.0, 15.0), (42.0, 22.0),
    (20.0, 20.0), (32.0, 35.0), (22.0, 50.0),
]


class TestPedotransfer:
    @pytest.mark.parametrize("sand,clay", TEXTURES)
    def test_ordering_wp_lt_fc_lt_sat(self, sand, clay):
        s, c, om = torch.tensor(sand), torch.tensor(clay), torch.tensor(2.0)
        wp, fc, sat = theta_wp(s, c, om), theta_fc(s, c, om), theta_sat(s, c)
        assert float(wp) < float(fc) < float(sat), (sand, clay, wp, fc, sat)

    def test_clay_holds_more_than_sand(self):
        om = torch.tensor(2.0)
        wp_clay = theta_wp(torch.tensor(22.0), torch.tensor(50.0), om)
        wp_sand = theta_wp(torch.tensor(92.0), torch.tensor(3.0), om)
        assert float(wp_clay) > float(wp_sand)


class TestProfile:
    def test_storages_scale_with_depth(self):
        p = SoilProfile.preset("loam")
        # layer2 is 1.50 m vs layer1 0.30 m → 5× the storage
        assert float(p.layer2.theta_fc) > float(p.layer1.theta_fc)
        assert float(p.layer2.theta_sat / p.layer1.theta_sat) == pytest.approx(1.5 / 0.3, rel=0.1)

    def test_bounds_hold_as_storages(self):
        for name in ("sand", "loamy_sand", "sandy_loam", "loam", "silt_loam",
                     "clay_loam", "clay"):
            p = SoilProfile.preset(name)
            for l in (p.layer1, p.layer2):
                assert float(l.theta_air) < float(l.theta_wp)
                assert float(l.theta_wp) < float(l.theta_fc)
                assert float(l.theta_fc) < float(l.theta_sat)

    def test_batch(self):
        p = SoilProfile.preset("loam", batch=4)
        assert p.batch == 4 and p.layer1.theta_fc.shape == (4,)

    def test_to_device(self):
        p = SoilProfile.preset("loam").to(torch.device("cpu"))
        assert p.layer1.theta_fc.device.type == "cpu"


class TestUncertaintySample:
    def test_bounds_and_internal_consistency(self):
        lo, hi = SoilProfile.preset("sandy_loam"), SoilProfile.preset("clay_loam")
        s = SoilProfile.sample(lo, hi, 16, seed=0)
        assert s.batch == 16
        for name, a, b in (("sand1", lo.sand1, hi.sand1),
                           ("clay1", lo.clay1, hi.clay1),
                           ("sand2", lo.sand2, hi.sand2),
                           ("clay2", lo.clay2, hi.clay2)):
            v = getattr(s, name)
            assert (v >= torch.minimum(a, b) - 1e-9).all()
            assert (v <= torch.maximum(a, b) + 1e-9).all()
        for l in (s.layer1, s.layer2):
            assert (l.theta_wp < l.theta_fc).all()
            assert (l.theta_fc < l.theta_sat).all()
            assert (l.theta_air < l.theta_wp).all()

    def test_deterministic_given_seed(self):
        lo, hi = SoilProfile.preset("loam"), SoilProfile.preset("clay_loam")
        a, b = SoilProfile.sample(lo, hi, 8, seed=3), \
            SoilProfile.sample(lo, hi, 8, seed=3)
        assert torch.equal(a.layer1.theta_fc, b.layer1.theta_fc)

    def test_multi_field_expands(self):
        """B bound rows × n draws → B·n members, field-major order."""
        lo = SoilProfile.preset("sandy_loam", batch=2)
        hi = SoilProfile.preset("clay_loam", batch=2)
        s = SoilProfile.sample(lo, hi, 4, seed=0)
        assert s.batch == 8
