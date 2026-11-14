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
        s, c, organic = torch.tensor(sand), torch.tensor(clay), torch.tensor(2.0)
        wp, fc, sat = theta_wp(s, c, organic), theta_fc(s, c, organic), theta_sat(s, c)
        assert float(wp) < float(fc) < float(sat), (sand, clay, wp, fc, sat)

    def test_clay_holds_more_than_sand(self):
        organic = torch.tensor(2.0)
        wp_clay = theta_wp(torch.tensor(22.0), torch.tensor(50.0), organic)
        wp_sand = theta_wp(torch.tensor(92.0), torch.tensor(3.0), organic)
        assert float(wp_clay) > float(wp_sand)


class TestProfile:
    def test_storages_scale_with_depth(self):
        p = SoilProfile.preset("loam")
        # layer2 is 1.50 m vs layer1 0.30 m → 5× the storage
        assert float(p.layer(1).theta_fc) > float(p.layer(0).theta_fc)
        assert float(p.layer(1).theta_sat / p.layer(0).theta_sat) == pytest.approx(1.5 / 0.3, relative=0.1)

    def test_bounds_hold_as_storages(self):
        for name in ("sand", "loamy_sand", "sandy_loam", "loam", "silt_loam",
                     "clay_loam", "clay"):
            p = SoilProfile.preset(name)
            for l in ((p.layer(0), p.layer(1))):
                assert float(l.theta_air) < float(l.theta_wp)
                assert float(l.theta_wp) < float(l.theta_fc)
                assert float(l.theta_fc) < float(l.theta_sat)

    def test_batch(self):
        p = SoilProfile.preset("loam", batch=4)
        assert p.batch == 4 and p.layer(0).theta_fc.shape == (4,)

    def test_to_device(self):
        p = SoilProfile.preset("loam").to(torch.device("cpu"))
        assert p.layer(0).theta_fc.device.type == "cpu"


class TestUncertaintySample:
    def test_bounds_and_internal_consistency(self):
        low, high = SoilProfile.preset("sandy_loam"), SoilProfile.preset("clay_loam")
        s = SoilProfile.sample(low, high, 16, seed=0)
        assert s.batch == 16
        for a, b, i in ((low.sand, high.sand, 0), (low.clay, high.clay, 0),
                        (low.sand, high.sand, 1), (low.clay, high.clay, 1)):
            v = s.sand[:, i] if a is low.sand else s.clay[:, i]
            assert (v >= torch.minimum(a[:, i], b[:, i]) - 1e-9).all()
            assert (v <= torch.maximum(a[:, i], b[:, i]) + 1e-9).all()
        for l in (s.layer(0), s.layer(1)):
            assert (l.theta_wp < l.theta_fc).all()
            assert (l.theta_fc < l.theta_sat).all()
            assert (l.theta_air < l.theta_wp).all()

    def test_deterministic_given_seed(self):
        low, high = SoilProfile.preset("loam"), SoilProfile.preset("clay_loam")
        a, b = SoilProfile.sample(low, high, 8, seed=3), \
            SoilProfile.sample(low, high, 8, seed=3)
        assert torch.equal(a.layer(0).theta_fc, b.layer(0).theta_fc)

    def test_multi_field_expands(self):
        """B bound rows × n draws → B·n members, field-major order."""
        low = SoilProfile.preset("sandy_loam", batch=2)
        high = SoilProfile.preset("clay_loam", batch=2)
        s = SoilProfile.sample(low, high, 4, seed=0)
        assert s.batch == 8
