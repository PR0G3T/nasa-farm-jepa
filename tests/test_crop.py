"""Canopy dynamics — phenology ordering, growth limits, harvest mass split."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.crop import (
    canopy_step, harvest, intercepted_fraction, leaf_area_index,
    potential_evaporation, potential_transpiration, traits,
)
from nasa_farm.state.crops import IDX


def t(v, n=1):
    return torch.full((n,), float(v), dtype=DTYPE)


def maize(crops):
    return crops.get("maize").unsqueeze(0)


class TestLAI:
    def test_zero_before_emergence(self, crops):
        tr = maize(crops)
        lai = leaf_area_index(t(0.0), tr)
        assert float(lai) == 0.0

    def test_peak_near_lai_max(self, crops):
        tr = maize(crops)
        lai = leaf_area_index(t(900.0), tr)  # past gdd_lai_max, before senescence
        assert float(lai) == pytest.approx(5.5 * (1 - torch.exp(torch.tensor(-3.0)).item()), rel=0.05)

    def test_zero_at_maturity(self, crops):
        tr = maize(crops)
        assert float(leaf_area_index(t(1750.0), tr)) == 0.0

    def test_perennial_never_senesces(self, crops):
        tr = crops.get("alfalfa").unsqueeze(0)
        lai = leaf_area_index(t(50_000.0), tr)
        assert float(lai) > 6.0  # lai_max 6.5, saturating toward it


class TestInterception:
    def test_monotonic_and_bounded(self):
        lai = torch.linspace(0, 10, 50, dtype=DTYPE)
        f = intercepted_fraction(lai, t(0.6))
        assert (f >= 0).all() and (f < 1).all()
        assert (f[1:] >= f[:-1]).all()


class TestCanopyStep:
    def _step(self, crops, gdd0=0.0, growing=True, water=1.0, n=1.0, **kw):
        tr = maize(crops)
        return canopy_step(
            t(gdd0), t(0.0), tr, t(kw.get("tmin", 15.0)), t(kw.get("tmax", 28.0)),
            t(kw.get("srad", 25.0)), t(water), t(n), torch.tensor([growing]), t(0.0))

    def test_gdd_accumulates_when_growing(self, crops):
        gdd, _, _, _ = self._step(crops)
        assert float(gdd) == pytest.approx((15 + 28) / 2 - 8.0)

    def test_no_accumulation_when_bare(self, crops):
        gdd, lai, bio, _ = self._step(crops, growing=False)
        assert float(gdd) == 0.0 and float(lai) == 0.0 and float(bio) == 0.0

    def test_biomass_needs_canopy(self, crops):
        """Before emergence: no LAI → no light interception → no growth."""
        _, lai, bio, f = self._step(crops, gdd0=0.0)
        assert float(f["delta_b"]) == 0.0

    def test_water_limited_growth(self, crops):
        tr = maize(crops)
        args = dict(tmin=15.0, tmax=28.0)
        _, _, _, fw1 = self._step(crops, gdd0=800.0, water=1.0, **args)
        _, _, _, fw0 = self._step(crops, gdd0=800.0, water=0.0, **args)
        assert float(fw1["delta_b"]) > 0.0
        assert float(fw0["delta_b"]) == 0.0

    def test_nitrogen_limited_growth(self, crops):
        """N stress reduces assimilation to ≥60 % (WOFOST canopy-expansion
        weighting), never fully stops it."""
        _, _, _, f0 = self._step(crops, gdd0=800.0, n=0.0)
        _, _, _, f1 = self._step(crops, gdd0=800.0, n=1.0)
        assert 0.0 < float(f0["delta_b"]) < float(f1["delta_b"])

    def test_maturity_flag(self, crops):
        tr = maize(crops)
        gdd0 = tr[0, IDX["gdd_mat"]] - 1.0
        _, _, _, f = canopy_step(t(float(gdd0)), t(5000.0), tr, t(20.0), t(30.0),
                                 t(25.0), t(1.0), t(1.0), torch.tensor([True]), t(5.0))
        assert bool(f["mature"][0])


class TestHarvest:
    def test_mass_split_conservative(self, crops):
        tr = maize(crops)
        y, r = harvest(t(10_000.0), tr)
        hi, res = tr[0, IDX["harvest_index"]], tr[0, IDX["residue_frac"]]
        assert float(y) == pytest.approx(10_000.0 * float(hi))
        assert float(r) == pytest.approx(10_000.0 * (1 - float(hi)) * float(res))
        assert float(y + r) <= 10_000.0


class TestPotentials:
    def test_evap_full_on_bare(self, crops):
        tr = maize(crops)
        ep = potential_evaporation(t(6.0), t(0.0), tr, torch.tensor([False]))
        assert float(ep) == pytest.approx(3.6)  # 0.6 × et0, fully bare

    def test_transp_zero_on_bare(self, crops):
        tr = maize(crops)
        tp = potential_transpiration(t(6.0), t(0.0), tr, torch.tensor([False]))
        assert float(tp) == 0.0

    def test_partition_shifts_with_canopy(self, crops):
        tr = maize(crops)
        growing = torch.tensor([True])
        tp = potential_transpiration(t(6.0), t(4.0), tr, growing)
        ep = potential_evaporation(t(6.0), t(4.0), tr, growing)
        assert float(tp) > 0.0 and float(ep) < 3.6


class TestTraits:
    def test_negative_crop_clamps_to_zero(self, crops):
        tr = traits(crops.table, torch.tensor([-1, 0, 3]))
        assert torch.equal(tr[0], crops.table[0])
        assert torch.equal(tr[1], crops.table[0])
        assert torch.equal(tr[2], crops.table[3])
