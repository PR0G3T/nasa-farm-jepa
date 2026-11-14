"""Canopy dynamics — phenology ordering, growth limits, harvest mass split."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.crop import (
    canopy_step, co2_factor, harvest, intercepted_fraction, leaf_area_index,
    photoperiod_factor, potential_evaporation, potential_transpiration, traits,
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
            t(gdd0), t(0.0), t(0.0), t(0.0), tr,
            t(kw.get("tmin", 15.0)), t(kw.get("tmax", 28.0)),
            t(kw.get("srad", 25.0)), None, None,
            t(water), t(n), torch.tensor([growing]))

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
        _, _, _, f = canopy_step(t(float(gdd0)), t(0.0), t(0.0), t(5000.0), tr,
                                 t(20.0), t(30.0), t(25.0), None, None,
                                 t(1.0), t(1.0), torch.tensor([True]))
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


class TestPhotoperiod:
    def test_long_days_delay_sensitive_crop(self, crops):
        tr = crops.get("soybean").unsqueeze(0)  # pp_sens 0.30, crit 13.5 h
        assert float(photoperiod_factor(t(12.0), tr)) == pytest.approx(1.0)
        assert float(photoperiod_factor(t(15.0), tr)) < 1.0

    def test_insensitive_crop_immune(self, crops):
        tr = crops.get("fallow").unsqueeze(0)  # pp_sens 0
        assert float(photoperiod_factor(t(16.0), tr)) == 1.0

    def test_none_daylen_neutral(self, crops):
        """daylen=None ⇒ identical gdd increment to a neutral photoperiod."""
        tr = crops.get("soybean").unsqueeze(0)
        g1, *_ = canopy_step(t(100.0), t(0.0), t(0.0), t(0.0), tr,
                             t(15.0), t(25.0), t(20.0), None, None,
                             t(1.0), t(1.0), torch.tensor([True]))
        g2, *_ = canopy_step(t(100.0), t(0.0), t(0.0), t(0.0), tr,
                             t(15.0), t(25.0), t(20.0), t(12.0), None,
                             t(1.0), t(1.0), torch.tensor([True]))
        assert float(g1) == pytest.approx(float(g2))


class TestVernalization:
    def _wheat(self, crops):
        return crops.get("wheat").unsqueeze(0)  # vern_req 45 d, gdd_fl 0.62

    def test_unvernalized_stalls_at_threshold(self, crops):
        tr = self._wheat(crops)
        gate = float(tr[0, IDX["gdd_fl"]] * tr[0, IDX["gdd_mat"]])
        gdd, *_ = canopy_step(t(gate), t(0.0), t(0.0), t(3000.0), tr,
                              t(15.0), t(25.0), t(20.0), None, None,
                              t(1.0), t(1.0), torch.tensor([True]))
        assert float(gdd) <= gate + 1e-9

    def test_vernalized_proceeds(self, crops):
        tr = self._wheat(crops)
        gate = float(tr[0, IDX["gdd_fl"]] * tr[0, IDX["gdd_mat"]])
        gdd, *_ = canopy_step(t(gate), t(50.0), t(0.0), t(3000.0), tr,
                              t(15.0), t(25.0), t(20.0), None, None,
                              t(1.0), t(1.0), torch.tensor([True]))
        assert float(gdd) > gate

    def test_chill_accumulates_only_in_window(self, crops):
        tr = self._wheat(crops)
        args = (tr, t(0.0), t(5.0), t(5.0), None, None,
                t(1.0), t(1.0), torch.tensor([True]))
        *_, f_cool = canopy_step(t(0.0), t(10.0), t(0.0), t(0.0), *args)
        assert float(f_cool["vern_new"]) == pytest.approx(11.0)
        *_, f_hot = canopy_step(t(0.0), t(10.0), t(0.0), t(0.0), tr,
                                t(15.0), t(30.0), t(5.0), None, None,
                                t(1.0), t(1.0), torch.tensor([True]))
        assert float(f_hot["vern_new"]) == pytest.approx(10.0)  # no chill


class TestFrost:
    def test_lethal_frost_matures_tender_crop(self, crops):
        tr = crops.get("maize").unsqueeze(0)  # t_kill −2
        _, _, _, f = canopy_step(t(500.0), t(0.0), t(0.0), t(2000.0), tr,
                                 t(-5.0), t(2.0), t(10.0), None, None,
                                 t(1.0), t(1.0), torch.tensor([True]))
        assert bool(f["frost"][0]) and bool(f["mature"][0])

    def test_hardy_crop_survives_same_frost(self, crops):
        tr = crops.get("wheat").unsqueeze(0)  # t_kill −15
        _, _, _, f = canopy_step(t(500.0), t(0.0), t(0.0), t(2000.0), tr,
                                 t(-5.0), t(2.0), t(10.0), None, None,
                                 t(1.0), t(1.0), torch.tensor([True]))
        assert not bool(f["frost"][0]) and not bool(f["mature"][0])

    def test_frost_salvages_partial_yield(self, crops):
        tr = maize(crops)
        y0, _ = harvest(t(10_000.0), tr)
        yf, _ = harvest(t(10_000.0), tr, frost=torch.tensor([True]))
        assert 0.0 < float(yf) < float(y0)


class TestHeatStress:
    def test_heat_counts_only_in_reproductive_window(self, crops):
        tr = crops.get("maize").unsqueeze(0)  # gdd_fl 0.60, heat_t 34
        lo = float(tr[0, IDX["gdd_fl"]] * tr[0, IDX["gdd_mat"]])

        def inc(gdd0):
            *_, f = canopy_step(t(gdd0), t(0.0), t(0.0), t(0.0), tr,
                                t(25.0), t(40.0), t(25.0), None, None,
                                t(1.0), t(1.0), torch.tensor([True]))
            return float(f["heat_dd"])

        assert inc(lo - 50.0) == 0.0
        assert inc(lo + 50.0) == pytest.approx(40.0 - 34.0)

    def test_heat_degrades_harvest_index(self, crops):
        tr = maize(crops)
        y0, _ = harvest(t(10_000.0), tr, t(0.0))
        yh, _ = harvest(t(10_000.0), tr, t(50.0))
        assert float(yh) < float(y0)

    def test_frost_and_heat_compound(self, crops):
        tr = maize(crops)
        yh, _ = harvest(t(10_000.0), tr, t(50.0))
        yhf, _ = harvest(t(10_000.0), tr, t(50.0), torch.tensor([True]))
        assert float(yhf) < float(yh)


class TestCO2:
    def test_c3_responds_more_than_c4(self, crops):
        co2 = t(560.0)
        f_w = co2_factor(co2, crops.get("wheat").unsqueeze(0))  # C3 0.20
        f_m = co2_factor(co2, crops.get("maize").unsqueeze(0))  # C4 0.05
        assert float(f_w) > float(f_m) > 1.0

    def test_reference_ppm_is_neutral(self, crops):
        f = co2_factor(t(420.0), crops.get("wheat").unsqueeze(0))
        assert float(f) == pytest.approx(1.0)

    def test_none_co2_neutral_in_step(self, crops):
        """co2=None ⇒ biomass increment identical to the 420 ppm case."""
        tr = maize(crops)
        args = (t(15.0), t(28.0), t(25.0), None)
        *_, f0 = canopy_step(t(800.0), t(0.0), t(0.0), t(5000.0), tr,
                             *args, None, t(1.0), t(1.0),
                             torch.tensor([True]))
        *_, f4 = canopy_step(t(800.0), t(0.0), t(0.0), t(5000.0), tr,
                             *args, t(420.0), t(1.0), t(1.0),
                             torch.tensor([True]))
        assert float(f0["delta_b"]) == pytest.approx(float(f4["delta_b"]))
