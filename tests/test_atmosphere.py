import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.atmosphere import (
    day_length, et0_hargreaves, et0_penman_monteith,
    extraterrestrial_radiation, growing_degree_days, psychrometric_constant,
    saturation_vapor_pressure, svp_slope, temperature_factor,
)


def t(v):
    return torch.tensor(v, dtype=DTYPE)


class TestExtraterrestrialRadiation:
    def test_summer_gt_winter_northern(self):
        ra_jun = extraterrestrial_radiation(t(45.0), t(172.0))
        ra_dec = extraterrestrial_radiation(t(45.0), t(355.0))
        assert ra_jun > ra_dec

    def test_southern_hemisphere_inverted(self):
        ra_jun = extraterrestrial_radiation(t(-45.0), t(172.0))
        ra_dec = extraterrestrial_radiation(t(-45.0), t(355.0))
        assert ra_dec > ra_jun

    def test_equator_nearly_constant(self):
        ra_jan = extraterrestrial_radiation(t(0.0), t(15.0))
        ra_jul = extraterrestrial_radiation(t(0.0), t(197.0))
        assert abs(float(ra_jan - ra_jul)) < 5.0

    def test_polar_night_is_zero(self):
        ra = extraterrestrial_radiation(t(80.0), t(355.0))
        assert float(ra) == pytest.approx(0.0, abs=1e-9)

    def test_physical_range(self):
        doy = torch.arange(1, 366, dtype=DTYPE)
        ra = extraterrestrial_radiation(t(30.0), doy)
        assert ra.min() > 10.0 and ra.max() < 45.0

    def test_solstice_symmetry_approximately(self):
        """Declination is near-symmetric about the solstice (doy 172)."""
        a = extraterrestrial_radiation(t(20.0), t(152.0))
        b = extraterrestrial_radiation(t(20.0), t(192.0))
        assert abs(float(a - b)) / float(a) < 0.02


class TestET0:
    def test_nonnegative(self):
        et = et0_hargreaves(t(-5.0), t(-1.0), t(10.0))
        assert float(et) >= 0.0

    def test_increases_with_radiation(self):
        e1 = et0_hargreaves(t(15.0), t(30.0), t(20.0))
        e2 = et0_hargreaves(t(15.0), t(30.0), t(35.0))
        assert e2 > e1

    def test_zero_when_tmax_le_tmin(self):
        assert float(et0_hargreaves(t(20.0), t(20.0), t(30.0))) == 0.0

    def test_typical_summer_magnitude(self):
        et = et0_hargreaves(t(16.0), t(28.0), t(25.0))
        assert 4.0 < float(et) < 9.0


class TestPenmanMonteith:
    """FAO-56 PM — the reference ET0 method."""

    def pm(self, **keywords):
        arguments = dict(tmin=t(16.0), tmax=t(30.0), rh_mean=t(70.0), u2=t(2.5),
                    srad=t(28.0), ra=t(40.0), elev_m=t(200.0))
        arguments.update(keywords)
        return et0_penman_monteith(arguments["tmin"], arguments["tmax"], arguments["rh_mean"],
                                   arguments["u2"], arguments["srad"], arguments["ra"],
                                   arguments["elev_m"])

    def test_svp_reference_values(self):
        """e° at 0/20/40 °C — textbook saturation vapour pressure."""
        assert float(saturation_vapor_pressure(t(0.0))) == pytest.approx(0.611, abs=1e-3)
        assert float(saturation_vapor_pressure(t(20.0))) == pytest.approx(2.338, abs=1e-2)
        assert float(saturation_vapor_pressure(t(40.0))) == pytest.approx(7.376, abs=1e-2)

    def test_psychrometric_constant_sea_level(self):
        """γ = 0.665e-3 × 101.3 kPa = 0.0674 at sea level (FAO-56 eq 8)."""
        assert float(psychrometric_constant(t(0.0))) == pytest.approx(0.0674, abs=2e-4)
        assert float(psychrometric_constant(t(2000.0))) < \
            float(psychrometric_constant(t(0.0)))

    def test_iowa_summer_magnitude(self):
        """Hand-checked ≈5.6 mm/d for humid midsummer conditions."""
        assert float(self.pm()) == pytest.approx(5.6, relative=0.15)

    def test_increases_with_wind(self):
        assert float(self.pm(u2=t(0.5))) < float(self.pm(u2=t(4.0)))

    def test_increases_with_radiation(self):
        assert float(self.pm(srad=t(15.0))) < float(self.pm(srad=t(30.0)))

    def test_decreases_with_humidity(self):
        assert float(self.pm(rh_mean=t(95.0))) < float(self.pm(rh_mean=t(40.0)))

    def test_below_hargreaves_at_high_radiation(self):
        """Hargreaves overestimates ~2× at high Ra — the bug PM fixes."""
        hg = et0_hargreaves(t(16.0), t(30.0), t(40.0))
        assert float(self.pm()) < float(hg)

    def test_nonnegative(self):
        assert float(self.pm(u2=t(0.2), rh_mean=t(100.0), srad=t(1.0))) >= 0.0


class TestGDD:
    def test_below_base_zero(self):
        assert float(growing_degree_days(t(0.0), t(4.0), t(8.0))) == 0.0

    def test_above_base(self):
        assert float(growing_degree_days(t(10.0), t(20.0), t(8.0))) == 7.0


class TestDayLength:
    def test_equator_twelve_hours(self):
        assert float(day_length(t(0.0), t(80.0))) == pytest.approx(12.0, abs=0.3)

    def test_nh_summer_longer_than_winter(self):
        assert day_length(t(45.0), t(172.0)) > day_length(t(45.0), t(355.0))

    def test_polar_day_and_night(self):
        assert float(day_length(t(80.0), t(172.0))) == pytest.approx(24.0)
        assert float(day_length(t(80.0), t(355.0))) == pytest.approx(0.0)

    def test_southern_inverted(self):
        assert day_length(t(-45.0), t(355.0)) > day_length(t(-45.0), t(172.0))


class TestTemperatureFactor:
    def test_zero_below_base(self):
        f = temperature_factor(t(0.0), t(5.0), t(8.0), t(30.0))
        assert float(f) == 0.0

    def test_one_at_optimum(self):
        f = temperature_factor(t(27.0), t(33.0), t(8.0), t(30.0))
        assert float(f) == pytest.approx(1.0)

    def test_declines_above_34(self):
        f_mid = temperature_factor(t(37.0), t(42.0), t(8.0), t(30.0))
        f_hot = temperature_factor(t(44.0), t(46.0), t(8.0), t(30.0))
        assert 0.0 < float(f_mid) < 1.0
        assert float(f_hot) == 0.0