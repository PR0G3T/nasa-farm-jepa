"""Water balance invariants — the theoretical guarantees of the model core."""

import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.water import (
    evaporation_stress, water_balance_step, water_stress,
)


def t(v, n=1):
    return torch.full((n,), float(v), dtype=DTYPE)


class TestStressFunctions:
    def test_water_stress_bounds(self, soil):
        l = soil.layer1
        assert float(water_stress(l.theta_wp, l.theta_wp, l.theta_fc)) == 0.0
        assert float(water_stress(l.theta_fc, l.theta_wp, l.theta_fc)) == 1.0
        assert float(water_stress(l.theta_sat, l.theta_wp, l.theta_fc)) == 1.0
        assert float(water_stress(t(0.0), l.theta_wp, l.theta_fc)) == 0.0

    def test_evap_stress_zero_at_air_dry(self, soil):
        l = soil.layer1
        assert float(evaporation_stress(l.theta_air, l.theta_air, l.theta_fc)) == 0.0


class TestWaterBalance:
    def test_exact_conservation(self, soil):
        """Δ storage ≡ inputs − outputs, to machine precision."""
        sw1, sw2 = soil.layer1.theta_fc.clone(), soil.layer2.theta_fc.clone()
        infil, ep, tp = t(25.0), t(4.0), t(3.0)
        s1, s2, f = water_balance_step(sw1, sw2, soil, infil, ep, tp, t(0.6))
        in_minus_out = infil - (f["runoff"] + f["es"] + f["t1"] + f["t2"] + f["deep"])
        delta_store = (s1 + s2) - (sw1 + sw2)
        assert torch.allclose(delta_store, in_minus_out, atol=1e-12)

    @given(
        st.floats(0, 400), st.floats(0, 15), st.floats(0, 15),
        st.floats(0, 1), st.floats(0, 1),
    )
    @settings(max_examples=200,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_conservation_and_bounds_property(self, soil, infil, ep, tp, f1, f2):
        """Random inputs: conservation holds, bounds hold, fluxes nonneg."""
        l1, l2 = soil.layer1, soil.layer2
        sw1 = l1.theta_air + f1 * (l1.theta_sat - l1.theta_air)
        sw2 = l2.theta_wp + f2 * (l2.theta_sat - l2.theta_wp)
        s1, s2, f = water_balance_step(
            sw1, sw2, soil, t(infil), t(ep), t(tp), t(0.6))

        in_minus_out = t(infil) - (f["runoff"] + f["es"] + f["t1"] + f["t2"] + f["deep"])
        assert torch.allclose((s1 + s2) - (sw1 + sw2), in_minus_out, atol=1e-9)
        assert float(s1) >= float(l1.theta_air) - 1e-9
        assert float(s1) <= float(l1.theta_sat) + 1e-9
        assert float(s2) >= float(l2.theta_wp) - 1e-9
        assert float(s2) <= float(l2.theta_sat) + 1e-9
        for k in ("es", "t1", "t2", "runoff", "perc", "deep"):
            assert float(f[k]) >= -1e-12, k

    def test_no_flux_when_dry(self, soil):
        """At lower bounds, nothing evaporates, transpires or drains."""
        sw1, sw2 = soil.layer1.theta_air.clone(), soil.layer2.theta_wp.clone()
        s1, s2, f = water_balance_step(sw1, sw2, soil, t(0.0), t(6.0), t(6.0), t(0.5))
        for k in ("es", "t1", "t2", "perc", "deep"):
            assert float(f[k]) == 0.0, k

    def test_runoff_only_beyond_saturation(self, soil):
        l1 = soil.layer1
        sw1 = l1.theta_fc.clone()
        small = (l1.theta_sat - l1.theta_fc) * 0.5
        _, _, f_small = water_balance_step(sw1, soil.layer2.theta_fc.clone(),
                                           soil, small, t(0.0), t(0.0), t(0.5))
        assert float(f_small["runoff"]) == 0.0

        big = t(2000.0)
        _, _, f_big = water_balance_step(sw1, soil.layer2.theta_fc.clone(),
                                         soil, big, t(0.0), t(0.0), t(0.5))
        assert float(f_big["runoff"]) > 0.0

    def test_drainage_only_above_field_capacity(self, soil):
        l1 = soil.layer1
        below = l1.theta_wp + 0.5 * (l1.theta_fc - l1.theta_wp)
        _, _, f = water_balance_step(below, soil.layer2.theta_fc.clone(),
                                     soil, t(0.0), t(0.0), t(0.0), t(0.5))
        assert float(f["perc"]) == 0.0

        above = l1.theta_fc + 0.5 * (l1.theta_sat - l1.theta_fc)
        _, _, f2 = water_balance_step(above, soil.layer2.theta_fc.clone(),
                                      soil, t(0.0), t(0.0), t(0.0), t(0.5))
        assert float(f2["perc"]) > 0.0

    def test_layer1_receives_percolation_layer2(self, soil):
        """Layer-2 storage rises exactly by layer-1 percolation (no T2)."""
        above1 = soil.layer1.theta_sat.clone()
        low2 = soil.layer2.theta_wp.clone()
        _, s2, f = water_balance_step(above1, low2, soil, t(0.0), t(0.0), t(0.0), t(1.0))
        gained = float(s2 - low2)
        assert gained == pytest.approx(float(f["perc"] - f["deep"]))
