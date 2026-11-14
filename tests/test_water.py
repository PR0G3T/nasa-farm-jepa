"""Water balance invariants — the theoretical guarantees of the model core."""

import pytest
import torch
from hypothesis import HealthCheck, given, settings, strategies as saved

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.water import (
    evaporation_stress, water_balance_step, water_stress,
)
from nasa_farm.state.soil import root_weights


def t(v, n=1):
    return torch.full((n,), float(v), dtype=DTYPE)


def rw(soil, fraction: float = 0.6):
    return root_weights(t(fraction, soil.batch), soil.n_layers)


def step(sw, soil, infil, evaporation, transpiration, fraction=0.6):
    if sw.dim() == 1:
        sw = sw.unsqueeze(1)
    return water_balance_step(sw, soil, infil, evaporation, transpiration, rw(soil, fraction))


class TestStressFunctions:
    def test_water_stress_bounds(self, soil):
        l = soil.layer(0)
        sw = torch.stack([l.theta_wp, l.theta_wp], dim=1)
        assert float(water_stress(sw, soil)[:, 0].max()) == 0.0
        sw = torch.stack([l.theta_fc, l.theta_fc], dim=1)
        assert float(water_stress(sw, soil)[:, 0].min()) == 1.0

    def test_evap_stress_zero_at_air_dry(self, soil):
        l = soil.layer(0)
        assert float(evaporation_stress(l.theta_air, l.theta_air, l.theta_fc)) == 0.0


class TestWaterBalance:
    def test_exact_conservation(self, soil):
        """Δ storage ≡ inputs − outputs, to machine precision."""
        sw = torch.stack([soil.layer(0).theta_fc.clone(),
                          soil.layer(1).theta_fc.clone()], dim=1)
        infil, evaporation, transpiration = t(25.0), t(4.0), t(3.0)
        s, f = step(sw, soil, infil, evaporation, transpiration)
        in_minus_out = infil - (f["runoff"] + f["es"] + f["t"].sum(-1) + f["deep"])
        assert torch.allclose(s.sum(-1) - sw.sum(-1), in_minus_out, atol=1e-12)

    @given(
        saved.floats(0, 400), saved.floats(0, 15), saved.floats(0, 15),
        saved.floats(0, 1), saved.floats(0, 1),
    )
    @settings(max_examples=200,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_conservation_and_bounds_property(self, soil, infil, evaporation, transpiration, f1, f2):
        """Random inputs: conservation holds, bounds hold, fluxes nonneg."""
        l1, l2 = soil.layer(0), soil.layer(1)
        sw = torch.stack([l1.theta_air + f1 * (l1.theta_sat - l1.theta_air),
                          l2.theta_wp + f2 * (l2.theta_sat - l2.theta_wp)], dim=1)
        s, f = step(sw, soil, t(infil), t(evaporation), t(transpiration))

        in_minus_out = t(infil) - (f["runoff"] + f["es"] + f["t"].sum(-1) + f["deep"])
        assert torch.allclose(s.sum(-1) - sw.sum(-1), in_minus_out, atol=1e-9)
        assert float(s[0, 0]) >= float(l1.theta_air) - 1e-9
        assert float(s[0, 0]) <= float(l1.theta_sat) + 1e-9
        assert float(s[0, 1]) >= 0.0
        assert float(s[0, 1]) <= float(l2.theta_sat) + 1e-9
        assert float(f["es"]) >= -1e-12 and float(f["t"].min()) >= -1e-12
        assert float(f["runoff"]) >= -1e-12 and float(f["deep"]) >= -1e-12

    def test_conservation_at_4_layers(self):
        """N-layer generalization keeps the same identity."""
        soil4 = __import__("nasa_farm.state.soil", fromlist=["SoilProfile"]) \
            .SoilProfile.preset("loam", depths=(0.1, 0.3, 0.6, 0.8))
        sw = (soil4.wp + 0.8 * (soil4.fc - soil4.wp))
        s, f = step(sw, soil4, t(30.0), t(5.0), t(4.0))
        in_minus_out = t(30.0) - (f["runoff"] + f["es"] + f["t"].sum(-1) + f["deep"])
        assert torch.allclose(s.sum(-1) - sw.sum(-1), in_minus_out, atol=1e-9)
        assert (s >= -1e-9).all()

    def test_no_flux_when_dry(self, soil):
        """At lower bounds, nothing evaporates, transpires or drains."""
        sw = torch.stack([soil.layer(0).theta_air.clone(),
                          soil.layer(1).theta_wp.clone()], dim=1)
        _, f = step(sw, soil, t(0.0), t(6.0), t(6.0), fraction=0.5)
        for k in ("es", "perc", "deep"):
            assert float(f[k]) == 0.0, k
        assert float(f["t"].sum()) == 0.0

    def test_runoff_only_beyond_saturation(self, soil):
        l1 = soil.layer(0)
        sw1 = l1.theta_fc.clone()
        small = (l1.theta_sat - l1.theta_fc) * 0.5
        sw = torch.stack([sw1, soil.layer(1).theta_fc.clone()], dim=1)
        _, f_small = step(sw, soil, small, t(0.0), t(0.0), fraction=0.5)
        assert float(f_small["runoff"]) == 0.0
        _, f_big = step(sw, soil, t(2000.0), t(0.0), t(0.0), fraction=0.5)
        assert float(f_big["runoff"]) > 0.0

    def test_drainage_only_above_field_capacity(self, soil):
        l1 = soil.layer(0)
        below = l1.theta_wp + 0.5 * (l1.theta_fc - l1.theta_wp)
        sw = torch.stack([below, soil.layer(1).theta_fc.clone()], dim=1)
        _, f = step(sw, soil, t(0.0), t(0.0), t(0.0), fraction=0.5)
        assert float(f["perc"]) == 0.0
        above = l1.theta_fc + 0.5 * (l1.theta_sat - l1.theta_fc)
        sw = torch.stack([above, soil.layer(1).theta_fc.clone()], dim=1)
        _, f2 = step(sw, soil, t(0.0), t(0.0), t(0.0), fraction=0.5)
        assert float(f2["perc"]) > 0.0

    def test_layer1_receives_percolation_layer2(self, soil):
        """Layer-2 storage rises exactly by layer-1 percolation minus its own drainage."""
        sw = torch.stack([soil.layer(0).theta_sat.clone(),
                          soil.layer(1).theta_wp.clone()], dim=1)
        s, f = step(sw, soil, t(0.0), t(0.0), t(0.0), fraction=1.0)
        gained = float(s[0, 1] - sw[0, 1])
        assert gained == pytest.approx(float(f["perc"] - f["deep"]))
