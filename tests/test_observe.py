"""Observation operators — physical sanity, differentiability, trace cadence."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.forcing import synthetic_year, tile_to_horizon
from nasa_farm.observe import CHANNELS, observables, sm_vol, vwc
from nasa_farm.observe.optical import fapar, ndvi
from nasa_farm.observe.radiometer import tb
from nasa_farm.observe.sar import sigma0
from nasa_farm.observe.trace import ObsTrace
from nasa_farm.state.field import Field
from nasa_farm.types import Actions


def t(v):
    return torch.tensor([v], dtype=DTYPE)


class TestOptical:
    def test_ndvi_bare_is_soil_background(self):
        assert float(ndvi(t(0.0))) == pytest.approx(0.15)

    def test_ndvi_monotonic_saturating(self):
        lai = torch.linspace(0, 10, 50, dtype=DTYPE)
        v = ndvi(lai)
        assert (v[1:] >= v[:-1]).all() and float(v.max()) < 0.92

    def test_fapar_bounded(self):
        lai = torch.linspace(0, 20, 50, dtype=DTYPE)
        f = fapar(lai)
        assert (f >= 0).all() and (f < 1).all()


class TestSAR:
    def test_wetter_soil_brighter_backscatter(self):
        assert float(sigma0(t(0.4), t(0.5))) > float(sigma0(t(0.05), t(0.5)))

    def test_finite_in_operational_range(self):
        mv = torch.linspace(0.02, 0.45, 20, dtype=DTYPE)
        vwc = torch.linspace(0.0, 5.0, 20, dtype=DTYPE)
        assert torch.isfinite(sigma0(mv, vwc)).all()
        assert torch.isfinite(sigma0(mv, vwc, band="c")).all()

    def test_canopy_attenuates_soil_signal(self):
        """Under a thick canopy, a soil-moisture step shows less σ⁰ change."""
        d_bare = float(sigma0(t(0.40), t(0.0)) - sigma0(t(0.05), t(0.0)))
        d_veg = float(sigma0(t(0.40), t(4.0)) - sigma0(t(0.05), t(4.0)))
        assert d_veg < d_bare


class TestRadiometer:
    def test_tb_falls_with_soil_moisture(self):
        """Wetter soil → higher dielectric → higher reflectivity → lower
        emissivity → lower brightness temperature (SMAP observable)."""
        wet, dry = tb(t(0.35), t(0.0)), tb(t(0.05), t(0.0))
        assert float(wet[0]) < float(dry[0])
        assert float(wet[1]) < float(dry[1])

    def test_bounds_within_surface_temperature(self):
        mv = torch.linspace(0.0, 0.5, 20, dtype=DTYPE)
        for ch in tb(mv, t(0.0), ts=300.0):
            assert (ch > 0).all() and (ch < 300.0).all()

    def test_vegetation_damps_sensitivity(self):
        d_bare = float(tb(t(0.4), t(0.0))[0] - tb(t(0.05), t(0.0))[0])
        d_veg = float(tb(t(0.4), t(5.0))[0] - tb(t(0.05), t(5.0))[0])
        assert abs(d_veg) < abs(d_bare)


class TestObservables:
    def test_stack_shape_and_channels(self, field):
        obs = observables(field.state, ("ndvi", "sm1", "s0v"))
        assert obs.shape == (field.batch, 3)
        assert torch.isfinite(obs).all()

    def test_all_registered_channels_evaluable(self, field):
        obs = observables(field.state)
        assert obs.shape == (field.batch, len(CHANNELS))
        assert torch.isfinite(obs).all()


class TestTrace:
    def test_records_every_k_days(self, physics_model, field):
        forcing = tile_to_horizon(synthetic_year(42.0, seed=1), 1)
        tr = ObsTrace(forcing.days, ("ndvi", "sm1"), every=5)
        physics_model.rollout(field, forcing,
                              Actions.idle(1, forcing.days), observer=tr)
        r = tr.result()
        assert r.shape == (1, 73, 2)
        assert float(r[0, 0, 0]) == pytest.approx(0.15)  # bare soil NDVI

    def test_unknown_channel_raises(self):
        with pytest.raises(KeyError):
            ObsTrace(10, ("bogus",))

    def test_gradients_flow_through_rollout(self, hybrid_model, field):
        """Predicted observables keep the graph — assimilation can
        differentiate series residuals back into dynamics parameters."""
        forcing = tile_to_horizon(synthetic_year(42.0, seed=1), 1)
        tr = ObsTrace(forcing.days, ("ndvi", "sm1"), every=30)
        hybrid_model.rollout(field, forcing,
                             Actions.idle(1, forcing.days), observer=tr)
        r = tr.result()
        r.sum().backward()
        grads = [p.grad for p in hybrid_model.residual.parameters()
                 if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)


class TestDerivedChannels:
    def test_sm_vol_units(self, field):
        """sw1 mm over 0.3 m → m3/m3."""
        s = field.state.replace(
            sw=torch.stack([t(120.0), t(300.0)], dim=1))
        assert float(sm_vol(s)) == pytest.approx(0.4)

    def test_vwc_scales_with_biomass(self, field):
        s0, s1 = field.state, field.state.replace(biomass=t(10_000.0))
        assert float(vwc(s1)) > float(vwc(s0))
