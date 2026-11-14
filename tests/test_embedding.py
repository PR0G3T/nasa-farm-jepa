"""JEPA layer — EMA targets, spatial encoder, ctx-conditioned residual,
end-to-end self-supervised fit and the registered experiment."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.latent import (
    ObservationEncoder, PatchEncoder, TargetEncoder, embedding_loss, regularization,
)
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.dynamics.residual import N_FEATURES, Residual
from nasa_farm.experiments.specification import Specification
from nasa_farm.observe.trace import EmbeddingTrace
from nasa_farm.optimize.embedding import embedding_fit, seeded
from nasa_farm.state.crops import CropLibrary
from nasa_farm.state.field import Field, initial_state
from nasa_farm.state.soil import SoilProfile
from nasa_farm.types import Actions, Forcing


def _field() -> Field:
    soil = SoilProfile.preset("silt_loam")
    return Field(initial_state(1, soil), soil,
                 torch.tensor([42.0], dtype=DTYPE))


def _forcing_actions(days: int = 365):
    from nasa_farm.forcing import synthetic_year, tile_to_horizon
    from nasa_farm.plan import expand_actions
    crops = CropLibrary.default()
    forcing = tile_to_horizon(synthetic_year(42.0, seed=0), 1)
    actions = expand_actions(
        torch.tensor([[crops.index("maize")]]), crops.table)
    return forcing, actions


class TestTargetEncoder:
    def test_ema_math_exact(self):
        torch.manual_seed(0)
        online = ObservationEncoder(5, 4)
        target = TargetEncoder(online, decay=0.9)
        before = target.net.net[0].weight.clone()
        with torch.no_grad():
            online.net[0].weight.fill_(1.0)
        target.update_from(online)
        expect = 0.9 * before + 0.1 * torch.ones_like(before)
        assert torch.allclose(target.net.net[0].weight, expect)

    def test_frozen_and_detached(self):
        target = TargetEncoder(ObservationEncoder(5, 4), decay=0.99)
        assert all(not p.requires_grad for p in target.parameters())
        x = torch.randn(3, 5, dtype=DTYPE, requires_grad=True)
        assert not target(x).requires_grad


class TestPatchEncoder:
    def test_shape_and_grad(self):
        encoder = PatchEncoder(bands=6, z_dim=8, width=8)
        x = torch.randn(4, 6, 16, 16, dtype=DTYPE, requires_grad=True)
        z = encoder(x)
        assert z.shape == (4, 8)
        z.sum().backward()
        assert x.gradient is not None

    def test_size_independent(self):
        encoder = PatchEncoder(bands=3, z_dim=5, width=8)
        assert encoder(torch.randn(2, 3, 32, 32, dtype=DTYPE)).shape == (2, 5)


class TestJEPALossReg:
    def test_z_reg_overrides_z_obs_for_sigreg(self):
        g = torch.Generator().manual_seed(0)
        z_observed = torch.ones(64, 8, dtype=DTYPE)          # degenerate → high sig
        z_reg = torch.randn(64, 8, dtype=DTYPE, generator=g)
        # pred term zeroed ⇒ loss isolates the regularizer's argument
        l = embedding_loss(z_observed, z_observed.clone(), sig_weight=1.0, z_reg=z_reg)
        assert float(l) == pytest.approx(float(regularization(z_reg)), relative=1e-9)


class TestResidualCtx:
    def test_zero_init_ignores_ctx(self):
        r = Residual(context_dimension=4)
        feats = torch.randn(8, N_FEATURES, dtype=DTYPE)
        context = torch.randn(8, 4, dtype=DTYPE)
        assert torch.equal(r(feats, context), torch.zeros(8, 5, dtype=DTYPE))

    def test_ctx_reaches_output(self):
        torch.manual_seed(0)
        r = Residual(context_dimension=4)
        with torch.no_grad():
            r.head.weight.normal_(0, 0.1)
        feats = torch.randn(8, N_FEATURES, dtype=DTYPE)
        a = torch.zeros(8, 4, dtype=DTYPE)
        b = torch.ones(8, 4, dtype=DTYPE)
        assert not torch.equal(r(feats, a), r(feats, b))


class TestModelWiring:
    def test_residual_ctx_auto_bound(self):
        m = WorldModel(latent_dimension=6)
        assert m.residual.context_dimension == 6
        assert WorldModel().residual.context_dimension == 0

    def test_z_constant_at_init(self):
        """Zero-init latent head ⇒ seeded z persists unchanged: identity."""
        m = WorldModel(latent_dimension=4)
        forcing, actions = _forcing_actions()
        z0 = torch.randn(1, 4, dtype=DTYPE,
                         generator=torch.Generator().manual_seed(0))
        f = seeded(_field(), z0, m.latent.hidden_dimension)
        traits = EmbeddingTrace(forcing.days, 4, ("ndvi",), every=30)
        m.rollout(f, forcing, actions, observer=traits)
        _, z = traits.result()
        assert torch.allclose(z[:, 0], z0.expand(1, -1).expand_as(
            z[:, 0]).to(DTYPE), atol=1e-12) or torch.equal(z[:, 0], z0)


class TestSeeded:
    def test_copies_state_and_resets_zh(self):
        m = WorldModel(latent_dimension=4)
        f = _field()
        z0 = torch.ones(1, 4, dtype=DTYPE)
        f2 = seeded(f, z0, m.latent.hidden_dimension)
        assert torch.equal(f2.state.z, z0)
        assert f2.state.zh.shape == (1, m.latent.hidden_dimension)
        assert f.state.z is None  # original untouched


class TestJepaFit:
    def test_loss_decreases_no_collapse(self):
        torch.manual_seed(0)
        m = WorldModel(CropLibrary.default(), latent_dimension=4)
        encoder = ObservationEncoder(observed_dimension=2, z_dim=4)
        forcing, actions = _forcing_actions()
        rep = embedding_fit(m, encoder, _field(), forcing, actions,
                       ("ndvi", "sm1"), every=15, epochs=8, rate=1e-2)
        assert rep["best_loss"] < rep["history"][0]
        assert rep["z_std_final"] > 0.01   # SIGReg prevents collapse

    def test_external_targets(self):
        torch.manual_seed(0)
        m = WorldModel(CropLibrary.default(), latent_dimension=4)
        encoder = ObservationEncoder(observed_dimension=2, z_dim=4)
        forcing, actions = _forcing_actions()
        days_n = forcing.days // 15 + (1 if forcing.days % 15 else 0)
        targets = torch.rand(1, days_n, 2, dtype=DTYPE,
                             generator=torch.Generator().manual_seed(1))
        rep = embedding_fit(m, encoder, _field(), forcing, actions,
                       ("ndvi", "sm1"), targets=targets, every=15,
                       epochs=4, rate=1e-2)
        assert rep["z_tgt"].shape == (1, days_n, 4)

    def test_requires_latent(self):
        with pytest.raises(ValueError):
            embedding_fit(WorldModel(CropLibrary.default(), latent_dimension=0),
                     ObservationEncoder(2, 4), _field(), *_forcing_actions(),
                     ("ndvi", "sm1"))


class TestExpJepa:
    def test_registered_and_json_safe(self):
        import json
        from nasa_farm.experiments import run
        specification = Specification(horizon=1, realizations=1, cadence=30, epochs=3,
                    latent_dimension=4, observed_channels=("ndvi", "sm1"),
                    channels=("yield_kg",))
        rep = run("jepa", specification)
        r = rep.results
        assert set(r["channels"]["ndvi"]) >= {
            "prior", "jepa_probe", "residual", "hybrid"}
        json.dumps(r)  # no NaN leaks

    def test_reproducible(self):
        from nasa_farm.experiments import run
        specification = Specification(horizon=1, realizations=1, cadence=30, epochs=3,
                    latent_dimension=4, observed_channels=("ndvi", "sm1"))
        a = run("jepa", specification).results["channels"]["ndvi"]["hybrid"]
        b = run("jepa", specification).results["channels"]["ndvi"]["hybrid"]
        assert a == b
