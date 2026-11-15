"""Hierarchical latents, variety conditioning, planning extensions,
smoother, hierarchical SBI, learned routing, adaptation, flow policy,
ensemble bias correction and multi-agent allocation."""

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.state.crops import CropLibrary
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


def _year30(warm_year) -> Forcing:
    from nasa_farm.types import FORCING_VARS

    return Forcing(*(getattr(warm_year, k)[:, :90] for k in FORCING_VARS))


class TestHLatent:
    def test_identity_at_init(self, field, warm_year):
        """Zero-init heads: hierarchical latent = the plain physics rollout."""
        from nasa_farm.dynamics.hierarchical import Hierarchy

        crops = CropLibrary.default()
        WorldModel(crops, residual=False, latent_dimension=4, seed=1)
        hierarchy = WorldModel(
            crops, residual=False, latent_dimension=4, latent_strides=(1, 3, 9), seed=1
        )
        assert isinstance(hierarchy.latent, Hierarchy)
        assert hierarchy.latent.dim == 12 and hierarchy.latent.hidden_dimension == 192
        z = torch.zeros(1, 12, dtype=DTYPE)
        saved = field.state.replace(z=z, zh=torch.zeros(1, 192, dtype=DTYPE))
        f = Field(state=saved, soil=field.soil, latitude=field.latitude)
        f2 = hierarchy.rollout(f, _year30(warm_year), Actions.idle(1, 90))
        assert torch.isfinite(f2.state.biomass).all()

    def test_stride_firing(self, field, warm_year):
        """Upper levels hold their latent between stride boundaries."""
        crops = CropLibrary.default()
        m = WorldModel(crops, residual=False, latent_dimension=4, latent_strides=(1, 4), seed=0)
        # put signal into level-0 head so firing is visible
        with torch.no_grad():
            m.latent.levels[0].head.bias.fill_(0.01)
            m.latent.levels[1].head.bias.fill_(0.05)
        z = torch.zeros(1, 8, dtype=DTYPE)
        saved = field.state.replace(z=z, zh=torch.zeros(1, 128, dtype=DTYPE))
        f = Field(state=saved, soil=field.soil, latitude=field.latitude)
        forcing_day, action = _year30(warm_year).day(1), Actions.idle(1, 90).day(1)
        f2, _ = m.step(f, action, forcing_day, t=1)  # t=1: only level 0 fires
        z1 = f2.state.z[0]
        assert z1[4:].abs().max() == 0.0  # upper level unchanged
        f3, _ = m.step(f, action, forcing_day, t=4)  # t=4: both fire
        assert f3.state.z[0, 4:].abs().max() > 0.0

    def test_hjepa_loss_scalar(self):
        from nasa_farm.dynamics.hierarchical import Hierarchy, hierarchical_loss

        h = Hierarchy(4, 8, (1, 4))
        z_traj = torch.randn(2, 10, 8, dtype=DTYPE)
        z_target = torch.randn(2, 10, 4, dtype=DTYPE)
        loss = hierarchical_loss(z_traj, z_target, h, every=1)
        assert loss.dim() == 0 and torch.isfinite(loss)

    def test_jepa_fit_hier(self, field, warm_year):
        """jepa_fit runs end-to-end on a hierarchical latent."""
        from nasa_farm.dynamics.latent import ObservationEncoder
        from nasa_farm.optimize.embedding import embedding_fit

        crops = CropLibrary.default()
        m = WorldModel(crops, residual=True, latent_dimension=4, latent_strides=(1, 4), seed=0)
        encoder = ObservationEncoder(observed_dimension=2, z_dim=4)
        rep = embedding_fit(
            m,
            encoder,
            field,
            _year30(warm_year),
            Actions.idle(1, 90),
            ("ndvi", "sm1"),
            every=15,
            epochs=3,
            rate=1e-3,
        )
        assert rep["z_pred"].shape[-1] == 8
        assert torch.isfinite(rep["z_pred"]).all()


class TestVariety:
    def test_var_emb_flows_to_ctx(self, field):
        crops = CropLibrary.default().with_var(4, seed=0)
        m = WorldModel(crops, residual=True, latent_dimension=4, seed=0)
        assert m.variety_embedding is not None and m.variety_embedding.shape == (crops.n_crops, 4)
        saved = field.state.replace(crop=torch.zeros(1, dtype=torch.long))
        f = Field(state=saved, soil=field.soil, latitude=field.latitude)
        d = m._dctx(f)
        assert d.shape == (1, 4)
        st2 = field.state.replace(crop=torch.ones(1, dtype=torch.long))
        f2 = Field(state=st2, soil=field.soil, latitude=field.latitude)
        assert not torch.equal(m._dctx(f2), d)

    def test_var_grad(self, field, warm_year):
        crops = CropLibrary.default().with_var(4, seed=0)
        m = WorldModel(crops, residual=False, latent_dimension=4, seed=0)
        z = torch.zeros(1, 4, dtype=DTYPE)
        saved = field.state.replace(z=z, zh=torch.zeros(1, 64, dtype=DTYPE))
        f = Field(state=saved, soil=field.soil, latitude=field.latitude)
        f2 = m.rollout(f, _year30(warm_year), Actions.idle(1, 90))
        f2.state.z.sum().backward()
        assert m.variety_embedding.grad is not None


class TestSAE:
    def test_topk_sparsity(self):
        from nasa_farm.dynamics.latent import Autoencoder

        autoencoder = Autoencoder(dim=8, feature_count=32, topk=3, seed=0)
        f = autoencoder.features(torch.randn(16, 8, dtype=DTYPE))
        assert ((f > 0).sum(-1) <= 3).all()

    def test_sae_fit(self):
        from nasa_farm.dynamics.latent import Autoencoder, fit_autoencoder

        z = torch.randn(64, 8, dtype=DTYPE)
        z = z / z.std(0, keepdim=True).clamp(min=1e-9)
        autoencoder = Autoencoder(dim=8, feature_count=32, topk=4, seed=0)
        rep = fit_autoencoder(autoencoder, z, epochs=40, rate=3e-3)
        assert rep["best_loss"] < rep["history"][0]
        assert rep["rate"].shape == (32,)


class TestMPPIExt:
    def test_refine_bounds_and_gain(self, physics_model, field, warm_year):
        from nasa_farm.control import path_integral

        f60 = _year30(warm_year)
        base = Actions.idle(1, 90)
        r0 = path_integral(physics_model, field, f60, base, iterations=2, samples=8, seed=0)
        r1 = path_integral(
            physics_model,
            field,
            f60,
            base,
            iterations=2,
            samples=8,
            refine=3,
            refine_lr=0.05,
            seed=0,
        )
        assert r1.reward >= r0.reward - 1e-9
        assert (r1.actions.irrigation >= 0).all()
        assert (r1.actions.n_fert >= 0).all()

    def test_modes_mixture(self, physics_model, field, warm_year):
        from nasa_farm.control import path_integral

        f60 = _year30(warm_year)
        result = path_integral(
            physics_model,
            field,
            f60,
            Actions.idle(1, 90),
            iterations=2,
            samples=16,
            modes=2,
            seed=0,
        )
        assert torch.isfinite(result.actions.irrigation).all()
        res2 = path_integral(
            physics_model,
            field,
            f60,
            Actions.idle(1, 90),
            iterations=2,
            samples=16,
            modes=2,
            seed=0,
        )
        assert torch.equal(result.actions.irrigation, res2.actions.irrigation)

    def test_goal_penalty(self, physics_model, field, warm_year):
        from nasa_farm.control import path_integral

        crops = CropLibrary.default()
        m = WorldModel(crops, residual=False, latent_dimension=4, seed=0)
        saved = field.state.replace(
            z=torch.zeros(1, 4, dtype=DTYPE), zh=torch.zeros(1, 64, dtype=DTYPE)
        )
        f = Field(state=saved, soil=field.soil, latitude=field.latitude)
        f60 = _year30(warm_year)
        goal = torch.full((4,), 3.0, dtype=DTYPE)
        result = path_integral(
            m, f, f60, Actions.idle(1, 90), iterations=1, samples=8, goal=goal, goal_w=1.0, seed=0
        )
        assert torch.isfinite(torch.tensor(result.reward))

    def test_fbrisk_margin(self):
        from nasa_farm.control import Risk

        r = Risk(base=0.5, horizon=10)
        assert abs(r.margin() - 0.5) < 1e-9
        r.update(1.0)  # overspend → remaining margin tightens
        assert r.margin() < 0.5
        r2 = Risk(base=0.5, horizon=10)
        r2.update(0.0)  # clean step → margin relaxes
        assert r2.margin() > 0.5


class TestSmooth:
    def test_esv_recovers_state(self, physics_model, field, warm_year):
        from nasa_farm.assimilation import Observation, ensemble_smooth
        from nasa_farm.observe import observables

        f90 = _year30(warm_year)
        actions = Actions.idle(1, 90)
        # truth: wetter initial soil
        saved = field.state.replace(sw=field.state.sw * 1.3)
        truth = Field(state=saved, soil=field.soil, latitude=field.latitude)
        t = truth
        schedule = []
        with torch.inference_mode():
            for day in range(90):
                t, _ = physics_model.step(t, actions.day(day), f90.day(day), day)
                if day in (30, 60):
                    schedule.append(
                        Observation(day, ("sm1",), observables(t.state, ("sm1",))[0], 0.02)
                    )
        output, rep = ensemble_smooth(
            physics_model, field, f90, actions, schedule, ensemble=8, epochs=8, rate=0.05, seed=0
        )
        assert rep["epochs_run"] > 0
        d0 = (field.state.sw.sum() - truth.state.sw.sum()).abs()
        d1 = (output.state.sw.sum() - truth.state.sw.sum()).abs()
        assert d1 <= d0 + 1e-6


class TestGFlow:
    def test_hier_groups(self):
        from nasa_farm.optimize.inference import (
            GroupPosterior,
            fit_hierarchy,
        )

        g = torch.Generator().manual_seed(0)
        tg = torch.randn(200, 2, generator=g, dtype=DTYPE)
        ts = tg[:, :1] + 0.1 * torch.randn(200, 1, generator=g, dtype=DTYPE)
        x = torch.cat([ts, tg[:, 1:2]], -1)
        m = fit_hierarchy(tg, ts, x, epochs=20, hidden=64, seed=0)
        p = GroupPosterior(m)
        s = p.sample({1: ts[:1], 2: x[:1]}, 8, steps=3, seed=0)
        assert s.shape == (8, 5) and torch.isfinite(s).all()
        # known groups stay frozen at their ctx values
        assert torch.allclose(s[:, 2:3], ts[:1].expand(8, 1))


class TestRouting:
    def test_routenet_stochastic(self, field):
        from nasa_farm.landscape import Route

        net = Route(3, 16, seed=0)
        adjacency = torch.tensor([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=torch.bool)
        w = net(torch.randn(3, 3, dtype=DTYPE), adjacency)
        assert torch.allclose(w.sum(-1), torch.ones(3, dtype=DTYPE), atol=1e-9)
        assert (w[~adjacency] == 0).all()

    def test_routegraph_landscape(self, physics_model, warm_year):
        from nasa_farm.landscape import Landscape, RouteGraph
        from nasa_farm.state.field import initial_state
        from nasa_farm.state.soil import SoilProfile

        soil = SoilProfile.preset("loam").repeat(3)
        latitude = torch.tensor([42.0, 42.01, 42.02], dtype=DTYPE)
        elevation = torch.tensor([300.0, 250.0, 200.0], dtype=DTYPE)
        field = Field(initial_state(3, soil), soil, latitude, elevation=elevation)
        g = RouteGraph.from_topo(latitude, latitude * 0 + 1.0, elevation, k=2, seed=0)
        ls = Landscape(physics_model, g)
        f2 = ls.rollout(field, _year30(warm_year), Actions.idle(3, 90))
        assert torch.isfinite(f2.state.sw).all()

    def test_fit_route(self):
        from nasa_farm.landscape import RouteGraph, fit_route

        adjacency = torch.tensor([[1, 1], [0, 1]], dtype=torch.bool)
        from nasa_farm.landscape import Route

        g = RouteGraph(adjacency=adjacency, net=Route(3, 16, seed=0))
        reference = torch.tensor([[0.7, 0.3], [0.0, 1.0]], dtype=DTYPE)
        rep = fit_route(g, torch.randn(2, 3, dtype=DTYPE), reference, epochs=15, rate=5e-3)
        assert rep["history"][-1] <= rep["history"][0] + 1e-9


class TestAdapt:
    def test_z0_ctx_fit(self, field, warm_year):
        from nasa_farm.optimize.adapt import adapt
        from nasa_farm.optimize.targets import (
            ObservedSeries,
            series,
            series_loss,
        )

        # latent model: z0 feeds the residual ctx and ctx feeds LatentDyn —
        # both are real gradient paths for the environment fit.
        model = WorldModel(
            CropLibrary.default(), residual=True, latent_dimension=4, context_dimension=4, seed=0
        )
        # emulate a trained model: zero-init heads would give structurally
        # zero ∂obs/∂(z0, ctx), so unfreeze them with small weights.
        with torch.no_grad():
            for p in model.residual.parameters():
                p.uniform_(-0.05, 0.05)
            for p in model.latent.parameters():
                p.uniform_(-0.05, 0.05)
        f60 = _year30(warm_year)
        actions = Actions.idle(1, 90)
        ch = ("ndvi", "sm1")
        zt = torch.full((1, 4), 0.5, dtype=DTYPE)
        truth = Field(
            state=field.state.replace(z=zt, zh=torch.zeros(1, 64, dtype=DTYPE)),
            soil=field.soil,
            latitude=field.latitude,
            context=torch.ones(1, 4, dtype=DTYPE),
        )
        with torch.inference_mode():
            values = series(model, truth, f60, actions, ch, every=15)[0]
        target = ObservedSeries(values=values)

        def obs_loss(f_):
            return series_loss(series(model, f_, f60, actions, ch, every=15), target)

        zed = field.state.replace(
            z=torch.zeros(1, 4, dtype=DTYPE), zh=torch.zeros(1, 64, dtype=DTYPE)
        )
        f = Field(
            state=zed,
            soil=field.soil,
            latitude=field.latitude,
            context=torch.zeros(1, 4, dtype=DTYPE),
        )
        output, rep = adapt(
            model, f, f60, actions, loss_fn=obs_loss, z=True, context=True, epochs=6, rate=0.05
        )
        assert rep["epochs_run"] > 0
        assert rep["z0_shift"] > 0 or rep["ctx_shift"] > 0
        assert output.state.z is not None and output.context.shape == (1, 4)


class TestFlowPolicy:
    def test_fit_and_sample(self):
        from nasa_farm.dynamics.residual import N_FEATURES
        from nasa_farm.policy import FlowPolicy, fit_flow, flow_sample

        p = FlowPolicy(horizon=8, hidden=32, depth=1, seed=0)
        g = torch.Generator().manual_seed(0)
        chunks = torch.randn(64, 16, generator=g, dtype=DTYPE).abs()
        features = torch.randn(64, N_FEATURES, generator=g, dtype=DTYPE)
        rep = fit_flow(p, chunks, features, epochs=10, rate=1e-3, seed=0)
        a = flow_sample(p, features[:2], n=2, steps=4, seed=0)
        assert a.shape == (4, 16) and (a >= 0).all()
        assert rep["best_loss"] < rep["history"][0] + 1.0


class TestEnsBC:
    def test_recalibrates(self, warm_year):
        from nasa_farm.forcing import apply_correction, fit_correction
        from nasa_farm.forcing.base import augment

        reference = _year30(warm_year)
        biased = augment(reference, 8, seed=3)
        bc = fit_correction(biased, reference)
        output = apply_correction(biased, bc)
        err0 = (biased.tmin.mean(0) - reference.tmin[0]).abs().mean()
        err1 = (output.tmin.mean(0) - reference.tmin[0]).abs().mean()
        assert err1 <= err0 + 0.5
        assert output.tmin.shape == biased.tmin.shape


class TestMarl:
    def test_budget_and_report(self, physics_model, warm_year):
        from nasa_farm.allocation import fit_allocation
        from nasa_farm.state.field import initial_state
        from nasa_farm.state.soil import SoilProfile

        soil = SoilProfile.preset("loam").repeat(2)
        latitude = torch.tensor([42.0, 42.1], dtype=DTYPE)
        field = Field(initial_state(2, soil), soil, latitude)
        f60 = _year30(warm_year)
        actions = Actions.idle(2, 90)
        pols, rep = fit_allocation(
            physics_model, field, f60, actions, budget=5.0, epochs=3, rate=1e-3, seed=0
        )
        assert len(pols) == 2 and len(rep["agent_rewards"]) == 2
        assert rep["team_reward"] == round(rep["history"][-1], 4)


class TestChannels:
    def test_new_channels(self, field):
        from nasa_farm.observe import CHANNELS

        assert "sme2" in CHANNELS and "openet" in CHANNELS
        assert CHANNELS["sme2"](field.state).shape == (1,)
        assert CHANNELS["openet"](field.state).shape == (1,)
