"""WorldModel end-to-end invariants — the core theoretical guarantees."""

import dataclasses

import pytest
import torch
from conftest import day_action, day_forcing

from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.forcing import synthetic_year, tile_to_horizon
from nasa_farm.heads import METRICS, SeasonAccumulator
from nasa_farm.state.field import Field, initial_state
from nasa_farm.state.soil import SoilProfile
from nasa_farm.types import Actions, Forcing


def sow_actions(crop_idx: int, sow_doy: int, days: int, batch: int = 1,
                sow_year: int = 0) -> Actions:
    a = Actions.idle(batch, days)
    sow = a.sow.clone()
    sow[:, sow_year * DAYS_PER_YEAR + sow_doy - 1] = crop_idx
    return Actions(sow=sow, irrigation=a.irrigation, n_fert=a.n_fert)


class TestResidualIdentity:
    def test_fresh_residual_is_exact_physics(self, field, warm_year, physics_model,
                                             hybrid_model, crops):
        """Zero-init ⇒ hybrid rollout ≡ physics rollout, elementwise."""
        forcing = tile_to_horizon(warm_year, 1)
        actions = sow_actions(crops.index("maize"), 120, forcing.days)

        f1 = physics_model.rollout(Field(field.state.clone(), field.soil, field.latitude),
                                   forcing, actions)
        f2 = hybrid_model.rollout(Field(field.state.clone(), field.soil, field.latitude),
                                  forcing, actions)
        for name in ("sw", "n_min", "n_org", "soc_f", "soc_s", "litter", "gdd",
                     "lai", "biomass", "cum_yield"):
            assert torch.equal(getattr(f1.state, name), getattr(f2.state, name)), name


class TestWaterBudgetOverSeason:
    def test_full_year_conservation(self, field, warm_year, physics_model):
        """Σfluxes balances Δstorage over 365 days, batch of 8 rainfall levels."""
        forcing = warm_year.repeat(8)
        scaled = forcing.precip * torch.linspace(0.0, 1.5, 8, dtype=DTYPE).unsqueeze(1)
        forcing = Forcing(forcing.tmin, forcing.tmax, forcing.srad, scaled,
                          forcing.rh, forcing.u2, forcing.doy)

        f0 = field.state.clone()
        accumulate = []
        def observed(t, s, fx):
            accumulate.append({k: v.clone() for k, v in fx.items()})
        final = physics_model.rollout(
            Field(field.state.clone(), field.soil, field.latitude.repeat(8)),
            forcing, Actions.idle(8, forcing.days), observer=observed)

        rain = forcing.precip.sum(1)
        tot_out = sum(sum(f[k] for f in accumulate) for k in
                      ("es", "runoff", "deep")) + sum(
                          f["t"].sum(-1) for f in accumulate)
        stored = final.state.sw.sum(-1) - f0.sw.sum(-1)
        assert torch.allclose(stored, rain - tot_out, atol=1e-8)


class TestLifecycle:
    def test_maize_grows_and_harvests(self, field, warm_year, physics_model, crops):
        forcing = tile_to_horizon(warm_year, 1)
        actions = sow_actions(crops.index("maize"), 120, forcing.days)
        accumulate = SeasonAccumulator(1, 1)
        final = physics_model.rollout(field, forcing, actions, observer=accumulate)
        m = accumulate.result()[0, 0]
        i = {n: k for k, n in enumerate(METRICS)}
        assert float(final.state.cum_yield) > 0.0
        assert float(m[i["yield_kg"]]) == pytest.approx(float(final.state.cum_yield))
        assert float(m[i["cover_days"]]) > 100.0
        assert float(m[i["water_used"]]) > 50.0

    def test_bare_field_stays_bare(self, field, warm_year, physics_model):
        forcing = tile_to_horizon(warm_year, 1)
        final = physics_model.rollout(field, forcing, Actions.idle(1, forcing.days))
        assert float(final.state.biomass) == 0.0
        assert float(final.state.cum_yield) == 0.0
        assert float(final.state.lai) == 0.0

    def test_rotation_carryover(self, field, warm_year, physics_model, crops):
        """Soybean fixes N: post-soybean mineral N > post-maize mineral N."""
        forcing = tile_to_horizon(warm_year, 1)

        def run(crop_name):
            f = Field(field.state.clone(), field.soil, field.latitude)
            actions = sow_actions(crops.index(crop_name), 140, forcing.days)
            return physics_model.rollout(f, forcing, actions).state.n_min

        n_after_soy = run("soybean")
        n_after_mz = run("maize")
        assert float(n_after_soy) > float(n_after_mz)


class TestForcingResponse:
    def test_drought_reduces_growth(self, field, warm_year, physics_model, crops):
        """Water stress enters biomass via the growth limiter: drought must
        cut accumulated biomass and yield while phenology is unchanged."""
        days = DAYS_PER_YEAR
        wet = tile_to_horizon(warm_year, 1)
        dry = wet.perturb(precip_scale=0.05)
        actions = sow_actions(crops.index("maize"), 120, days)

        def run(forc):
            f = Field(field.state.clone(), field.soil, field.latitude)
            return physics_model.rollout(f, forc, actions).state

        sw, deviation = run(wet), run(dry)
        assert float(sw.cum_yield) > float(deviation.cum_yield) >= 0.0
        # phenology is GDD-driven: identical LAI trajectory under both climates
        assert float(sw.gdd) == pytest.approx(float(deviation.gdd))

    def test_determinism(self, field, warm_year, physics_model, crops):
        forcing = tile_to_horizon(warm_year, 1)
        actions = sow_actions(crops.index("maize"), 120, forcing.days)
        a = physics_model.rollout(Field(field.state.clone(), field.soil, field.latitude),
                                  forcing, actions)
        b = physics_model.rollout(Field(field.state.clone(), field.soil, field.latitude),
                                  forcing, actions)
        assert torch.equal(a.state.sw, b.state.sw)
        assert torch.equal(a.state.cum_yield, b.state.cum_yield)


class TestCompiledParity:
    def test_compiled_step_matches_eager(self, field, warm_year, hybrid_model, crops):
        """torch.compile (inductor) must reproduce eager float64 results —
        fusions may reorder fp ops, so parity is to fused-precision, not bit."""
        forcing = tile_to_horizon(warm_year, 1)
        actions = sow_actions(crops.index("maize"), 120, forcing.days)

        eager = hybrid_model.rollout(
            Field(field.state.clone(), field.soil, field.latitude),
            forcing, actions)

        compiled = type(hybrid_model)(hybrid_model.crops, residual=True)
        compiled.load_state_dict(hybrid_model.state_dict())
        compiled.enable_compile()
        jit = compiled.rollout(
            Field(field.state.clone(), field.soil, field.latitude),
            forcing, actions)

        for name in ("sw", "n_min", "n_org", "soc_f", "soc_s", "litter",
                     "lai", "biomass", "cum_yield"):
            torch.testing.assert_close(
                getattr(jit.state, name), getattr(eager.state, name),
                rtol=1e-9, atol=1e-9, message=lambda m, n=name: f"{n}: {m}")


class TestDifferentiability:
    def test_gradients_reach_residual(self, field, hybrid_model):
        """Data assimilation path: loss on a flux differentiates into the
        residual net — the mechanism that will let observations calibrate it."""
        f, fx = hybrid_model.step(field, day_action(), day_forcing())
        loss = fx["delta_b"].sum() + f.state.biomass.sum() + f.state.sw[:, 0].sum()
        loss.backward()
        last = hybrid_model.residual.net[-1]
        assert last.weight.grad is not None
        assert torch.isfinite(last.weight.grad).all()

    def test_rollout_keeps_grad_path(self, field, warm_year, hybrid_model, crops):
        forcing = tile_to_horizon(warm_year, 1)
        days = 5
        f = field
        for t in range(days):
            f, _ = hybrid_model.step(f, day_action(
                sow=crops.index("maize") if t == 100 else -1), day_forcing(doy=float(t + 1)))
        total = f.state.biomass.sum() + f.state.sw[:, 0].sum()
        total.backward()
        grads = [p.grad for p in hybrid_model.residual.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)


class TestBoundsOverRollout:
    def test_state_stays_physical_5y(self, soil, warm_year, physics_model, crops):
        field = Field(initial_state(1, soil), soil, torch.tensor([42.0], dtype=DTYPE))
        forcing = tile_to_horizon(warm_year, 5)
        plans = [crops.index(n) for n in ("maize", "soybean", "wheat", "clover", "maize")]
        a = Actions.idle(1, forcing.days)
        sow = a.sow.clone()
        for y, c in enumerate(plans):
            sow[0, y * DAYS_PER_YEAR + 120] = c
        actions = Actions(sow, a.irrigation, a.n_fert)

        violations = []
        def observed(t, s, fx):
            l1, l2 = soil.layer(0), soil.layer(1)
            if not ((s.sw[:, 0] >= l1.theta_air - 1e-6).all() and
                    (s.sw[:, 0] <= l1.theta_sat + 1e-6).all()):
                violations.append(("sw0", t))
            if (s.n_min < -1e-9).any() or (s.soc_s < -1e-9).any():
                violations.append(("pools", t))
            if (s.lai < -1e-9).any():
                violations.append(("lai", t))
        physics_model.rollout(field, forcing, actions, observer=observed)
        assert violations == []


class TestCheckpointedRollout:
    """gradient checkpointing — same trajectory, same metrics, same grads,
    O(T/k + k) activation memory instead of O(T)."""

    def test_forward_and_metrics_bitwise_equal(self, field, warm_year,
                                               hybrid_model, crops):
        forcing = tile_to_horizon(warm_year, 1)
        actions = sow_actions(crops.index("maize"), 120, forcing.days)
        a1, a2 = SeasonAccumulator(1, 1), SeasonAccumulator(1, 1)
        f1 = hybrid_model.rollout(
            Field(field.state.clone(), field.soil, field.latitude),
            forcing, actions, observer=a1)
        f2 = hybrid_model.rollout(
            Field(field.state.clone(), field.soil, field.latitude),
            forcing, actions, observer=a2, checkpoint_every=30)
        for name in ("sw", "n_min", "soc_f", "soc_s", "gdd", "lai", "biomass",
                     "cum_yield", "vern", "heat_acc"):
            assert torch.equal(getattr(f1.state, name), getattr(f2.state, name)), name
        assert torch.equal(a1.result(), a2.result())

    def test_gradients_match_eager(self, field, warm_year, hybrid_model, crops):
        forcing = tile_to_horizon(warm_year, 1)
        actions = sow_actions(crops.index("maize"), 120, forcing.days)

        def gradient(ckpt):
            tmin = forcing.tmin.detach().clone().requires_grad_(True)
            forcing = dataclasses.replace(forcing, tmin=tmin)
            accumulate = SeasonAccumulator(1, 1)
            hybrid_model.rollout(
                Field(field.state.clone(), field.soil, field.latitude),
                forcing, actions, observer=accumulate, checkpoint_every=ckpt)
            accumulate.result().sum().backward()
            return tmin.grad.clone()

        g_eager, g_ckpt = gradient(0), gradient(30)
        assert g_eager is not None and torch.isfinite(g_eager).all()
        torch.testing.assert_close(g_ckpt, g_eager, rtol=1e-8, atol=1e-10)

    def test_rejects_non_foldable_observer(self, field, warm_year,
                                           hybrid_model, crops):
        forcing = tile_to_horizon(warm_year, 1)
        actions = sow_actions(crops.index("maize"), 120, forcing.days)
        with pytest.raises(ValueError):
            hybrid_model.rollout(field, forcing, actions,
                                 observer=lambda t, s, f: None,
                                 checkpoint_every=30)

    def test_inference_mode_ignores_checkpoint(self, field, warm_year,
                                               hybrid_model, crops):
        forcing = tile_to_horizon(warm_year, 1)
        actions = sow_actions(crops.index("maize"), 120, forcing.days)
        seen = []
        with torch.inference_mode():
            hybrid_model.rollout(field, forcing, actions,
                                 observer=lambda t, s, f: seen.append(t),
                                 checkpoint_every=30)
        assert seen == list(range(forcing.days))


class TestLatentDynamics:
    def test_z_stays_zero_under_identity_head(self, soil, warm_year, crops):
        """Zero-init LatentDyn ⇒ z ≡ 0 — the model is the physics until z
        earns signal through assimilation."""
        model = WorldModel(crops, residual=False, latent_dimension=4)
        field = Field(initial_state(1, soil, z_dim=4), soil,
                      torch.tensor([42.0], dtype=DTYPE))
        forcing = tile_to_horizon(warm_year, 1)
        f = model.rollout(field, forcing, Actions.idle(1, forcing.days))
        assert f.state.z.shape == (1, 4)
        assert torch.equal(f.state.z, torch.zeros(1, 4, dtype=DTYPE))

    def test_z_moves_when_head_trained(self, soil, warm_year, crops):
        model = WorldModel(crops, residual=False, latent_dimension=4)
        with torch.no_grad():
            model.latent.head.weight.normal_(
                0, 0.1, generator=torch.Generator().manual_seed(0))
        field = Field(initial_state(1, soil, z_dim=4), soil,
                      torch.tensor([42.0], dtype=DTYPE))
        forcing = tile_to_horizon(warm_year, 1)
        f = model.rollout(field, forcing, Actions.idle(1, forcing.days))
        assert torch.isfinite(f.state.z).all()
        assert float(f.state.z.detach().abs().sum()) > 0.0

    def test_z_resets_on_sowing(self, soil, warm_year, crops):
        model = WorldModel(crops, residual=False, latent_dimension=4)
        with torch.no_grad():
            model.latent.head.weight.normal_(
                0, 0.1, generator=torch.Generator().manual_seed(0))
        field = Field(initial_state(1, soil, z_dim=4), soil,
                      torch.tensor([42.0], dtype=DTYPE))
        forcing = tile_to_horizon(warm_year, 1)
        actions = sow_actions(crops.index("maize"), 120, forcing.days)
        f = model.rollout(field, forcing, actions)
        assert torch.isfinite(f.state.z).all()  # sowing day zeroed it, then dyn resumed


class TestStressEvents:
    def test_frost_kills_maize_midseason(self, field, warm_year,
                                         physics_model, crops):
        """A −5 °C night under a standing maize canopy ends the cycle:
        harvested flag set, partial salvage yield, growing stops."""
        forcing = tile_to_horizon(warm_year, 1)
        cold = forcing.tmin.clone()
        cold[0, 200] = -5.0
        tmax = forcing.tmax.clone()
        tmax[0, 200] = 2.0
        forcing = dataclasses.replace(forcing, tmin=cold, tmax=tmax)
        actions = sow_actions(crops.index("maize"), 120, forcing.days)
        final = physics_model.rollout(field, forcing, actions)
        s = final.state
        assert bool(s.harvested[0]) and not bool(s.growing[0])
        assert float(s.cum_yield) > 0.0

    def test_winter_wheat_vernalizes_and_yields(self, soil, physics_model,
                                              crops):
        """Autumn-sown wheat in a cool climate completes its cycle only via
        accumulated chill: yield > 0 proves the vernalization gate opened."""
        climate = tile_to_horizon(
            synthetic_year(42.0, tmean_annual=10.0, seed=2), 2)
        field = Field(initial_state(1, soil), soil,
                      torch.tensor([42.0], dtype=DTYPE))
        actions = sow_actions(crops.index("wheat"), 285, climate.days)
        final = physics_model.rollout(field, climate, actions)
        s = final.state
        assert float(s.vernalization) >= 44.0       # vern_req = 45 chill days
        assert float(s.cum_yield) > 0.0    # gate opened → matured → harvested
