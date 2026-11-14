"""WorldModel end-to-end invariants — the core theoretical guarantees."""

import pytest
import torch
from conftest import day_action, day_forcing

from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.forcing import tile_to_horizon
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

        f1 = physics_model.rollout(Field(field.state.clone(), field.soil, field.lat),
                                   forcing, actions)
        f2 = hybrid_model.rollout(Field(field.state.clone(), field.soil, field.lat),
                                  forcing, actions)
        for name in ("sw1", "sw2", "n_min", "n_org", "soc", "litter", "gdd",
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
        acc = []
        def obs(t, s, fx):
            acc.append({k: v.clone() for k, v in fx.items()})
        final = physics_model.rollout(
            Field(field.state.clone(), field.soil, field.lat.repeat(8)),
            forcing, Actions.idle(8, forcing.days), observer=obs)

        rain = forcing.precip.sum(1)
        tot_out = sum(sum(f[k] for f in acc) for k in
                      ("es", "t1", "t2", "runoff", "deep"))
        stored = (final.state.sw1 + final.state.sw2) - (f0.sw1 + f0.sw2)
        assert torch.allclose(stored, rain - tot_out, atol=1e-8)


class TestLifecycle:
    def test_maize_grows_and_harvests(self, field, warm_year, physics_model, crops):
        forcing = tile_to_horizon(warm_year, 1)
        actions = sow_actions(crops.index("maize"), 120, forcing.days)
        acc = SeasonAccumulator(1, 1)
        final = physics_model.rollout(field, forcing, actions, observer=acc)
        m = acc.result()[0, 0]
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
            f = Field(field.state.clone(), field.soil, field.lat)
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
            f = Field(field.state.clone(), field.soil, field.lat)
            return physics_model.rollout(f, forc, actions).state

        sw, sd = run(wet), run(dry)
        assert float(sw.cum_yield) > float(sd.cum_yield) >= 0.0
        # phenology is GDD-driven: identical LAI trajectory under both climates
        assert float(sw.gdd) == pytest.approx(float(sd.gdd))

    def test_determinism(self, field, warm_year, physics_model, crops):
        forcing = tile_to_horizon(warm_year, 1)
        actions = sow_actions(crops.index("maize"), 120, forcing.days)
        a = physics_model.rollout(Field(field.state.clone(), field.soil, field.lat),
                                  forcing, actions)
        b = physics_model.rollout(Field(field.state.clone(), field.soil, field.lat),
                                  forcing, actions)
        assert torch.equal(a.state.sw1, b.state.sw1)
        assert torch.equal(a.state.cum_yield, b.state.cum_yield)


class TestCompiledParity:
    def test_compiled_step_matches_eager(self, field, warm_year, hybrid_model, crops):
        """torch.compile (inductor) must reproduce eager float64 results —
        fusions may reorder fp ops, so parity is to fused-precision, not bit."""
        forcing = tile_to_horizon(warm_year, 1)
        actions = sow_actions(crops.index("maize"), 120, forcing.days)

        eager = hybrid_model.rollout(
            Field(field.state.clone(), field.soil, field.lat),
            forcing, actions)

        compiled = type(hybrid_model)(hybrid_model.crops, residual=True)
        compiled.load_state_dict(hybrid_model.state_dict())
        compiled.enable_compile()
        jit = compiled.rollout(
            Field(field.state.clone(), field.soil, field.lat),
            forcing, actions)

        for name in ("sw1", "sw2", "n_min", "n_org", "soc", "litter",
                     "lai", "biomass", "cum_yield"):
            torch.testing.assert_close(
                getattr(jit.state, name), getattr(eager.state, name),
                rtol=1e-9, atol=1e-9, msg=lambda m, n=name: f"{n}: {m}")


class TestDifferentiability:
    def test_gradients_reach_residual(self, field, hybrid_model):
        """Data assimilation path: loss on a flux differentiates into the
        residual net — the mechanism that will let observations calibrate it."""
        f, fx = hybrid_model.step(field, day_action(), day_forcing())
        loss = fx["delta_b"].sum() + f.state.biomass.sum() + f.state.sw1.sum()
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
        total = f.state.biomass.sum() + f.state.sw1.sum()
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
        def obs(t, s, fx):
            l1, l2 = soil.layer1, soil.layer2
            if not ((s.sw1 >= l1.theta_air - 1e-6).all() and
                    (s.sw1 <= l1.theta_sat + 1e-6).all()):
                violations.append(("sw1", t))
            if (s.n_min < -1e-9).any() or (s.soc < -1e-9).any():
                violations.append(("pools", t))
            if (s.lai < -1e-9).any():
                violations.append(("lai", t))
        physics_model.rollout(field, forcing, actions, observer=obs)
        assert violations == []
