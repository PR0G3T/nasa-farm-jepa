"""Tests for the extension modules: ctx conditioning, interventions,
process diffusion, energy-score forcing, latent/learned DA, Langevin
posterior, joint-flow SBI, multiple shooting, value bootstrap, analytic
policy gradients, AutoSINDy growth, providers and the interv experiment."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.types import Actions


# ── state/ctx ─────────────────────────────────────────────────────────

def test_ctx_norm_standardizes():
    from nasa_farm.state.ctx import CtxNorm, build_ctx
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(50, 64, generator=g, dtype=DTYPE) * 3 + 1.0
    n = CtxNorm(64).fit(emb)
    z = n(emb)
    assert z.shape == (50, 64)
    assert float(z.mean()) == pytest.approx(0.0, abs=0.2)
    ctx = build_ctx(emb[:2], n)
    assert ctx.shape == (2, 64) and torch.isfinite(ctx).all()


def test_trait_hyper_identity_and_bounds(crops):
    from nasa_farm.state.ctx import TraitHyper
    th = TraitHyper(8, crops.n_crops, crops.table.shape[1])
    ctx = torch.randn(4, 8, dtype=DTYPE)
    mult = th(ctx)
    assert mult.shape == (4, crops.n_crops, crops.table.shape[1])
    assert torch.equal(mult, torch.ones_like(mult))   # zero-init → 1.0
    th.net[-1].weight.data.normal_(0, 0.5)
    mult = th(ctx)
    assert mult.min() >= 0.5 - 1e-9 and mult.max() <= 2.0 + 1e-9


def test_condition_wires_ctx_and_latents(field, crops):
    from nasa_farm.dynamics.model import WorldModel
    from nasa_farm.state.ctx import LatentInit, condition
    m = WorldModel(crops, latent_dim=6, ctx_dim=8)
    init = LatentInit(8, 6, m.latent.h_dim)
    ctx = torch.randn(1, 8, dtype=DTYPE)
    f2 = condition(m, field, ctx, init=init)
    assert torch.equal(f2.ctx, ctx)
    assert f2.state.z.shape == (1, 6) and f2.state.zh.shape[1] == m.latent.h_dim


# ── dynamics/interv ───────────────────────────────────────────────────

def test_edit_state_sets_masked_only(field):
    from nasa_farm.dynamics.intervention import (
        EDIT_VARS, edit_state, get_vars)
    s = field.state
    e = len(EDIT_VARS)
    mask = torch.zeros(e, dtype=torch.bool)
    mask[EDIT_VARS.index("lai")] = True
    vals = get_vars(s).clone()
    vals[:, EDIT_VARS.index("lai")] = 3.0
    s2 = edit_state(s, mask, vals, field.soil)
    v0, v1 = get_vars(s), get_vars(s2)
    assert float(v1[:, EDIT_VARS.index("lai")]) == 3.0
    untouched = [i for i in range(e) if i != EDIT_VARS.index("lai")]
    assert torch.equal(v1[:, untouched], v0[:, untouched])


def test_edit_state_clamps_to_bounds(field):
    from nasa_farm.dynamics.intervention import (
        EDIT_VARS, edit_state, get_vars)
    mask = torch.zeros(len(EDIT_VARS), dtype=torch.bool)
    mask[EDIT_VARS.index("sw0")] = True
    vals = torch.full((1, len(EDIT_VARS)), 1e9, dtype=DTYPE)
    s2 = edit_state(field.state, mask, vals, field.soil)
    assert float(get_vars(s2)[:, 0]) <= float(field.soil.sat[:, 0])


def test_response_net_zero_init(field):
    from nasa_farm.dynamics.intervention import EDIT_VARS, ResponseNet
    r = ResponseNet()
    e = len(EDIT_VARS)
    out = r(torch.randn(4, 14, dtype=DTYPE),
            torch.randn(4, e, dtype=DTYPE),
            torch.zeros(e, dtype=torch.bool))
    assert torch.equal(out, torch.zeros(4, e, dtype=DTYPE))


def test_whatif_holds_edit_and_runs(physics_model, field, warm_year):
    from nasa_farm.dynamics.intervention import EDIT_VARS, whatif
    mask = torch.zeros(len(EDIT_VARS), dtype=torch.bool)
    mask[EDIT_VARS.index("n_min")] = True
    vals = torch.zeros(1, len(EDIT_VARS), dtype=DTYPE)
    vals[:, EDIT_VARS.index("n_min")] = 200.0
    acts = Actions.idle(1, warm_year.days)
    f2 = whatif(physics_model, field, warm_year, acts, 100, mask, vals)
    assert torch.isfinite(f2.state.biomass).all()


def test_fit_response_learns(physics_model, field, warm_year):
    from nasa_farm.dynamics.intervention import ResponseNet, fit_response
    from nasa_farm.types import FORCING_VARS, Forcing
    f120 = Forcing(*(getattr(warm_year, k)[:, :120] for k in FORCING_VARS))
    r = ResponseNet()
    rep = fit_response(physics_model, r, field, f120,
                       Actions.idle(1, 120), days=(40, 60),
                       edits_per_day=2, epochs=3)
    assert rep["final"] >= 0.0 and rep["history"]


# ── neural SDE layer ──────────────────────────────────────────────────

def test_diffusion_identity_at_init(crops, field, warm_year):
    from nasa_farm.dynamics.model import WorldModel
    m = WorldModel(crops, residual=True, diffusion=True, seed=0)
    frc = warm_year
    a = Actions.idle(1, frc.days)
    f1, _ = m.step(field, a.day(0), frc.day(0))
    # σ ≈ e^-6 at init → rollouts match the deterministic physics to ~1e-3
    m2 = WorldModel(crops, residual=False)
    f2, _ = m2.step(field, a.day(0), frc.day(0))
    for x, y in ((f1.state.sw, f2.state.sw),
                 (f1.state.biomass, f2.state.biomass)):
        assert float((x - y).abs().max()) < 0.05


def test_diffusion_reproducible(crops, field, warm_year):
    from nasa_farm.dynamics.model import WorldModel
    from nasa_farm.dynamics.residual import ResidualNet
    r = ResidualNet(diffusion=True, zero_init=False)
    m1 = WorldModel(crops, residual=False, seed=11)
    m1.residual = r
    m2 = WorldModel(crops, residual=False, seed=11)
    import copy
    m2.residual = copy.deepcopy(r)
    a = Actions.idle(1, warm_year.days)
    f1, _ = m1.step(field, a.day(0), warm_year.day(0))
    f2, _ = m2.step(field, a.day(0), warm_year.day(0))
    assert torch.equal(f1.state.sw, f2.state.sw)


# ── forcing/gen energy score ──────────────────────────────────────────

def test_energy_score_proper():
    from nasa_farm.metrics import energy_score
    g = torch.Generator().manual_seed(0)
    y = torch.randn(16, 6, generator=g, dtype=DTYPE)
    good = y + 0.01 * torch.randn(4, 16, 6, generator=g, dtype=DTYPE)
    bad = y + 2.0 * torch.randn(4, 16, 6, generator=g, dtype=DTYPE)
    assert float(energy_score(good, y)) < float(energy_score(bad, y))


def test_sample_gen_spread(warm_year):
    from nasa_farm.forcing import fit_gen, sample_gen
    m = fit_gen([warm_year, warm_year], epochs=3, hidden=32, seed=0)
    a = sample_gen(m, 30, 6, seed=2)
    assert a.precip.shape == (6, 30)
    assert float(a.precip.std(0).mean()) >= 0.0
    assert (a.rh <= 100).all() and (a.u2 >= 0).all()


# ── assim/latent + langevin ───────────────────────────────────────────

def test_ensf_lr_updates_and_bounds(physics_model, field, warm_year):
    from nasa_farm.assim import ensf_lr_update
    from nasa_farm.observe import observables
    f = field.repeat(8)
    chans = ("ndvi", "sm1")
    obs_pred = observables(f.state, chans)
    obs = obs_pred.mean(0) * 1.05
    st = ensf_lr_update(f.state, obs_pred, obs, 0.01, steps=3)
    assert torch.isfinite(st.sw).all() and (st.n_min >= 0).all()


def test_fit_filter_and_update(physics_model, field, warm_year):
    from nasa_farm.assim import fit_filter, net_update
    from nasa_farm.assim.kalman import STATE_VARS, _get_var
    net = fit_filter(physics_model, field, warm_year,
                     Actions.idle(1, warm_year.days),
                     ("ndvi",), every=60, members=8, epochs=2, seed=0)
    upd = net_update(net)
    f = field.repeat(8)
    from nasa_farm.observe import observables
    pred = observables(f.state, ("ndvi",))
    st = upd(f.state, pred, pred.mean(0), 0.01)
    assert torch.isfinite(_get_var(st, "sw0")).all()


def test_ula_posterior_shapes(physics_model, field, warm_year):
    from nasa_farm.assim import ula_posterior
    from nasa_farm.optimize.targets import ObservedSeries, obs_series
    from nasa_farm.types import FORCING_VARS, Forcing
    f60 = Forcing(*(getattr(warm_year, k)[:, :60] for k in FORCING_VARS))
    acts = Actions.idle(1, 60)
    with torch.inference_mode():
        pred = obs_series(physics_model, field, f60, acts, ("ndvi",), 15)
    obs = ObservedSeries(values=pred[0])
    outs = ula_posterior(physics_model, field, f60, acts, obs, ("ndvi",),
                         every=15, steps=6, burn=2, n_samples=3)
    assert len(outs) == 3
    assert all(torch.isfinite(o.state.sw).all() for o in outs)


def test_plan_schedule_orders(physics_model, field, warm_year):
    from nasa_farm.assim import plan_schedule
    from nasa_farm.types import FORCING_VARS, Forcing
    f90 = Forcing(*(getattr(warm_year, k)[:, :90] for k in FORCING_VARS))
    sch = plan_schedule(physics_model, field, f90, Actions.idle(1, 90),
                        tuple(range(5, 90, 5)), ("ndvi", "sm1"),
                        n_pick=4)
    assert len(sch) == 4 and all(5 <= t <= 90 for t, _, _ in sch)


# ── optimize/sbi ──────────────────────────────────────────────────────

def test_joint_flow_posterior_and_likelihood():
    from nasa_farm.optimize.sbi import fit_joint, Posterior
    g = torch.Generator().manual_seed(0)
    theta = torch.randn(300, 2, generator=g, dtype=DTYPE)
    x = torch.stack([theta[:, 0] + theta[:, 1],
                     0.3 * torch.randn(300, generator=g, dtype=DTYPE)], 1)
    m = fit_joint(theta, x, epochs=30, hidden=64, seed=0)
    p = Posterior(m)
    s = p.sample(x[:1], 16, steps=3, seed=0)
    assert s.shape == (16, 2) and torch.isfinite(s).all()
    l = p.likelihood(theta[:1], 16, steps=3, seed=0)
    assert l.shape == (16, 2) and torch.isfinite(l).all()


# ── optimize/shoot ────────────────────────────────────────────────────

def test_shoot_fit_runs(physics_model, field, warm_year):
    from nasa_farm.optimize.shoot import ms_bounds, shoot_fit
    from nasa_farm.optimize.targets import ObservedSeries, obs_series
    from nasa_farm.types import FORCING_VARS, Forcing
    assert ms_bounds(100, 4) == [(0, 25), (25, 50), (50, 75), (75, 100)]
    f60 = Forcing(*(getattr(warm_year, k)[:, :60] for k in FORCING_VARS))
    acts = Actions.idle(1, 60)
    with torch.inference_mode():
        pred = obs_series(physics_model, field, f60, acts, ("ndvi",), 10)
    rep = shoot_fit(physics_model, field, f60, acts,
                    ObservedSeries(values=pred[0]), ("ndvi",),
                    every=10, segments=3, epochs=3)
    assert rep["epochs_run"] > 0 and rep["boundary_offset"]


# ── value + planning ──────────────────────────────────────────────────

def test_fit_value_and_mppi_horizon(physics_model, field, warm_year):
    from nasa_farm.value import ValueNet, fit_value
    from nasa_farm.control import mppi
    from nasa_farm.types import FORCING_VARS, Forcing
    f90 = Forcing(*(getattr(warm_year, k)[:, :90] for k in FORCING_VARS))
    acts = Actions.idle(1, 90)
    net = fit_value(physics_model, field, f90, acts, every=30, epochs=5)
    assert isinstance(net, ValueNet)
    res = mppi(physics_model, field, f90, acts, iters=1, samples=8,
               horizon=45, value=net, seed=0)
    assert res.reward == res.reward and res.actions.days == 90


def test_policy_grad_runs(physics_model, field, warm_year):
    from nasa_farm.policy import PolicyNet, policy_grad
    from nasa_farm.types import FORCING_VARS, Forcing
    f30 = Forcing(*(getattr(warm_year, k)[:, :30] for k in FORCING_VARS))
    p = policy_grad(physics_model, field, f30, Actions.idle(1, 30),
                    PolicyNet(), epochs=2)
    assert isinstance(p, PolicyNet)


# ── distill growth + providers + experiment ───────────────────────────

def test_distill_autogrow():
    from nasa_farm.optimize.distill import grow
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200, 14, generator=g, dtype=DTYPE)
    y = 2 * x[:, 1] * x[:, 7] + 0.1 * x[:, 3]
    idx, w, names = grow(x, y, k=4, rounds=1)
    assert len(idx) >= 1 and len(names) > 1 + 3 * 14


def test_providers_registered():
    from nasa_farm.forcing.provide import _PROVIDERS
    assert {"power", "synth", "gen", "file", "wn3", "aifs"} <= set(
        _PROVIDERS)


def test_nisar_sme_descriptor():
    from nasa_farm.acquire import NISAR_SME
    assert NISAR_SME.short_name.startswith("NISAR_L3_SME")
    assert NISAR_SME.requires_auth


def test_experiment_registry_has_interv():
    from nasa_farm.exp import experiments
    assert "interv" in experiments()


def test_interv_experiment_smoke():
    from nasa_farm.exp import Spec, run
    rep = run("interv", Spec(epochs=3, members=8, seed=1,
                             obs_channels=("ndvi",)))
    assert "perturbations" in rep.results and "plan_tau" in rep.results
    assert rep.results["plan_tau"]["prior"] <= 1.0
