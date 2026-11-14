"""Tests for the phase-2 modules: generative forcing, providers, analogues,
bias correction, BOED scheduling, action detection, sow-day search, policy
distillation, trajectory UQ, SWAG, SBI, symbolic distillation, landscape
routing and MRV accounting."""

import torch

from conftest import day_action, idle
from nasa_farm.constants import DTYPE
from nasa_farm.types import Actions, Forcing


def _short(f: Forcing, days: int) -> Forcing:
    sl = lambda t: t[:, :days]
    return Forcing(*(sl(getattr(f, k)) for k in
                     ("tmin", "tmax", "srad", "precip", "rh", "u2", "doy",
                      "co2")))


# ── forcing ──────────────────────────────────────────────────────────

def test_wxgen_shapes_and_seed(warm_year):
    from nasa_farm.forcing import fit_gen, sample_gen
    m = fit_gen([warm_year, warm_year], epochs=5, hidden=32, seed=0)
    a, b = (sample_gen(m, 30, 4, seed=s) for s in (1, 1))
    assert a.tmin.shape == (4, 30) and a.precip.min() >= 0
    assert torch.equal(a.tmin, b.tmin)          # seeded → reproducible
    assert (a.tmax >= a.tmin).all()


def test_analog_weights_normalized(warm_year):
    from nasa_farm.forcing import analog_weights, weighted_ensemble
    hist = [warm_year, warm_year.perturb(delta_t=3.0),
            warm_year.perturb(precip_scale=0.3)]
    w = analog_weights(hist, warm_year)
    assert w.shape == (3,) and torch.isclose(
        w.sum(), torch.tensor(1.0, dtype=w.dtype))
    assert w[0] > w[1] and w[0] > w[2]          # self is the best analogue
    ens = weighted_ensemble(hist, w, horizon=1, seed=0)
    assert ens.batch == 3 and ens.days == 365


def test_qmap_removes_bias(warm_year):
    from nasa_farm.forcing import apply_qmap, fit_qmap
    biased = warm_year.perturb(delta_t=2.0, precip_scale=1.5)
    qm = fit_qmap(biased, warm_year, windows=12)
    fixed = apply_qmap(biased, qm)
    assert fixed.batch == 1 and fixed.days == warm_year.days
    med_b = float((biased.tmax - warm_year.tmax).median())
    med_f = float((fixed.tmax - warm_year.tmax).median())
    assert abs(med_f) < abs(med_b)


def test_provider_dispatch(warm_year):
    from nasa_farm.forcing import ensemble
    f = ensemble("synth", lat=42.0, horizon=1, members=3, seed=0)
    assert f.batch == 3 and f.days == 365
    try:
        ensemble("nope", lat=0.0)
    except KeyError:
        pass
    else:
        raise AssertionError("unknown source must raise")


# ── assimilation design & detection ─────────────────────────────────

def test_design_schedule_ranks(physics_model, field, warm_year):
    from nasa_farm.assimilation import design_schedule, eig
    f = _short(warm_year, 40)
    acts = Actions.idle(8, 40)
    fld = field.repeat(8)
    ranked = design_schedule(physics_model, fld, f, acts,
                             days=(10, 20, 30), channels=("sm", "ndvi"),
                             n_pick=4)
    assert len(ranked) == 4
    gains = [g for _, _, g in ranked]
    assert gains == sorted(gains, reverse=True) and all(g >= 0 for g in gains)
    ens = torch.arange(8, dtype=DTYPE)
    assert float(eig(ens, 1.0)) > float(eig(ens * 0.01, 1.0))


def test_detect_sow_and_till():
    from nasa_farm.observe import infer_actions
    t = torch.arange(120, dtype=DTYPE)
    ndvi = torch.full((120,), 0.15, dtype=DTYPE)
    ndvi[40:80] += torch.linspace(0, 0.6, 40)
    ndvi[80:] = torch.linspace(0.75, 0.2, 40)
    s0v = torch.full((120,), -12.0, dtype=DTYPE)
    s0v[30] = -9.0                                  # bare-soil spike
    a = infer_actions(ndvi, t, crop_idx=2, s0v=s0v)
    assert int((a.sow[0] == 2).nonzero()[0]) in range(35, 55)
    assert float(a.till[0, 30]) == 1.0
    assert float(a.irrigation.sum()) == 0.0


# ── control: sow-day search & policy ────────────────────────────────

def _tiny_run(physics_model, field, warm_year, **kw):
    from nasa_farm.control import mppi
    ens = warm_year.repeat(3)
    base = Actions.idle(1, ens.days)
    base = Actions(sow=torch.full((1, ens.days), -1, dtype=torch.long),
                   irrigation=base.irrigation, n_fert=base.n_fert)
    sow = base.sow.clone()
    sow[0, 150] = 0
    base = Actions(sow=sow, irrigation=base.irrigation,
                   n_fert=base.n_fert)
    return mppi(physics_model, field, ens, base, iters=2, samples=12,
                seed=0, **kw), base


def test_mppi_sow_shifts(physics_model, field, warm_year):
    res, base = _tiny_run(physics_model, field, warm_year,
                          sow_shifts=(0, -30, 30))
    i = int((res.actions.sow[0] >= 0).nonzero()[0])
    assert i in (120, 150, 180)


def test_mppi_policy_warmstart(physics_model, field, warm_year):
    res0, base = _tiny_run(physics_model, field, warm_year)
    warm = torch.stack([res0.actions.irrigation[0],
                        res0.actions.n_fert[0]])
    res1, _ = _tiny_run(physics_model, field, warm_year, base_mu=warm)
    assert res1.reward >= res0.reward - 1e-6


def test_policy_net_nonnegative():
    from nasa_farm.policy import PolicyNet
    from nasa_farm.dynamics.residual import N_FEATURES
    p = PolicyNet(hidden=16, depth=1)
    out = p(torch.randn(4, N_FEATURES, dtype=torch.float64))
    assert out.shape == (4, 2) and (out >= 0).all()


# ── UQ: trajectory bands, ACI, EnbPI, SWAG ───────────────────────────

def test_trajectory_conformal_joint_band():
    from nasa_farm.uncertainty import TrajectoryConformal
    pred = torch.zeros(30, 50, dtype=DTYPE)
    y = torch.randn(30, 50, dtype=torch.float64) * 0.1
    tc = TrajectoryConformal(alpha=0.1).fit(pred, y)
    lo, hi = tc.bound(torch.zeros(1, 50, dtype=DTYPE))
    covered = (y >= lo) & (y <= hi)                 # whole-row coverage
    rate = covered.all(dim=1).to(DTYPE).mean()
    assert float(rate) >= 0.8                       # joint ≥ marginal
    assert (hi > lo).all()


def test_aci_direction():
    from nasa_farm.uncertainty import aci
    a = aci(0.10, err=True)
    assert a < 0.10                                 # miscoverage drops → widen
    b = aci(0.10, err=False)
    assert b > 0.10                                 # covered → narrow


def test_enbpi_band_ordered():
    from nasa_farm.uncertainty import enbpi
    res = torch.randn(60, 20, dtype=torch.float64)
    pred = torch.zeros(20, dtype=DTYPE)
    lo, hi = enbpi(pred, res, alpha=0.1)
    assert (hi >= lo).all() and lo.shape == (20,)


def test_swag_collect_sample_restore():
    import torch.nn as nn
    from nasa_farm.optimize.averaging import Swag
    net = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 2)).to(torch.float64)
    ref = torch.cat([p.clone().reshape(-1) for p in net.parameters()])
    sw = Swag(net)
    for _ in range(3):
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.01)
        sw.collect()
    assert sw.n == 3 and sw.mean.shape == ref.shape
    s = sw.sample()
    assert s.shape == ref.shape
    cur = torch.cat([p.clone().reshape(-1) for p in net.parameters()])
    sw.load(ref)
    restored = torch.cat([p.clone().reshape(-1) for p in net.parameters()])
    assert torch.equal(restored, ref)
    assert not torch.equal(cur, ref)


# ── inference: joint-flow posterior, symbolic distillation ───────────

def test_posterior_recovers_gaussian():
    from nasa_farm.optimize.inference import fit_posterior
    g = torch.Generator().manual_seed(0)
    theta = torch.randn(400, 2, generator=g, dtype=torch.float64)
    x = theta[:, :1] * 0.5 + 0.1 * torch.randn(400, 1, generator=g,
                                               dtype=torch.float64)
    post = fit_posterior(theta, x, epochs=50, seed=0)
    s = post.sample(x[:1], 64)
    assert s.shape == (64, 2)
    # posterior mean of θ0 should track x*2 within noise
    assert abs(float(s[:, 0].mean() - 2 * float(x[0, 0]))) < 1.5


def test_omp_recovers_sparse_signal():
    from nasa_farm.optimize.distill import basis, basis_names, omp
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200, 4, generator=g, dtype=torch.float64)
    F = basis(x)
    y = 3.0 * F[:, 1] - 2.0 * F[:, 5] + 0.01 * torch.randn(
        200, generator=g, dtype=torch.float64)
    idx, w = omp(F, y, k=6)
    assert {1, 5} <= set(idx.tolist())
    resid = (y - F[:, idx] @ w).abs().max()
    assert float(resid) < 0.1
    assert len(basis_names(4)) == F.shape[1]


# ── landscape routing ────────────────────────────────────────────────

def test_landscape_conserves_runoff(physics_model, soil):
    from nasa_farm.landscape import FieldGraph, Landscape
    from nasa_farm.state.field import Field, initial_state
    n = 3
    soil_n = soil.repeat(n)
    fld = Field(initial_state(n, soil_n), soil_n,
                torch.tensor([42.0] * n, dtype=DTYPE))
    f = _short(__import__("nasa_farm.forcing", fromlist=["synthetic_year"])
               .synthetic_year(42.0, seed=1), 10)
    f = Forcing(f.tmin.repeat(n, 1), f.tmax.repeat(n, 1),
                f.srad.repeat(n, 1),
                torch.full((n, 10), 80.0, dtype=DTYPE),  # saturated runoff
                f.rh.repeat(n, 1), f.u2.repeat(n, 1), f.doy.repeat(n, 1))
    acts = Actions.idle(n, 10)
    ls = Landscape(physics_model, FieldGraph.chain(n, frac=0.5))
    out = ls.rollout(fld, f, acts)
    assert out.state.sw.shape == (n, soil_n.n_layers)
    assert ls.pending is not None and float(ls.pending[1]) > 0


# ── MRV ──────────────────────────────────────────────────────────────

def test_mrv_signs_and_units():
    from nasa_farm.crediting import C_TO_CO2, N_TO_N2O, account
    soc0 = torch.tensor([4.0], dtype=DTYPE)
    soc1 = torch.tensor([5.0], dtype=DTYPE)          # +1 kg C m-2
    n2o = torch.zeros(1, dtype=DTYPE)
    ac = account(soc0, soc1, n2o, n2o, area_ha=1.0)
    assert torch.isclose(ac.soc_co2e[0],
                         torch.tensor(10.0 * C_TO_CO2, dtype=DTYPE))
    assert float(ac.net_co2e) > 0
    ac2 = account(soc0, soc0, n2o, n2o + 1.0)         # +1 kg N ha-1
    dn2o = 1.0 * N_TO_N2O / 1000.0 * 273.0
    assert torch.isclose(ac2.n2o_co2e[0], torch.tensor(dn2o, dtype=DTYPE))
    assert float(ac2.net_co2e) < 0


def test_mrv_creditable_lower_bound():
    from nasa_farm.crediting import creditable
    net = torch.randn(200, 4, dtype=torch.float64)   # posterior samples
    lo = creditable(net, alpha=0.1)
    assert lo.shape == (4,)
    assert (lo <= net.mean(0)).all()


# ── emulation pretraining & closed-loop policy ───────────────────────

def test_emulate_shapes(physics_model, field, warm_year):
    from nasa_farm.optimize.pretrain import emulate
    obs = emulate(physics_model, [field], [warm_year],
                  [Actions.idle(1, 365)], channels=("yield_kg",))
    assert len(obs) == 1 and obs[0].values.shape[-1] > 0


def test_policy_actions_closed_loop(physics_model, field, warm_year):
    from nasa_farm.policy import PolicyNet, policy_actions
    f = _short(warm_year, 20)
    base = Actions.idle(1, 20)
    out = policy_actions(PolicyNet(hidden=16, depth=1),
                         physics_model, field, f, base)
    assert out.irrigation.shape == (1, 20) and (out.irrigation >= 0).all()
    assert (out.n_fert >= 0).all()
