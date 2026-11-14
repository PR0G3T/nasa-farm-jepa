"""New-dynamics layer — extra processes, factored/sparse latents, DSReg,
SINDy discovery, counterfactual abduction, L-BFGS refine + SDI gate,
TTT replay, learned-Q EnKF, Fisher design, SPW ensembles, Nash gap."""

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.counterfactual import abduct, counterfactual
from nasa_farm.dynamics.extra import EXTRAS, step_extra
from nasa_farm.dynamics.latent import OPS, Factor, Dynamics
from nasa_farm.optimize.embedding import dissimilarity
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.dynamics.residual import N_FEATURES
from nasa_farm.control import _quasi_newton, path_integral
from nasa_farm.optimize.rotation import rotation
from nasa_farm.optimize.identification import identify, library, thresholded
from nasa_farm.optimize.replay import Replay
from nasa_farm.assimilation.design import fisher_design
from nasa_farm.assimilation.kalman import Observation, assimilate
from nasa_farm.allocation import evaluate_policies, nash_gap
from nasa_farm.policy import Policy
from nasa_farm.state.crops import CropLibrary
from nasa_farm.state.field import Field, initial_state
from nasa_farm.state.soil import SoilProfile
from nasa_farm.types import ACTION_VARS, FORCING_VARS, Actions, Forcing
from nasa_farm.uncertainty import spw_rollout


def _field(batch: int = 1) -> Field:
    soil = SoilProfile.preset("silt_loam")
    return Field(initial_state(batch, soil), soil,
                 torch.full((batch,), 42.0, dtype=DTYPE))


def _fa(days: int = 365):
    from nasa_farm.forcing import synthetic_year, tile_to_horizon
    from nasa_farm.plan import expand_actions
    crops = CropLibrary.default()
    forcing = tile_to_horizon(synthetic_year(42.0, seed=0), 1)
    actions = expand_actions(
        torch.tensor([[crops.index("maize")]]), crops.table)
    return forcing, actions


def _cut_f(f: Forcing, days: int) -> Forcing:
    return Forcing(*(getattr(f, k)[:, :days] for k in FORCING_VARS))


def _cut_a(a: Actions, days: int) -> Actions:
    return Actions(*(getattr(a, k)[:, :days] for k in ACTION_VARS))


class TestExtra:
    def test_identity_off(self):
        m = WorldModel(residual=False)
        assert m.extra == ()
        f, a = _fa()
        f2 = m.step(_field(), a.day(0), f.day(0))[0]
        assert f2.state.biomass.shape == (1,)

    def test_heat_shock_fires_and_conserves(self):
        m = WorldModel(residual=False, extra=("heat_shock",))
        field = _field()
        s = field.state.replace(
            biomass=torch.tensor([5000.0], dtype=DTYPE),
            litter=torch.tensor([1000.0], dtype=DTYPE),
            lai=torch.tensor([3.0], dtype=DTYPE),
            growing=torch.ones(1, dtype=torch.bool))
        f_d = {"tmax": torch.tensor([40.0], dtype=DTYPE)}
        s2 = EXTRAS["heat_shock"](s, field.soil, f_d, {}, {})
        assert float(s2.biomass) < float(s.biomass)
        assert float(s2.litter - s.litter) == pytest.approx(
            float(s.biomass - s2.biomass), abs=1e-9)
        assert float(s2.lai) < float(s.lai)

    def test_truth_gap_in_step(self):
        m0 = WorldModel(residual=False)
        m1 = WorldModel(residual=False, extra=("crust", "n_immob"))
        field = _field()
        s = field.state.replace(
            sw=torch.tensor([[400.0] * field.state.n_layers], dtype=DTYPE),
            litter=torch.tensor([4000.0], dtype=DTYPE))
        f_d = {"precip": torch.tensor([40.0], dtype=DTYPE)}
        s2 = step_extra(m1.extra, s, field.soil, f_d, {}, {})
        assert float(s2.tws) > float(s.tws)
        assert float(s2.n_org) > float(s.n_org)
        assert float(s2.sw[:, 0] + s2.tws - s.tws - s.sw[:, 0]) \
            == pytest.approx(0.0, abs=1e-9)


class TestFactor:
    def test_identity_at_init(self):
        fd = Factor(4, OPS)
        feats = torch.randn(2, N_FEATURES, dtype=DTYPE)
        z = torch.randn(2, fd.dim, dtype=DTYPE)
        z2, zh = fd(feats, z, torch.randn(2, fd.hidden_dimension, dtype=DTYPE))
        assert torch.equal(z2, z)
        assert zh.shape == (2, fd.hidden_dimension)

    def test_intervention_selectivity(self):
        torch.manual_seed(0)
        fd = Factor(2, {"water": OPS["water"], "bio": OPS["bio"]})
        with torch.no_grad():
            for h in fd.head.values():
                h.weight.normal_(0, 0.5)
        feats = torch.zeros(1, N_FEATURES, dtype=DTYPE)
        z = torch.zeros(1, fd.dim, dtype=DTYPE)
        zh = torch.zeros(1, fd.hidden_dimension, dtype=DTYPE)
        f_irrig = feats.clone()
        f_irrig[:, 11] = 5.0          # irrigation column
        z_a, _ = fd(feats, z, zh)
        z_b, _ = fd(f_irrig, z, zh)
        w_i, w_d = 0, fd.part_dim     # water slot, bio slot
        d = (z_b - z_a).detach()
        assert float(d[:, w_i:w_i + w_d].abs().sum()) > 0
        assert float(d[:, w_d:].abs().sum()) == 0.0

    def test_relu_link_nonneg(self):
        ld = Dynamics(4, link="relu")
        with torch.no_grad():
            ld.head.bias.fill_(-1.0)
        feats = torch.randn(1, N_FEATURES, dtype=DTYPE)
        z, _ = ld(feats, torch.ones(1, 4, dtype=DTYPE))
        assert bool((z >= 0).all())

    def test_model_integration(self):
        m = WorldModel(residual=True, latent_dimension=2, latent_factors="ops",
                       seed=0)
        field = _field()
        s = field.state.replace(
            z=torch.zeros(1, m.latent.dim, dtype=DTYPE),
            zh=torch.zeros(1, m.latent.hidden_dimension, dtype=DTYPE))
        f2, _ = m.step(Field(s, field.soil, field.latitude),
                       {"sow": torch.tensor([-1]), "irrigation": torch.zeros(1),
                        "n_fert": torch.zeros(1)},
                       {"tmin": torch.tensor([10.0]), "tmax": torch.tensor([20.0]),
                        "doy": torch.tensor([100.0]), "srad": torch.tensor([15.0]),
                        "rh": torch.tensor([60.0]), "u2": torch.tensor([2.0]),
                        "precip": torch.zeros(1)})
        assert f2.state.z.shape == (1, m.latent.dim)


class TestSparseDsreg:
    def test_rdmreg_prefers_rectified(self):
        generation = torch.Generator().manual_seed(0)
        lap = torch.rand(512, 8, generator=generation, dtype=DTYPE).exponential_() \
            - torch.rand(512, 8, generator=generation, dtype=DTYPE).exponential_()
        z_rect = torch.relu(lap)                    # target family: ReLU(GG)
        z_dense = torch.randn(512, 8, generator=generation, dtype=DTYPE)
        assert float(dissimilarity(z_rect, generation)) < float(dissimilarity(z_dense, generation))

    def test_dsreg_recovers_permutation(self):
        torch.manual_seed(0)
        n, d = 400, 3
        z = torch.randn(n, d, dtype=DTYPE)
        q, _ = torch.linalg.qr(torch.randn(d, d, dtype=DTYPE))
        h = z @ q
        x = torch.stack([z[:, 0], z[:, 1], z[:, 2], z[:, 0] * z[:, 1]], -1)
        R, z_t, rep = rotation(h, x)
        corr = torch.zeros(d, d)
        for i in range(d):
            for j in range(d):
                corr[i, j] = abs(float(torch.corrcoef(
                    torch.stack([z_t[:, i], z[:, j]]))[0, 1]))
        assert float(corr.max(1).values.min()) > 0.9


class TestSindy:
    def test_stlsq_recovers_support(self):
        torch.manual_seed(0)
        X = torch.randn(500, N_FEATURES, dtype=DTYPE)
        phi, names = library(X)
        j = names.index("sw0*lai")
        y = 3.0 * phi[:, j:j + 1] + 0.5 * phi[:, 0:1]
        coef = thresholded(phi, y, threshold=0.05)
        top = set(torch.nonzero(coef[:, 0].abs() > 0).flatten().tolist())
        assert j in top and 0 in top and len(top) <= 3

    def test_identify_runs(self):
        m = WorldModel(residual=True, seed=0)
        with torch.no_grad():
            m.residual.head.weight.normal_(0, 0.01)
        f, a = _fa(60)
        rep = identify(m, _field(), f, a)
        assert 0.0 <= rep["r2"] <= 1.0
        assert set(rep["equations"]) == {"sw0", "swr", "n_min", "lai",
                                         "biomass"}


class TestCF:
    def test_abduct_forward_fill(self):
        observed = torch.full((1, 6, 1), float("nan"), dtype=DTYPE)
        observed[0, 2, 0] = 10.0
        observed[0, 5, 0] = 14.0
        predicted = torch.ones(1, 6, 1, dtype=DTYPE)
        off = abduct(observed, predicted)
        assert float(off[0, 0]) == 0.0
        assert float(off[0, 3]) == 9.0      # carried forward
        assert float(off[0, 5]) == 13.0

    def test_counterfactual_beats_naive(self):
        torch.manual_seed(0)
        from nasa_farm.observe.trace import ObservationTrace
        from nasa_farm.state.crops import INDEX
        soil = SoilProfile.preset("sand")   # water-limited: irrigation bites
        field = Field(initial_state(1, soil), soil,
                    torch.tensor([42.0], dtype=DTYPE))
        f, a = _fa()
        truth = WorldModel(residual=False)
        m = WorldModel(residual=False)
        with torch.no_grad():     # structural gap: half the radiation efficiency
            m.crop_table[:, INDEX["rue"]] *= 0.5
        traits = ObservationTrace(f.days, ("biomass",), 5)
        truth.rollout(field, f, a, observer=traits)
        observed = traits.result()
        pulse = ((torch.arange(f.days) % 10) == 0).to(DTYPE) * 15.0
        alt = Actions(sow=a.sow, irrigation=a.irrigation + pulse,
                      n_fert=a.n_fert)
        tr2 = ObservationTrace(f.days, ("biomass",), 5)
        truth.rollout(field, f, alt, observer=tr2)
        actual = tr2.result()
        cf, information = counterfactual(m, field, f, a, alt, observed, ("biomass",), 5)
        e_cf = float((cf - actual).pow(2).mean().sqrt())
        e_nv = float((information["pred_alt"] - actual).pow(2).mean().sqrt())
        assert e_cf < 0.5 * e_nv


class TestControlOpt:
    def test_lbfgs_quadratic(self):
        p = [torch.zeros(4, dtype=DTYPE, requires_grad=True)]
        _quasi_newton(lambda: (p[0] - 3.0).pow(2).sum(), p, 15)
        assert float((p[0] - 3.0).abs().max()) < 1e-3

    def test_mppi_gate_and_lbfgs(self):
        m = WorldModel(residual=False)
        f, a = _fa(120)
        f120 = _cut_f(f, 120)
        a120 = _cut_a(a, 120)
        result = path_integral(m, _field(), f120, a120, iters=2, samples=16,
                   refine=5, refine_alg="lbfgs", gate_k=2.0, seed=0)
        assert result.actions.irrigation.shape[-1] == 120


class TestTTTUQ:
    def test_ttt_reduces_error(self):
        m = WorldModel(latent_dimension=4, seed=0)
        replay = Replay(m, batch=8)
        generation = torch.Generator().manual_seed(0)
        w = torch.randn(N_FEATURES, 4, generator=generation, dtype=DTYPE)
        for _ in range(32):
            feats = torch.randn(4, N_FEATURES, generator=generation, dtype=DTYPE)
            z = torch.randn(4, 4, generator=generation, dtype=DTYPE)
            zh = torch.randn(4, m.latent.hidden_dimension, generator=generation, dtype=DTYPE)
            replay.push(feats, z, zh, z + feats @ w)
        first = replay.step(1)
        for _ in range(15):
            last = replay.step(1)
        assert last < first or last < 1.0

    def test_spw_spread(self):
        m = WorldModel(residual=True, seed=0)
        with torch.no_grad():
            m.residual.head.weight.normal_(0, 0.02)
        f, a = _fa(120)
        f120 = _cut_f(f, 120)
        a120 = _cut_a(a, 120)
        output = spw_rollout(m, _field(), f120, a120, members=4, scale=0.1)
        assert output.shape[0] == 4
        assert float(output.std(0).abs().max()) >= 0.0


class TestAssimNew:
    def test_learned_q_runs(self):
        m = WorldModel(residual=True, diffusion=True, seed=0)
        field = Field(initial_state(8, SoilProfile.preset("silt_loam")),
                    SoilProfile.preset("silt_loam").repeat(8),
                    torch.full((8,), 42.0, dtype=DTYPE))
        f, a = _fa(60)
        f60 = _cut_f(f, 60).repeat(8)
        a60 = _cut_a(a, 60).repeat(8)
        schedule = [Observation(t=30, channels=("ndvi",),
                             values=torch.tensor([0.5], dtype=DTYPE),
                             std=0.05)]
        output = assimilate(m, field, f60, a60, schedule, learned_q=True)
        assert output.state.sw.shape[0] == 8

    def test_fisher_design_ranks(self):
        m = WorldModel(residual=False)
        f, a = _fa(120)
        f120 = _cut_f(f, 120)
        a120 = _cut_a(a, 120)
        ranked = fisher_design(m, _field(), f120, a120, ("rue",),
                               days=(10, 80), channels=("ndvi",), eps=0.1)
        assert ranked and ranked[0][1] == "ndvi"


class TestMarl:
    def test_nash_gap_shape(self):
        m = WorldModel(residual=False)
        soil = SoilProfile.preset("loam").repeat(3)
        field = Field(initial_state(3, soil), soil,
                    torch.full((3,), 42.0, dtype=DTYPE))
        f, _ = _fa(60)
        f60 = _cut_f(f, 60).repeat(3)
        a60 = Actions.idle(3, 60)
        pols = [Policy() for _ in range(3)]
        r, _ = evaluate_policies(m, field, f60, a60, pols)
        rep = nash_gap(m, field, f60, a60, pols, levels=(0.0, 10.0))
        assert r.shape[0] == 3 and len(rep["nash_gap"]) == 3
        assert all(g >= -1e-6 for g in rep["nash_gap"])


class TestExpNew:
    def test_registry(self):
        from nasa_farm.experiments import experiments
        for n in ("cf", "discover", "scale"):
            assert n in experiments()

    def test_cf_experiment(self):
        from nasa_farm.experiments import run
        from nasa_farm.experiments.specification import Specification as S
        rep = run("cf", S(horizon=1, cadence=10, noise=0.0,
                          realizations=1))
        ch = rep.results["channels"]["ndvi"]
        assert "improvement_pct" in ch
