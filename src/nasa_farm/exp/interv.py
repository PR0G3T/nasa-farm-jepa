"""Interventional evaluation — is the model a world model or a schedule memorizer?

A predictive crop emulator can nail observed-season metrics while failing
the only question that makes it a *world model*: does it predict the effect
of an action it never saw? This experiment answers three ways on held-out
forcing realizations:

- `perturbations`: Δmetrics induced by counterfactual action edits
  (extra irrigation, extra N, shifted sowing) — model delta vs truth-model
  delta, per channel. The "schedule memory trap" detector.
- `plan_tau`: Kendall τ between the model's rotation ranking and the
  truth's, on the scalarized Pareto objective — decision-level fidelity.
- `response`: edit_state what-if consistency — apply the same mid-season
  state edit to both models, divergence after edit vs before (a model
  whose post-edit rollout diverges less tracks the true coupling).
"""

from __future__ import annotations

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.interv import EDIT_VARS, edit_state
from nasa_farm.exp.run import register
from nasa_farm.exp.scene import calibrate, observe, scene
from nasa_farm.exp.spec import Spec
from nasa_farm.heads import METRICS
from nasa_farm.optimize.targets import season_metrics
from nasa_farm.plan import (
    enumerate_rotations, evaluate, objective_matrix, scalarize)
from nasa_farm.state.field import Field
from nasa_farm.types import ACTION_VARS, Actions


def _tau(a: torch.Tensor, b: torch.Tensor) -> float:
    """Kendall τ between two rankings (ties ignored)."""
    n = a.numel()
    if n < 2:
        return 1.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = float((a[i] - a[j]).sign() * (b[i] - b[j]).sign())
            conc += s > 0
            disc += s < 0
    return (conc - disc) / max(1, conc + disc)


def _perturbed(actions: Actions, channel: str, day: int, span: int,
               delta: float | int) -> Actions:
    kw = {k: getattr(actions, k).clone() for k in ACTION_VARS}
    if channel == "sow":
        kw["sow"] = kw["sow"].roll(int(delta), dims=1)
    else:
        kw[channel][:, day:day + span] += delta
    return Actions(**kw)


def _metrics(model, sc, forcing, actions) -> torch.Tensor:
    return season_metrics(model, sc.field, forcing, actions)[0]  # (H, M)


@register("interv")
def run(spec: Spec) -> dict:
    sc = scene(spec)
    obs = observe(spec, sc)
    cal, rep = calibrate(spec, sc, obs)
    f_t = sc.realization(0)
    idx = {n: i for i, n in enumerate(METRICS)}

    # --- action-perturbation fidelity -------------------------------------
    suite = [
        ("irrig+10d", "irrigation", 130, 10, 10.0),
        ("fert+40", "n_fert", 150, 1, 40.0),
        ("sow-14d", "sow", 0, 0, -14),
        ("sow+14d", "sow", 0, 0, 14),
    ]
    chans = ["yield_kg", "biomass_end", "water_used", "n_leached"]
    chans = [c for c in chans if c in idx]
    pert: dict = {}
    for name, ch, day, span, d in suite:
        a2 = _perturbed(sc.actions, ch, day, span, d)
        row = {}
        for label, mdl in (("prior", sc.prior), ("cal", cal),
                           ("oracle", sc.truth)):
            m0 = _metrics(mdl, sc, f_t, sc.actions)
            m1 = _metrics(mdl, sc, f_t, a2)
            dm = (m1 - m0).mean(0)                       # (M,) over years
            row[label] = dm
        truth_d = row.pop("oracle")
        for label in ("prior", "cal"):
            for c in chans:
                t, p = float(truth_d[idx[c]]), float(row[label][idx[c]])
                err = p - t
                pert.setdefault(c, {}).setdefault(name, {})[label] = {
                    "truth_delta": round(t, 4), "model_delta": round(p, 4),
                    "err": round(err, 4)}

    # --- plan-ranking fidelity ---------------------------------------------
    i_c = sc.crops.index(spec.crop)
    cands = [i_c, sc.crops.index("soybean"), sc.crops.index("wheat")]
    hh = max(1, min(spec.horizon, 2))
    plans = enumerate_rotations(cands, hh)
    if hh * 365 < sc.forcing_e.days:
        from nasa_farm.types import FORCING_VARS, Forcing
        ens = Forcing(*(getattr(sc.forcing_e, k)[:, :hh * 365]
                        for k in FORCING_VARS))
    else:
        ens = sc.forcing_e
    taus = {}
    with torch.inference_mode():
        ev_t = evaluate(sc.truth, sc.field, plans, ens)
        obj_t = scalarize(*objective_matrix(
            {k: v for k, v in ev_t.objectives.items()}), torch.ones(
            len(ev_t.objectives), dtype=DTYPE))
        for label, mdl in (("prior", sc.prior), ("cal", cal)):
            ev_m = evaluate(mdl, sc.field, plans, ens)
            obj_m = scalarize(*objective_matrix(
                {k: v for k, v in ev_m.objectives.items()}),
                torch.ones(len(ev_m.objectives), dtype=DTYPE))
            taus[label] = round(_tau(obj_m, obj_t), 4)

    # --- mid-season edit consistency ---------------------------------------
    e = len(EDIT_VARS)
    mask = torch.zeros(e, dtype=torch.bool)
    mask[EDIT_VARS.index("swr")] = True
    day = min(150, f_t.days - 60)

    def run_edit(mdl):
        f = sc.field
        for t in range(day):
            f, _ = mdl.step(f, sc.actions.day(t), f_t.day(t))
        cur = f.state
        st_e = edit_state(cur, mask,
                          torch.full((1, e), 400.0, dtype=DTYPE), f.soil)
        fe = Field(state=st_e, soil=f.soil, lat=f.lat, elev=f.elev,
                   ctx=f.ctx)
        fa, fb = f, fe
        for t in range(day, f_t.days):
            fa, _ = mdl.step(fa, sc.actions.day(t), f_t.day(t))
            fb, _ = mdl.step(fb, sc.actions.day(t), f_t.day(t))
        return fb.state.biomass, fb.state.sw.sum(-1), fb.state.n_min

    resp = {}
    b_t, w_t, n_t = run_edit(sc.truth)
    for label, mdl in (("prior", sc.prior), ("cal", cal)):
        b_m, w_m, n_m = run_edit(mdl)
        resp[label] = {
            "biomass_err": round(float(b_m - b_t), 2),
            "water_err": round(float(w_m - w_t), 2),
            "nmin_err": round(float(n_m - n_t), 4)}

    return {"calibration": rep, "perturbations": pert,
            "plan_tau": taus, "edit_response": resp}
