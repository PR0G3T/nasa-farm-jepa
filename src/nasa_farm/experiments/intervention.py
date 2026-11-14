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
from nasa_farm.dynamics.intervention import EDIT_VARS, edit_state
from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import calibrate, observe, scene
from nasa_farm.experiments.specification import Specification
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
    keywords = {k: getattr(actions, k).clone() for k in ACTION_VARS}
    if channel == "sow":
        keywords["sow"] = keywords["sow"].roll(int(delta), dims=1)
    else:
        keywords[channel][:, day:day + span] += delta
    return Actions(**keywords)


def _metrics(model, scenario, forcing, actions) -> torch.Tensor:
    return season_metrics(model, scenario.field, forcing, actions)[0]  # (H, M)


@register("interv")
def run(specification: Specification) -> dict:
    scenario = scene(specification)
    observed = observe(specification, scenario)
    cal, rep = calibrate(specification, scenario, observed)
    held = list(scenario.held_indices())
    index = {n: i for i, n in enumerate(METRICS)}

    # --- action-perturbation fidelity -------------------------------------
    suite = [
        ("irrig+10d", "irrigation", 130, 10, 10.0),
        ("fert+40", "n_fert", 150, 1, 40.0),
        ("sow-14d", "sow", 0, 0, -14),
        ("sow+14d", "sow", 0, 0, 14),
    ]
    chans = ["yield_kg", "biomass_max", "water_used", "n_leached"]
    chans = [c for c in chans if c in index]
    pert: dict = {}
    for name, ch, day, span, d in suite:
        a2 = _perturbed(scenario.actions, ch, day, span, d)
        deltas: dict[str, torch.Tensor] = {}
        for label, model in (("prior", scenario.prior), ("cal", cal),
                           ("oracle", scenario.truth)):
            per_row = [
                (_metrics(model, scenario, scenario.realization(i), a2)
                 - _metrics(model, scenario, scenario.realization(i), scenario.actions)).mean(0)
                for i in held]
            deltas[label] = torch.stack(per_row).mean(0)
        truth_d = deltas.pop("oracle")
        for label in ("prior", "cal"):
            for c in chans:
                t, p = float(truth_d[index[c]]), float(deltas[label][index[c]])
                error = p - t
                pert.setdefault(c, {}).setdefault(name, {})[label] = {
                    "truth_delta": round(t, 4), "model_delta": round(p, 4),
                    "err": round(error, 4)}

    # --- plan-ranking fidelity ---------------------------------------------
    i_c = scenario.crops.index(specification.crop)
    cands = [i_c, scenario.crops.index("soybean"), scenario.crops.index("wheat")]
    hh = max(1, min(specification.horizon, 2))
    plans = enumerate_rotations(cands, hh)
    if hh * 365 < scenario.forcing_e.days:
        from nasa_farm.types import FORCING_VARS, Forcing
        ensemble = Forcing(*(getattr(scenario.forcing_e, k)[:, :hh * 365]
                        for k in FORCING_VARS))
    else:
        ensemble = scenario.forcing_e
    taus = {}
    with torch.inference_mode():
        ev_t = evaluate(scenario.truth, scenario.field, plans, ensemble)
        obj_t = scalarize(*objective_matrix(
            {k: v for k, v in ev_t.objectives.items()}), torch.ones(
            len(ev_t.objectives), dtype=DTYPE))
        for label, model in (("prior", scenario.prior), ("cal", cal)):
            ev_m = evaluate(model, scenario.field, plans, ensemble)
            obj_m = scalarize(*objective_matrix(
                {k: v for k, v in ev_m.objectives.items()}),
                torch.ones(len(ev_m.objectives), dtype=DTYPE))
            taus[label] = round(_tau(obj_m, obj_t), 4)

    # --- mid-season edit consistency ---------------------------------------
    e = len(EDIT_VARS)
    mask = torch.zeros(e, dtype=torch.bool)
    mask[EDIT_VARS.index("swr")] = True

    def run_edit(model, f_t):
        day = min(150, f_t.days - 60)
        f = scenario.field
        for t in range(day):
            f, _ = model.step(f, scenario.actions.day(t), f_t.day(t), t)
        current = f.state
        st_e = edit_state(current, mask,
                          torch.full((1, e), 400.0, dtype=DTYPE), f.soil)
        fe = Field(state=st_e, soil=f.soil, latitude=f.latitude, elevation=f.elevation,
                   context=f.context)
        edited = fe
        for t in range(day, f_t.days):
            edited, _ = model.step(edited, scenario.actions.day(t), f_t.day(t), t)
        return edited.state.biomass, edited.state.sw.sum(-1), edited.state.n_min

    response: dict = {}
    for i in held:
        f_t = scenario.realization(i)
        b_t, w_t, n_t = run_edit(scenario.truth, f_t)
        for label, model in (("prior", scenario.prior), ("cal", cal)):
            b_m, w_m, n_m = run_edit(model, f_t)
            row = {
                "biomass_err": float(b_m - b_t),
                "water_err": float(w_m - w_t),
                "nmin_err": float(n_m - n_t)}
            acc = response.setdefault(label, [])
            acc.append(row)
    response = {label: {
        k: round(sum(r[k] for r in rows) / len(rows), 4)
        for k in rows[0]} for label, rows in response.items()}

    return {"calibration": rep, "perturbations": pert,
            "plan_tau": taus, "edit_response": response,
            "held_realizations": len(held)}
