"""Assimilation experiment — does satellite updating beat open-loop?

Twin protocol at observation cadence: the truth model rolls the held-out
realization, emitting noisy spec.obs_channels observables every
spec.obs_every days. An E-member prior ensemble (jittered soil-water /
mineral-N initial conditions, uncalibrated traits) runs the same weather
and is analyzed at each obs time by EnKF, by the kernel-score filter
(EnSF), or not at all (open loop).

All errors are measured *before* the analysis — pre-analysis skill, the
honest number. Skill is reported three ways: observable-space RMSE/CRPS,
state-space RMSE on every analysis variable vs the hidden truth state
(the actual question DA answers), and end-of-season metric scores.

Ensemble members carry *parameter* spread (±10 % trait jitter) on top of
jittered initial conditions — without it the ensemble collapses to zero
spread within weeks, the cross-covariance vanishes, and no filter can
update anything. Parametric ensembles are how operational ag-DA works.
"""

from __future__ import annotations

import torch

from nasa_farm.assimilation.kalman import STATE_VARS, _get_variable, kalman_update
from nasa_farm.assimilation.score import score_update
from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import scene
from nasa_farm.experiments.specification import Specification
from nasa_farm.heads import METRICS, SeasonAccumulator
from nasa_farm.metrics import crps, rmse, score_ens
from nasa_farm.observe import observables
from nasa_farm.state.crops import INDEX, CropLibrary
from nasa_farm.state.field import Field
from nasa_farm.types import Actions

_OBS_STD_FLOOR = 1e-3
_STRUCT = ("sow_doy", "legume", "vern_req", "t_kill", "crit_daylen",
           "pp_sens", "gdd_em")
"""Traits not jittered: discrete/structural switches and sowing date."""


def _ic_ensemble(field: Field, specification: Specification) -> Field:
    """Member spread from jittered initial conditions: soil water ±15 %,
    mineral N ±30 %, clamped to hydraulic bounds — the uncertainty the
    analysis is supposed to shrink."""
    f = field.repeat(specification.members)
    g = torch.Generator().manual_seed(specification.seed + 101)
    st = f.state

    def jit(x: torch.Tensor, relative: float) -> torch.Tensor:
        return x * (1.0 + relative * torch.randn(
            x.shape, generator=g, dtype=DTYPE))

    sw = torch.minimum(torch.maximum(jit(st.sw, 0.15), f.soil.air),
                       f.soil.sat)
    n_min = jit(st.n_min, 0.30).clamp(min=0.0)
    return Field(state=st.replace(sw=sw, n_min=n_min), soil=f.soil,
                 latitude=f.latitude, elevation=f.elevation, context=f.context)


def _truth(sc, specification: Specification, forcing, obs_days: set[int],
           generation: torch.Generator) -> tuple[dict, dict, torch.Tensor]:
    """Truth rollout → at each obs day: noisy observables + obs std +
    the hidden state on every analysis variable."""
    tr_obs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    tr_state: dict[int, dict[str, torch.Tensor]] = {}
    accumulate = SeasonAccumulator(1, specification.horizon)
    f = sc.field
    for t in range(forcing.days):
        f, fl = sc.truth.step(f, sc.actions.day(t), forcing.day(t), t)
        accumulate(t, f.state, fl)
        if t in obs_days:
            o = observables(f.state, specification.observed_channels)[0]
            std = (specification.noise * o.abs()).clamp(min=_OBS_STD_FLOOR)
            tr_obs[t] = (o + torch.randn(
                o.shape, generator=generation, dtype=DTYPE) * std, std)
            tr_state[t] = {v: _get_variable(f.state, v)[0] for v in STATE_VARS}
    return tr_obs, tr_state, accumulate.result()[0]


def _member_model(sc, specification: Specification) -> WorldModel:
    """E-member parameter ensemble: one library row per member — spec.crop
    traits × U(0.9, 1.1) jitter (structural traits restored). Members sow
    their own row index, so a single batched rollout carries per-member
    physics with zero Python overhead."""
    i_c = sc.crops.index(specification.crop)
    base = sc.crops.table[i_c]
    g = torch.Generator().manual_seed(specification.seed + 131)
    table = base.unsqueeze(0).repeat(specification.members, 1)
    table = table * (0.9 + 0.2 * torch.rand(
        table.shape, generator=g, dtype=DTYPE))
    for p in _STRUCT:
        table[:, INDEX[p]] = base[INDEX[p]]
    names = tuple(f"{specification.crop}#{i}" for i in range(specification.members))
    return WorldModel(CropLibrary(table=table, names=names), residual=False)


def _member_actions(specification: Specification, base: torch.Tensor) -> Actions:
    """Member i sows library row i at the crop's default DOY each year."""
    a = Actions.idle(specification.members, specification.horizon * DAYS_PER_YEAR)
    sow_doy = int(base[INDEX["sow_doy"]]) - 1
    members = torch.arange(specification.members)
    for y in range(specification.horizon):
        a.sow[members, y * DAYS_PER_YEAR + sow_doy] = members
    return a


def _update(function, state, predicted, observed, r, generation):
    """Dispatch: enkf takes perturb/gen; ensf's deterministic flow doesn't."""
    try:
        return function(state, predicted, observed, r, perturb=True, generation=generation)
    except TypeError:
        return function(state, predicted, observed, r)


def _variant(sc, specification: Specification, forcing_time, tr_obs: dict, tr_state: dict,
             tr_season: torch.Tensor, update) -> dict:
    e, days = specification.members, forcing_time.days
    generation = torch.Generator().manual_seed(specification.seed + 77)
    model = _member_model(sc, specification)
    acts = _member_actions(specification, sc.crops.table[sc.crops.index(specification.crop)])
    field = _ic_ensemble(sc.field, specification)
    forcing = forcing_time.repeat(e)
    accumulate = SeasonAccumulator(e, specification.horizon)
    errs, crs, spr = [], [], []
    serrs: dict[str, list[float]] = {v: [] for v in STATE_VARS}

    with torch.inference_mode():
        for t in range(days):
            field, fl = model.step(field, acts.day(t), forcing.day(t), t)
            accumulate(t, field.state, fl)
            if t not in tr_obs:
                continue
            observed, std = tr_obs[t]
            predicted = observables(field.state, specification.observed_channels)   # (E, C)
            errs.append(float(rmse(predicted.mean(0), observed)))
            crs.append(float(crps(predicted.t(), observed)))
            spr.append(float(predicted.std(0).mean()))
            for v in STATE_VARS:
                serrs[v].append(float(rmse(_get_variable(field.state, v),
                                           tr_state[t][v])))
            if update is None:
                continue
            st = _update(update, field.state, predicted, observed, std ** 2, generation)
            field = Field(state=st, soil=field.soil, latitude=field.latitude,
                        elevation=field.elevation, context=field.context)

    m = accumulate.result()                                    # (E, H, M)
    mean = lambda v: round(sum(v) / len(v), 6) if v else 0.0
    return {"obs_rmse": mean(errs), "obs_crps": mean(crs),
            "spread": mean(spr),
            "state_rmse": {v: mean(s) for v, s in serrs.items()},
            "ensemble": m, "truth": tr_season}


def _delta(base: dict, new: dict) -> dict:
    """% reduction of `new` vs `base` on every shared scalar leaf."""
    output = {}
    for k, v in base.items():
        if isinstance(v, dict) and k in new:
            output[k] = _delta(v, new[k])
        elif isinstance(v, (int, float)) and isinstance(
                new.get(k), (int, float)):
            output[k] = round(100 * (v - new[k]) / max(abs(v), 1e-9), 2)
    return output


@register("assim")
def run(specification: Specification) -> dict:
    sc = scene(specification)
    held = list(sc.held_indices())
    pooled: dict[str, dict] = {}
    for i in held:
        forcing_time = sc.realization(i)
        obs_days = set(range(specification.cadence, forcing_time.days,
                             specification.cadence))
        generation = torch.Generator().manual_seed(specification.seed + 55)
        tr_obs, tr_state, tr_season = _truth(
            sc, specification, forcing_time, obs_days, generation)
        for name, update in (("open", None), ("enkf", kalman_update),
                             ("ensf", score_update)):
            v = _variant(sc, specification, forcing_time, tr_obs, tr_state,
                         tr_season, update)
            acc = pooled.setdefault(name, {"scalars": [], "m": [], "t": []})
            acc["scalars"].append({k: v[k] for k in
                                   ("obs_rmse", "obs_crps", "spread",
                                    "state_rmse")})
            acc["m"].append(v["ensemble"])
            acc["t"].append(v["truth"])

    index = {n: i for i, n in enumerate(METRICS)}
    generation = torch.Generator().manual_seed(specification.seed + 97)
    output = {}
    for name, acc in pooled.items():
        mean = lambda vals: round(sum(vals) / len(vals), 6)
        scalars = {
            "obs_rmse": mean([s["obs_rmse"] for s in acc["scalars"]]),
            "obs_crps": mean([s["obs_crps"] for s in acc["scalars"]]),
            "spread": mean([s["spread"] for s in acc["scalars"]]),
            "state_rmse": {
                v: mean([s["state_rmse"][v] for s in acc["scalars"]])
                for v in STATE_VARS},
        }
        ch_scores = {}
        for ch in specification.channels:
            i = index[ch]
            ensemble = torch.cat([m[:, :, i].t() for m in acc["m"]])
            truth = torch.cat([t[:, i] for t in acc["t"]])
            ch_scores[ch] = score_ens(ensemble, truth, specification.alphas,
                                     generation=generation)
        output[name] = {**scalars, "channels": ch_scores}
    output["reduction_pct"] = {
        "enkf": _delta(output["open"], output["enkf"]),
        "ensf": _delta(output["open"], output["ensf"])}
    output["held_realizations"] = len(held)
    return output
