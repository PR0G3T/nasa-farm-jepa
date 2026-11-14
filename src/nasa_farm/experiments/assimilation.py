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

from nasa_farm.assimilation.kalman import STATE_VARS, _get_var, enkf_update
from nasa_farm.assimilation.score import ensf_update
from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import scene
from nasa_farm.experiments.specification import Spec
from nasa_farm.heads import METRICS, SeasonAccumulator
from nasa_farm.metrics import crps, rmse, score_ens
from nasa_farm.observe import observables
from nasa_farm.state.crops import IDX, CropLibrary
from nasa_farm.state.field import Field
from nasa_farm.types import Actions

_OBS_STD_FLOOR = 1e-3
_STRUCT = ("sow_doy", "legume", "vern_req", "t_kill", "crit_daylen",
           "pp_sens", "gdd_em")
"""Traits not jittered: discrete/structural switches and sowing date."""


def _ic_ensemble(field: Field, spec: Spec) -> Field:
    """Member spread from jittered initial conditions: soil water ±15 %,
    mineral N ±30 %, clamped to hydraulic bounds — the uncertainty the
    analysis is supposed to shrink."""
    f = field.repeat(spec.members)
    g = torch.Generator().manual_seed(spec.seed + 101)
    st = f.state

    def jit(x: torch.Tensor, rel: float) -> torch.Tensor:
        return x * (1.0 + rel * torch.randn(
            x.shape, generator=g, dtype=DTYPE))

    sw = torch.minimum(torch.maximum(jit(st.sw, 0.15), f.soil.air),
                       f.soil.sat)
    n_min = jit(st.n_min, 0.30).clamp(min=0.0)
    return Field(state=st.replace(sw=sw, n_min=n_min), soil=f.soil,
                 lat=f.lat, elev=f.elev, ctx=f.ctx)


def _truth(sc, spec: Spec, forcing, obs_days: set[int],
           gen: torch.Generator) -> tuple[dict, dict, torch.Tensor]:
    """Truth rollout → at each obs day: noisy observables + obs std +
    the hidden state on every analysis variable."""
    tr_obs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    tr_state: dict[int, dict[str, torch.Tensor]] = {}
    acc = SeasonAccumulator(1, spec.horizon)
    f = sc.field
    for t in range(forcing.days):
        f, fl = sc.truth.step(f, sc.actions.day(t), forcing.day(t), t)
        acc(t, f.state, fl)
        if t in obs_days:
            o = observables(f.state, spec.obs_channels)[0]
            std = (spec.noise * o.abs()).clamp(min=_OBS_STD_FLOOR)
            tr_obs[t] = (o + torch.randn(
                o.shape, generator=gen, dtype=DTYPE) * std, std)
            tr_state[t] = {v: _get_var(f.state, v)[0] for v in STATE_VARS}
    return tr_obs, tr_state, acc.result()[0]


def _member_model(sc, spec: Spec) -> WorldModel:
    """E-member parameter ensemble: one library row per member — spec.crop
    traits × U(0.9, 1.1) jitter (structural traits restored). Members sow
    their own row index, so a single batched rollout carries per-member
    physics with zero Python overhead."""
    i_c = sc.crops.index(spec.crop)
    base = sc.crops.table[i_c]
    g = torch.Generator().manual_seed(spec.seed + 131)
    tbl = base.unsqueeze(0).repeat(spec.members, 1)
    tbl = tbl * (0.9 + 0.2 * torch.rand(
        tbl.shape, generator=g, dtype=DTYPE))
    for p in _STRUCT:
        tbl[:, IDX[p]] = base[IDX[p]]
    names = tuple(f"{spec.crop}#{i}" for i in range(spec.members))
    return WorldModel(CropLibrary(table=tbl, names=names), residual=False)


def _member_actions(spec: Spec, base: torch.Tensor) -> Actions:
    """Member i sows library row i at the crop's default DOY each year."""
    a = Actions.idle(spec.members, spec.horizon * DAYS_PER_YEAR)
    sow_doy = int(base[IDX["sow_doy"]]) - 1
    members = torch.arange(spec.members)
    for y in range(spec.horizon):
        a.sow[members, y * DAYS_PER_YEAR + sow_doy] = members
    return a


def _update(fn, state, pred, obs, r, gen):
    """Dispatch: enkf takes perturb/gen; ensf's deterministic flow doesn't."""
    try:
        return fn(state, pred, obs, r, perturb=True, gen=gen)
    except TypeError:
        return fn(state, pred, obs, r)


def _variant(sc, spec: Spec, forcing_t, tr_obs: dict, tr_state: dict,
             tr_season: torch.Tensor, update) -> dict:
    e, days = spec.members, forcing_t.days
    gen = torch.Generator().manual_seed(spec.seed + 77)
    model = _member_model(sc, spec)
    acts = _member_actions(spec, sc.crops.table[sc.crops.index(spec.crop)])
    fld = _ic_ensemble(sc.field, spec)
    forcing = forcing_t.repeat(e)
    acc = SeasonAccumulator(e, spec.horizon)
    errs, crs, spr = [], [], []
    serrs: dict[str, list[float]] = {v: [] for v in STATE_VARS}

    with torch.inference_mode():
        for t in range(days):
            fld, fl = model.step(fld, acts.day(t), forcing.day(t), t)
            acc(t, fld.state, fl)
            if t not in tr_obs:
                continue
            obs, std = tr_obs[t]
            pred = observables(fld.state, spec.obs_channels)   # (E, C)
            errs.append(float(rmse(pred.mean(0), obs)))
            crs.append(float(crps(pred.t(), obs)))
            spr.append(float(pred.std(0).mean()))
            for v in STATE_VARS:
                serrs[v].append(float(rmse(_get_var(fld.state, v),
                                           tr_state[t][v])))
            if update is None:
                continue
            st = _update(update, fld.state, pred, obs, std ** 2, gen)
            fld = Field(state=st, soil=fld.soil, lat=fld.lat,
                        elev=fld.elev, ctx=fld.ctx)

    m = acc.result()                                    # (E, H, M)
    idx = {n: i for i, n in enumerate(METRICS)}
    ch_scores = {}
    for ch in spec.channels:
        i = idx[ch]
        ch_scores[ch] = score_ens(m[:, :, i].t(), tr_season[:, i],
                                  spec.alphas)
    mean = lambda v: round(sum(v) / len(v), 6) if v else 0.0
    return {"obs_rmse": mean(errs), "obs_crps": mean(crs),
            "spread": mean(spr),
            "state_rmse": {v: mean(s) for v, s in serrs.items()},
            "channels": ch_scores}


def _delta(base: dict, new: dict) -> dict:
    """% reduction of `new` vs `base` on every shared scalar leaf."""
    out = {}
    for k, v in base.items():
        if isinstance(v, dict) and k in new:
            out[k] = _delta(v, new[k])
        elif isinstance(v, (int, float)) and isinstance(
                new.get(k), (int, float)):
            out[k] = round(100 * (v - new[k]) / max(abs(v), 1e-9), 2)
    return out


@register("assim")
def run(spec: Spec) -> dict:
    sc = scene(spec)
    forcing_t = sc.realization(0)
    obs_days = set(range(spec.obs_every, forcing_t.days, spec.obs_every))
    gen = torch.Generator().manual_seed(spec.seed + 55)
    tr_obs, tr_state, tr_season = _truth(sc, spec, forcing_t, obs_days, gen)

    out = {name: _variant(sc, spec, forcing_t, tr_obs, tr_state,
                          tr_season, upd)
           for name, upd in (("open", None), ("enkf", enkf_update),
                             ("ensf", ensf_update))}
    out["reduction_pct"] = {
        "enkf": _delta(out["open"], out["enkf"]),
        "ensf": _delta(out["open"], out["ensf"])}
    return out
