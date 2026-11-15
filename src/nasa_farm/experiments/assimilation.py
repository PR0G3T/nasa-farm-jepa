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
from nasa_farm.dynamics.latent import ObservationEncoder
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.experiments.hybrid import obs_fit, trace
from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import scene
from nasa_farm.experiments.specification import Specification
from nasa_farm.heads import METRICS, SeasonAccumulator
from nasa_farm.metrics import crps, rmse, score_ens
from nasa_farm.observe import observables
from nasa_farm.optimize.embedding import embedding_fit, seeded
from nasa_farm.state.crops import INDEX, CropLibrary
from nasa_farm.state.field import Field
from nasa_farm.types import Actions

_OBS_STD_FLOOR = 1e-3
_STRUCT = ("sow_doy", "legume", "vern_req", "t_kill", "crit_daylen", "pp_sens", "gdd_em")
"""Traits not jittered: discrete/structural switches and sowing date."""


def _ic_ensemble(field: Field, specification: Specification) -> Field:
    """Member spread from jittered initial conditions: soil water ±15 %,
    mineral N ±30 %, clamped to hydraulic bounds — the uncertainty the
    analysis is supposed to shrink."""
    f = field.repeat(specification.members)
    g = torch.Generator().manual_seed(specification.seed + 101)
    saved = f.state

    def jit(x: torch.Tensor, relative: float) -> torch.Tensor:
        return x * (1.0 + relative * torch.randn(x.shape, generator=g, dtype=DTYPE))

    sw = torch.minimum(torch.maximum(jit(saved.sw, 0.15), f.soil.air), f.soil.sat)
    n_min = jit(saved.n_min, 0.30).clamp(min=0.0)
    return Field(
        state=saved.replace(sw=sw, n_min=n_min),
        soil=f.soil,
        latitude=f.latitude,
        elevation=f.elevation,
        context=f.context,
    )


def _truth(
    scenario, specification: Specification, forcing, obs_days: set[int], generation: torch.Generator
) -> tuple[dict, dict, torch.Tensor]:
    """Truth rollout → at each obs day: noisy observables + obs std +
    the hidden state on every analysis variable."""
    truth_obs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    truth_state: dict[int, dict[str, torch.Tensor]] = {}
    accumulate = SeasonAccumulator(1, specification.horizon)
    f = scenario.field
    for t in range(forcing.days):
        f, fluxes = scenario.truth.step(f, scenario.actions.day(t), forcing.day(t), t)
        accumulate(t, f.state, fluxes)
        if t in obs_days:
            o = observables(f.state, specification.observed_channels)[0]
            std = (specification.noise * o.abs()).clamp(min=_OBS_STD_FLOOR)
            truth_obs[t] = (o + torch.randn(o.shape, generator=generation, dtype=DTYPE) * std, std)
            truth_state[t] = {v: _get_variable(f.state, v)[0] for v in STATE_VARS}
    return truth_obs, truth_state, accumulate.result()[0]


def _member_library(scenario, specification: Specification) -> CropLibrary:
    """E-member parameter ensemble: one library row per member — spec.crop
    traits × U(0.9, 1.1) jitter (structural traits restored). Members sow
    their own row index, so a single batched rollout carries per-member
    physics with zero Python overhead."""
    i_c = scenario.crops.index(specification.crop)
    base = scenario.crops.table[i_c]
    g = torch.Generator().manual_seed(specification.seed + 131)
    table = base.unsqueeze(0).repeat(specification.members, 1)
    table = table * (0.9 + 0.2 * torch.rand(table.shape, generator=g, dtype=DTYPE))
    for p in _STRUCT:
        table[:, INDEX[p]] = base[INDEX[p]]
    names = tuple(f"{specification.crop}#{i}" for i in range(specification.members))
    return CropLibrary(table=table, names=names)


def _hybrid_arm(scenario, specification: Specification):
    """JEPA latent + ctx-residual fitted on the calibration stream
    (realization 0, the same path E6's hybrid arm takes), then transplanted
    onto the jittered member ensemble — the learned-correction arms.
    Members share the fitted weights; each realization seeds z from the
    encoder on its own first noisy observation (E6's eval convention).
    Returns (model, encoder, report), or None when the spec disables the
    latent."""
    k, ch = specification.latent_dimension, specification.observed_channels
    if not k:
        return None
    f0, a0 = scenario.realization(0), scenario.actions
    observed, _ = trace(
        scenario.truth, scenario.field, f0, a0, ch, every=specification.cadence
    )
    generation = torch.Generator().manual_seed(specification.seed + 31)
    targets = observed * (
        1.0
        + specification.noise
        * torch.randn(observed.shape, generator=generation, dtype=DTYPE)
    )
    fitted = WorldModel(
        scenario.crops, residual=True, latent_dimension=k, seed=specification.seed
    )
    encoder = ObservationEncoder(observed_dimension=len(ch), z_dim=k)
    rep = embedding_fit(
        fitted,
        encoder,
        scenario.field,
        f0,
        a0,
        ch,
        every=specification.cadence,
        ema=specification.ema,
        sig_weight=specification.sig_weight,
        epochs=specification.epochs,
        rate=specification.latent_rate,
        targets=targets,
    )
    f_z = seeded(
        scenario.field,
        fitted.latent.seed(encoder(targets[:, 0])).detach(),
        fitted.latent.hidden_dimension,
    )
    result = obs_fit(
        fitted,
        f_z,
        f0,
        a0,
        targets,
        ch,
        every=specification.cadence,
        epochs=specification.epochs,
        rate=specification.rate,
    )
    member = WorldModel(
        _member_library(scenario, specification),
        residual=True,
        latent_dimension=k,
        seed=specification.seed,
    )
    member.residual.load_state_dict(fitted.residual.state_dict())
    member.latent.load_state_dict(fitted.latent.state_dict())
    report = {
        "jepa": {
            "best_loss": rep["best_loss"],
            "epochs_run": rep["epochs_run"],
            "z_std_final": rep["z_std_final"],
            "sigreg_final": rep["sigreg_final"],
            "history": rep["history"],
        },
        "residual": {
            "best_loss": round(result.best_loss, 8),
            "epochs_run": result.epochs_run,
            "history": result.history,
        },
    }
    return member, encoder, report


def _member_actions(specification: Specification, base: torch.Tensor) -> Actions:
    """Member i sows library row i at the crop's default DOY each year."""
    a = Actions.idle(specification.members, specification.horizon * DAYS_PER_YEAR)
    sow_doy = int(base[INDEX["sow_doy"]]) - 1
    members = torch.arange(specification.members)
    for y in range(specification.horizon):
        a.sow[members, y * DAYS_PER_YEAR + sow_doy] = members
    return a


def _update(function, state, predicted, observed, r, generation, channels):
    """Dispatch analysis updates; tolerates update callables that accept a
    narrower signature."""
    try:
        return function(
            state, predicted, observed, r, channels=channels, perturb=True, generation=generation
        )
    except TypeError:
        return function(state, predicted, observed, r)


def _variant(
    scenario,
    specification: Specification,
    forcing_time,
    truth_obs: dict,
    truth_state: dict,
    truth_season: torch.Tensor,
    update,
    model: WorldModel,
    encoder: ObservationEncoder | None = None,
) -> dict:
    e, days = specification.members, forcing_time.days
    generation = torch.Generator().manual_seed(specification.seed + 77)
    actions = _member_actions(
        specification, scenario.crops.table[scenario.crops.index(specification.crop)]
    )
    field = _ic_ensemble(scenario.field, specification)
    if encoder is not None:
        # Seed every member's z from this realization's first noisy obs —
        # the same look-ahead convention E6's held-out eval uses.
        z = model.latent.seed(encoder(truth_obs[min(truth_obs)][0])).detach()
        field = seeded(field, z.repeat(e, 1), model.latent.hidden_dimension)
    forcing = forcing_time.repeat(e)
    accumulate = SeasonAccumulator(e, specification.horizon)
    series = {
        "day": [],
        "obs_rmse": [],
        "obs_crps": [],
        "spread": [],
        "state_rmse": {v: [] for v in STATE_VARS},
        "state_rmse_post": {v: [] for v in STATE_VARS},
        "innovation": {c: [] for c in specification.observed_channels},
    }

    with torch.inference_mode():
        for t in range(days):
            field, fluxes = model.step(field, actions.day(t), forcing.day(t), t)
            accumulate(t, field.state, fluxes)
            if t not in truth_obs:
                continue
            observed, std = truth_obs[t]
            predicted = observables(field.state, specification.observed_channels)  # (E, C)
            series["day"].append(t)
            series["obs_rmse"].append(float(rmse(predicted.mean(0), observed)))
            series["obs_crps"].append(float(crps(predicted.t(), observed)))
            series["spread"].append(float(predicted.std(0).mean()))
            for j, c in enumerate(specification.observed_channels):
                series["innovation"][c].append(float(observed[j] - predicted[:, j].mean()))
            for v in STATE_VARS:
                series["state_rmse"][v].append(
                    float(rmse(_get_variable(field.state, v), truth_state[t][v]))
                )
            if update is not None:
                saved = _update(
                    update,
                    field.state,
                    predicted,
                    observed,
                    std**2,
                    generation,
                    specification.observed_channels,
                )
                field = Field(
                    state=saved,
                    soil=field.soil,
                    latitude=field.latitude,
                    elevation=field.elevation,
                    context=field.context,
                )
            for v in STATE_VARS:
                series["state_rmse_post"][v].append(
                    float(rmse(_get_variable(field.state, v), truth_state[t][v]))
                )

    m = accumulate.result()  # (E, H, M)

    def mean(v):
        return round(sum(v) / len(v), 6) if v else 0.0

    return {
        "obs_rmse": mean(series["obs_rmse"]),
        "obs_crps": mean(series["obs_crps"]),
        "spread": mean(series["spread"]),
        "state_rmse": {v: mean(s) for v, s in series["state_rmse"].items()},
        "ensemble": m,
        "truth": truth_season,
        "series": series,
    }


def _pool_series(series: list[dict], channels: tuple[str, ...]) -> dict:
    """Mean over realizations at each obs index — identical obs days
    across realizations (same horizon, same cadence)."""

    def mean(lists: list[list[float]]) -> list[float]:
        return [round(sum(x) / len(x), 6) for x in zip(*lists, strict=True)]

    return {
        "day": series[0]["day"],
        "obs_rmse": mean([s["obs_rmse"] for s in series]),
        "obs_crps": mean([s["obs_crps"] for s in series]),
        "spread": mean([s["spread"] for s in series]),
        "state_rmse": {v: mean([s["state_rmse"][v] for s in series]) for v in STATE_VARS},
        "state_rmse_post": {
            v: mean([s["state_rmse_post"][v] for s in series]) for v in STATE_VARS
        },
        "innovation": {c: mean([s["innovation"][c] for s in series]) for c in channels},
    }


def _delta(base: dict, new: dict) -> dict:
    """% reduction of `new` vs `base` on every shared scalar leaf."""
    output = {}
    for k, v in base.items():
        if isinstance(v, dict) and k in new:
            d = _delta(v, new[k])
            if d:
                output[k] = d
        elif isinstance(v, (int, float)) and isinstance(new.get(k), (int, float)):
            output[k] = round(100 * (v - new[k]) / max(abs(v), 1e-9), 2)
    return output


@register("assim")
def run(specification: Specification) -> dict:
    torch.manual_seed(specification.seed)
    scenario = scene(specification)
    held = list(scenario.held_indices())
    physics = WorldModel(_member_library(scenario, specification), residual=False)
    arms = [
        ("open", None, physics, None),
        ("enkf", kalman_update, physics, None),
        ("ensf", score_update, physics, None),
    ]
    hybrid = _hybrid_arm(scenario, specification)
    if hybrid is not None:
        model_j, encoder, fit_report = hybrid
        arms += [
            ("jepa", None, model_j, encoder),
            ("jepa_enkf", kalman_update, model_j, encoder),
        ]

    pooled: dict[str, dict] = {}
    for i in held:
        forcing_time = scenario.realization(i)
        obs_days = set(range(specification.cadence, forcing_time.days, specification.cadence))
        generation = torch.Generator().manual_seed(specification.seed + 55)
        truth_obs, truth_state, truth_season = _truth(
            scenario, specification, forcing_time, obs_days, generation
        )
        for name, update, model, encoder in arms:
            v = _variant(
                scenario,
                specification,
                forcing_time,
                truth_obs,
                truth_state,
                truth_season,
                update,
                model,
                encoder,
            )
            entry = pooled.setdefault(name, {"scalars": [], "m": [], "t": [], "series": []})
            entry["scalars"].append(
                {k: v[k] for k in ("obs_rmse", "obs_crps", "spread", "state_rmse")}
            )
            entry["m"].append(v["ensemble"])
            entry["t"].append(v["truth"])
            entry["series"].append(v["series"])

    index = {n: i for i, n in enumerate(METRICS)}
    generation = torch.Generator().manual_seed(specification.seed + 97)
    output = {}
    for name, entry in pooled.items():

        def mean(values):
            return round(sum(values) / len(values), 6)

        scalars = {
            "obs_rmse": mean([s["obs_rmse"] for s in entry["scalars"]]),
            "obs_crps": mean([s["obs_crps"] for s in entry["scalars"]]),
            "spread": mean([s["spread"] for s in entry["scalars"]]),
            "state_rmse": {
                v: mean([s["state_rmse"][v] for s in entry["scalars"]]) for v in STATE_VARS
            },
        }
        ch_scores = {}
        for ch in specification.channels:
            i = index[ch]
            ensemble = torch.cat([m[:, :, i].t() for m in entry["m"]])
            truth = torch.cat([t[:, i] for t in entry["t"]])
            ch_scores[ch] = score_ens(ensemble, truth, specification.alphas, generation=generation)
        output[name] = {
            **scalars,
            "channels": ch_scores,
            "series": _pool_series(entry["series"], specification.observed_channels),
        }
    output["reduction_pct"] = {
        name: _delta(output["open"], output[name]) for name, *_ in arms if name != "open"
    }
    if hybrid is not None:
        output["reduction_pct"]["jepa_enkf_over_enkf"] = _delta(
            output["enkf"], output["jepa_enkf"]
        )
        output["reduction_pct"]["jepa_enkf_over_jepa"] = _delta(
            output["jepa"], output["jepa_enkf"]
        )
        output["hybrid_fit"] = fit_report
    output["held_realizations"] = len(held)
    return output
