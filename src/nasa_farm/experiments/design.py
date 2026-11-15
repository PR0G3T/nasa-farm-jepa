"""Observation design — does the measurement set fix identifiability?

The twin experiment shows trait calibration recovers wrong parameters
from seasonal yield alone (equifinality). This experiment makes that
failure formal — the singular spectrum of the exact trait-sensitivity
Jacobian J = dy/dθ over each observation set predicts it before any fit —
then tests the remedy: greedy E-optimal design over the satellite-channel
Jacobians picks the sub-seasonal series that maximize the weakest trait
direction, and the same variational fit driven by that series recovers
the traits the seasonal fit cannot.

Three calibration arms share one spec, one budget and one bounded
reparameterization — `seasonal` (spec.channels), `series` (the designed
channels at spec.cadence) and `union` (both) — so any difference in
recovery is attributable to the observation set alone. The design
machinery itself is exercised end to end: exact jacfwd Jacobians,
fisher_design per-channel gains and the ensemble-EIG acquisition
calendar.
"""

from __future__ import annotations

import torch

from nasa_farm.assimilation import (
    design_schedule,
    fisher_design,
    metric_jacobian,
    select_channels,
    series_jacobian,
)
from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import (
    calibrate,
    heldout_channel,
    observe,
    scene,
    trait_fit,
)
from nasa_farm.experiments.specification import Specification
from nasa_farm.heads import METRICS
from nasa_farm.metrics import score
from nasa_farm.observe import CHANNELS
from nasa_farm.optimize.targets import (
    ObservedSeries,
    season_metrics,
    series,
    series_loss,
    trajectory_loss,
)


def _spectrum(jacobian: torch.Tensor, scale: torch.Tensor) -> dict:
    """Singular spectrum of the noise-whitened Jacobian — σ_min/σ_max is
    the worst-to-best trait-direction signal ratio; a near-zero σ_min is
    the formal signature of equifinality."""
    j = jacobian / scale.reshape(-1, 1).clamp(min=1e-9)
    s = torch.linalg.svdvals(j)
    return {
        "singular": [round(float(v), 6) for v in s],
        "sigma_ratio": round(float(s[-1] / s[0].clamp(min=1e-12)), 8),
        "rank": int((s > 1e-6 * s[0]).sum()),
        "rows": int(jacobian.shape[0]),
    }


def _heldout(model, scenario, specification: Specification) -> dict:
    out = {}
    for ch in specification.channels:
        t = heldout_channel(scenario.truth, scenario, ch)
        out[ch] = score(heldout_channel(model, scenario, ch), t)
    return out


@register("design")
def run(specification: Specification) -> dict:
    scenario = scene(specification)
    traits = tuple(specification.perturb)
    if not traits:
        raise ValueError("design needs spec.perturb traits to analyze")
    i_c = scenario.crops.index(specification.crop)
    f0, a0 = scenario.realization(0), scenario.actions
    all_ch = tuple(CHANNELS)
    days = tuple(range(0, f0.days, specification.cadence))

    # --- formal identifiability: exact trait-sensitivity spectra --------
    with torch.inference_mode():
        y_all = series(
            scenario.prior, scenario.field, f0, a0, all_ch, every=specification.cadence
        )[0]
        y_m = season_metrics(scenario.prior, scenario.field, f0, a0)[0]
    j_all = series_jacobian(
        scenario.prior, scenario.field, f0, a0, i_c, traits, all_ch, specification.cadence
    )
    j_m = metric_jacobian(
        scenario.prior, scenario.field, f0, a0, i_c, traits, specification.channels
    )
    idx = [METRICS.index(c) for c in specification.channels]
    scale_all = y_all.abs().mean(0) * specification.noise
    scale_m = (y_m[:, idx].abs().mean(0) * specification.noise).repeat(y_m.shape[0])

    # --- greedy E-optimal channel set over the full candidate bank ------
    picked, _info = select_channels(j_all, scale_all, len(traits))
    designed = tuple(all_ch[c] for c in picked)
    j_d = j_all[:, picked, :].reshape(-1, len(traits))
    scale_d = scale_all[list(picked)].repeat(j_all.shape[0])

    # --- the two existing design estimators, exercised ------------------
    fisher_per_channel: dict[str, float] = {}
    for _t, c, gain in fisher_design(
        scenario.prior, scenario.field, f0, a0, traits, days, all_ch
    ):
        fisher_per_channel[c] = fisher_per_channel.get(c, 0.0) + gain
    fisher_top = sorted(fisher_per_channel.items(), key=lambda kv: -kv[1])[:10]

    s = scenario.forcing_e.batch
    calendar = design_schedule(
        scenario.prior,
        scenario.field.repeat(s),
        scenario.forcing_e,
        scenario.actions.repeat(s),
        days,
        all_ch,
        r=(scale_all.clamp(min=1e-9)) ** 2,
        n_pick=8,
    )

    # --- series targets from the truth rollout --------------------------
    with torch.inference_mode():
        target = series(
            scenario.truth, scenario.field, f0, a0, designed, every=specification.cadence
        )[0]
    generation = torch.Generator().manual_seed(specification.seed + 17)
    target = target * (
        1.0
        + specification.noise * torch.randn(target.shape, generator=generation, dtype=target.dtype)
    )
    observed_series = ObservedSeries(
        values=target, channel_scale=target.abs().mean(0).clamp(min=1e-6)
    )
    observed_seasonal = observe(specification, scenario)

    # --- calibration arms: identical fit, different observation set -----
    arms: dict[str, dict] = {}
    candidates = {}
    candidates["seasonal"], arms["seasonal"] = calibrate(
        specification, scenario, observed_seasonal
    )
    candidates["series"], arms["series"] = trait_fit(
        specification,
        scenario,
        lambda m: series_loss(
            series(m, scenario.field, f0, a0, designed, every=specification.cadence),
            observed_series,
        ),
    )
    candidates["union"], arms["union"] = trait_fit(
        specification,
        scenario,
        lambda m: series_loss(
            series(m, scenario.field, f0, a0, designed, every=specification.cadence),
            observed_series,
        )
        + trajectory_loss(season_metrics(m, scenario.field, f0, a0), observed_seasonal),
    )
    for name, model in candidates.items():
        arms[name]["heldout"] = _heldout(model, scenario, specification)

    return {
        "traits": list(traits),
        "designed": list(designed),
        "spectra": {
            "seasonal": _spectrum(j_m, scale_m),
            "designed": _spectrum(j_d, scale_d),
            "union": _spectrum(torch.cat([j_d, j_m]), torch.cat([scale_d, scale_m])),
        },
        "design": {
            "fisher_per_channel": [
                {"channel": c, "gain": round(g, 6)} for c, g in fisher_top
            ],
            "eig_calendar": [
                {"day": t, "channel": c, "gain": round(g, 6)} for t, c, g in calendar
            ],
        },
        "arms": arms,
    }
