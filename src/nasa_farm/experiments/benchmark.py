"""Bench experiment — predictive skill vs baselines on held-out years.

Each predictor rolls the members ensemble once; the truth model rolls
every held-out realization, so probabilistic scores (CRPS, PICP, PIT)
aggregate R·H genuine out-of-sample verifications:

- prior   — literature traits, uncalibrated (the baseline to beat)
- cal     — traits gradient-fitted on sparse noisy observations of the
            calibration realization
- oracle  — the truth model itself; the irreducible-skill ceiling (its
            error is pure forcing uncertainty, not model bias)
"""

from __future__ import annotations

import torch

from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import calibrate, observe, scene
from nasa_farm.experiments.specification import Specification
from nasa_farm.heads import METRICS
from nasa_farm.metrics import score_ens
from nasa_farm.optimize.targets import season_metrics

PREDICTORS: tuple[str, ...] = ("prior", "cal", "oracle")


@register("bench")
def run(specification: Specification) -> dict:
    sc = scene(specification)
    observed = observe(specification, sc)
    cal, rep = calibrate(specification, sc, observed)

    s, r = sc.forcing_e.batch, sc.held_out().batch
    field, acts = sc.field.repeat(s), sc.actions.repeat(s)
    index = {n: i for i, n in enumerate(METRICS)}
    target = season_metrics(sc.truth, sc.field.repeat(r), sc.held_out(),
                         sc.actions.repeat(r))           # (R, H, M)
    generation = torch.Generator().manual_seed(specification.seed + 31)

    output: dict = {"calibration": rep, "channels": {}}
    for name, model in (("prior", sc.prior), ("cal", cal),
                        ("oracle", sc.truth)):
        ensemble = season_metrics(model, field, sc.forcing_e, acts)  # (S, H, M)
        for ch in specification.channels:
            i = index[ch]
            # pair every realization-year with all members → (R·H, S)
            e = (ensemble[:, :, i].t().unsqueeze(0)
                 .expand(r, -1, -1).reshape(-1, s))
            output["channels"].setdefault(ch, {})[name] = score_ens(
                e, target[:, :, i].reshape(-1), specification.alphas, generation=generation)

    for ch, d in output["channels"].items():
        p, c, o = d["prior"], d["cal"], d["oracle"]
        d["summary"] = {
            "rmse_reduction_pct": round(
                100 * (p["rmse"] - c["rmse"]) / max(p["rmse"], 1e-9), 2),
            "crps_reduction_pct": round(
                100 * (p["crps"] - c["crps"]) / max(p["crps"], 1e-9), 2),
            "cal_vs_oracle_rmse_gap": round(c["rmse"] - o["rmse"], 6)}
    return output
