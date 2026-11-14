"""Bench experiment — predictive skill vs baselines on a held-out year.

The truth model rolls the held-out forcing realization once; each predictor
rolls the members ensemble and is scored as a probabilistic forecast:

- prior   — literature traits, uncalibrated (the baseline to beat)
- cal     — traits gradient-fitted on sparse noisy observations
- oracle  — the truth model itself; the irreducible-skill ceiling (its
            error is pure forcing uncertainty, not model bias)

Per channel: full deterministic suite on the ensemble mean plus CRPS,
PICP/interval-score/width, spread–skill and PIT reliability — everything a
paper table needs, from one rollout per predictor.
"""

from __future__ import annotations

import torch

from nasa_farm.exp.run import register
from nasa_farm.exp.scene import calibrate, observe, scene
from nasa_farm.exp.spec import Spec
from nasa_farm.heads import METRICS
from nasa_farm.metrics import score_ens
from nasa_farm.optimize.targets import season_metrics

PREDICTORS: tuple[str, ...] = ("prior", "cal", "oracle")


@register("bench")
def run(spec: Spec) -> dict:
    sc = scene(spec)
    obs = observe(spec, sc)
    cal, rep = calibrate(spec, sc, obs)

    s = sc.forcing_e.batch
    fld, acts = sc.field.repeat(s), sc.actions.repeat(s)
    idx = {n: i for i, n in enumerate(METRICS)}
    tgt = season_metrics(sc.truth, sc.field, sc.forcing_t, sc.actions)[0]
    gen = torch.Generator().manual_seed(spec.seed + 31)

    out: dict = {"calibration": rep, "channels": {}}
    for name, model in (("prior", sc.prior), ("cal", cal),
                        ("oracle", sc.truth)):
        ens = season_metrics(model, fld, sc.forcing_e, acts)
        for ch in spec.channels:
            # (S, H) → (H, S): score_ens wants members last
            e = ens[:, :, idx[ch]].t()
            out["channels"].setdefault(ch, {})[name] = score_ens(
                e, tgt[:, idx[ch]], spec.alphas, gen=gen)

    for ch, d in out["channels"].items():
        p, c, o = d["prior"], d["cal"], d["oracle"]
        d["summary"] = {
            "rmse_reduction_pct": round(
                100 * (p["rmse"] - c["rmse"]) / max(p["rmse"], 1e-9), 2),
            "crps_reduction_pct": round(
                100 * (p["crps"] - c["crps"]) / max(p["crps"], 1e-9), 2),
            "cal_vs_oracle_rmse_gap": round(c["rmse"] - o["rmse"], 6)}
    return out
