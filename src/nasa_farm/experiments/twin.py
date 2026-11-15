"""Twin experiment — parameter identifiability and calibration skill.

Nature (truth) is the physics core with spec.crop's traits scaled by
spec.perturb; the forecaster calibrates on sparse noisy observations of
one realization and is then scored on every held-out realization — the
classic OSSE protocol, so reported skill is out-of-sample by construction.
"""

from __future__ import annotations

import torch

from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import calibrate, observe, scene
from nasa_farm.experiments.specification import Specification
from nasa_farm.heads import METRICS
from nasa_farm.metrics import score
from nasa_farm.optimize.targets import season_metrics


def _channel(model, scenario, ch: str) -> torch.Tensor:
    """(R·H,) seasonal metric across held-out realizations only — the
    calibration row (row 0) is excluded from verification."""
    i = METRICS.index(ch)
    f_held = scenario.held_out()
    r = f_held.batch
    m = season_metrics(model, scenario.field.repeat(r), f_held, scenario.actions.repeat(r))
    return m[..., i].reshape(-1)


@register("twin")
def run(specification: Specification) -> dict:
    scenario = scene(specification)
    observed = observe(specification, scenario)
    cal, rep = calibrate(specification, scenario, observed)

    output: dict = {"calibration": rep, "channels": {}}
    for ch in specification.channels:
        t = _channel(scenario.truth, scenario, ch)
        s_p = score(_channel(scenario.prior, scenario, ch), t)
        s_c = score(_channel(cal, scenario, ch), t)
        red = 100.0 * (s_p["rmse"] - s_c["rmse"]) / s_p["rmse"] if s_p["rmse"] > 0 else 0.0
        output["channels"][ch] = {"prior": s_p, "cal": s_c, "rmse_reduction_pct": round(red, 2)}
    return output
