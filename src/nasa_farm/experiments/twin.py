"""Twin experiment — parameter identifiability and calibration skill.

Nature (truth) is the physics core with spec.crop's traits scaled by
spec.perturb; the forecaster calibrates on sparse noisy observations of
one realization and is then scored on every held-out realization — the
classic OSSE protocol, so reported skill is out-of-sample by construction.
"""

from __future__ import annotations

import torch

from nasa_farm.exp.run import register
from nasa_farm.exp.scene import calibrate, observe, scene
from nasa_farm.exp.spec import Spec
from nasa_farm.heads import METRICS
from nasa_farm.metrics import score
from nasa_farm.optimize.targets import season_metrics


def _channel(model, sc, ch: str) -> torch.Tensor:
    """(R·H,) seasonal metric across all held-out realizations."""
    i = METRICS.index(ch)
    r = sc.forcing_t.batch
    m = season_metrics(model, sc.field.repeat(r), sc.forcing_t,
                       sc.actions.repeat(r))
    return m[..., i].reshape(-1)


@register("twin")
def run(spec: Spec) -> dict:
    sc = scene(spec)
    obs = observe(spec, sc)
    cal, rep = calibrate(spec, sc, obs)

    out: dict = {"calibration": rep, "channels": {}}
    for ch in spec.channels:
        t = _channel(sc.truth, sc, ch)
        s_p = score(_channel(sc.prior, sc, ch), t)
        s_c = score(_channel(cal, sc, ch), t)
        red = (100.0 * (s_p["rmse"] - s_c["rmse"]) / s_p["rmse"]
               if s_p["rmse"] > 0 else 0.0)
        out["channels"][ch] = {"prior": s_p, "cal": s_c,
                               "rmse_reduction_pct": round(red, 2)}
    return out
