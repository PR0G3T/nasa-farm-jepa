"""E9 — counterfactual fidelity.

Observed prediction error is abducted as a persistent factor and
transferred to an alternative-action rollout: cf = obs + (pred' − pred).
The scored quantity is how much that one-subtraction estimate beats the
naive counterfactual (the prior rollout under the alternative action)
against the truth model's actual alternative rollout — the O(t²) claim,
measured.
"""

from __future__ import annotations

import torch

from nasa_farm.dynamics.cf import counterfactual
from nasa_farm.exp.run import register
from nasa_farm.exp.scene import scene
from nasa_farm.exp.spec import Spec
from nasa_farm.metrics import rmse
from nasa_farm.observe.trace import ObsTrace
from nasa_farm.types import Actions


@register("cf")
def run(spec: Spec) -> dict:
    torch.manual_seed(spec.seed)
    sc = scene(spec)
    f0, ch = sc.realization(0), spec.obs_channels
    gen = torch.Generator().manual_seed(spec.seed + 57)

    # factual obs of the truth realization (noisy, sparse cadence)
    tr = ObsTrace(f0.days, ch, spec.obs_every)
    with torch.inference_mode():
        sc.truth.rollout(sc.field, f0, sc.actions, observer=tr)
    obs = tr.result() * (1.0 + spec.noise * torch.randn(
        tr.result().shape, generator=gen, dtype=tr.result().dtype))

    # alternative management: +15 mm irrigation every 10 days
    t = f0.days
    pulse = ((torch.arange(t) % 10) == 0).to(obs.dtype) * 15.0
    alt = Actions(sow=sc.actions.sow,
                  irrigation=sc.actions.irrigation + pulse,
                  n_fert=sc.actions.n_fert)

    tr_t = ObsTrace(t, ch, spec.obs_every)
    with torch.inference_mode():
        sc.truth.rollout(sc.field, f0, alt, observer=tr_t)
    actual = tr_t.result()

    cf, info = counterfactual(sc.prior, sc.field, f0, sc.actions, alt,
                              obs, ch, every=spec.obs_every)
    out = {}
    for j, c in enumerate(ch):
        e_cf = float(rmse(cf[0, :, j], actual[0, :, j]))
        e_nv = float(rmse(info["pred_alt"][0, :, j], actual[0, :, j]))
        out[c] = {"naive": round(e_nv, 6), "cf": round(e_cf, 6),
                  "improvement_pct": round(100 * (e_nv - e_cf) / e_nv, 2)
                  if e_nv > 0 else 0.0}
    return {"channels": out}
