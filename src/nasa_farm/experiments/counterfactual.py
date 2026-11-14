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

from nasa_farm.dynamics.counterfactual import counterfactual
from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import scene
from nasa_farm.experiments.specification import Specification
from nasa_farm.metrics import rmse
from nasa_farm.observe.trace import ObservationTrace
from nasa_farm.types import Actions


@register("cf")
def run(specification: Specification) -> dict:
    torch.manual_seed(specification.seed)
    scenario = scene(specification)
    f0, ch = scenario.realization(next(iter(scenario.held_indices()))), specification.observed_channels
    generation = torch.Generator().manual_seed(specification.seed + 57)

    # factual obs of the truth realization (noisy, sparse cadence)
    traits = ObservationTrace(f0.days, ch, specification.cadence)
    with torch.inference_mode():
        scenario.truth.rollout(scenario.field, f0, scenario.actions, observer=traits)
    observed = traits.result() * (1.0 + specification.noise * torch.randn(
        traits.result().shape, generator=generation, dtype=traits.result().dtype))

    # alternative management: +15 mm irrigation every 10 days
    t = f0.days
    pulse = ((torch.arange(t) % 10) == 0).to(observed.dtype) * 15.0
    alt = Actions(sow=scenario.actions.sow,
                  irrigation=scenario.actions.irrigation + pulse,
                  n_fert=scenario.actions.n_fert)

    tr_t = ObservationTrace(t, ch, specification.cadence)
    with torch.inference_mode():
        scenario.truth.rollout(scenario.field, f0, alt, observer=tr_t)
    actual = tr_t.result()

    cf, information = counterfactual(scenario.prior, scenario.field, f0, scenario.actions, alt,
                                     observed, ch, every=specification.cadence)
    output = {}
    for j, c in enumerate(ch):
        e_cf = float(rmse(cf[0, :, j], actual[0, :, j]))
        e_nv = float(rmse(information["pred_alt"][0, :, j], actual[0, :, j]))
        output[c] = {"naive": round(e_nv, 6), "cf": round(e_cf, 6),
                  "improvement_pct": round(100 * (e_nv - e_cf) / e_nv, 2)
                  if e_nv > 0 else 0.0}
    return {"channels": output}
