"""E10 — process rediscovery.

The truth model runs with extra processes (spec.extra) the forecaster
does not know; a ResidualNet is fitted against the truth's observable
stream, then SINDy distills the learned correction into a sparse
equation. Reports the recovered terms and the symbolic surrogate's R² —
the path from "the physics missed something" to "here is what it missed".
"""

from __future__ import annotations

import torch

from nasa_farm.dynamics.model import WorldModel
from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import scene
from nasa_farm.experiments.specification import Specification
from nasa_farm.observe.trace import ObservationTrace
from nasa_farm.optimize.identification import identify
from nasa_farm.optimize.trainer import fit


@register("discover")
def run(specification: Specification) -> dict:
    torch.manual_seed(specification.seed)
    scenario = scene(specification)
    f0, ch = scenario.realization(0), specification.observed_channels
    generation = torch.Generator().manual_seed(specification.seed + 63)

    traits = ObservationTrace(f0.days, ch, specification.cadence)
    with torch.inference_mode():
        scenario.truth.rollout(scenario.field, f0, scenario.actions, observer=traits)
    targets = traits.result() * (1.0 + specification.noise * torch.randn(
        traits.result().shape, generator=generation, dtype=traits.result().dtype))

    m = WorldModel(scenario.crops, residual=True, seed=specification.seed)
    parameters = {f"p{i}": p for i, p in enumerate(m.residual.parameters())}
    scale = targets.std(dim=1, keepdim=True).clamp(min=1e-6)

    def loss() -> torch.Tensor:
        tr2 = ObservationTrace(f0.days, ch, specification.cadence)
        m.rollout(scenario.field, f0, scenario.actions, observer=tr2)
        return (((tr2.result() - targets) / scale) ** 2).mean()

    result = fit(loss, parameters, epochs=specification.epochs, rate=specification.rate)
    rep = identify(m, scenario.field, f0, scenario.actions)
    return {"fit_loss": round(result.best_loss, 8),
            "r2_symbolic": rep["r2"],
            "equations": rep["equations"],
            "extra": list(specification.extra)}
