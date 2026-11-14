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
from nasa_farm.exp.run import register
from nasa_farm.exp.scene import scene
from nasa_farm.exp.spec import Spec
from nasa_farm.observe.trace import ObsTrace
from nasa_farm.optimize.sindy import identify
from nasa_farm.optimize.trainer import fit


@register("discover")
def run(spec: Spec) -> dict:
    torch.manual_seed(spec.seed)
    sc = scene(spec)
    f0, ch = sc.realization(0), spec.obs_channels
    gen = torch.Generator().manual_seed(spec.seed + 63)

    tr = ObsTrace(f0.days, ch, spec.obs_every)
    with torch.inference_mode():
        sc.truth.rollout(sc.field, f0, sc.actions, observer=tr)
    obs_tgt = tr.result() * (1.0 + spec.noise * torch.randn(
        tr.result().shape, generator=gen, dtype=tr.result().dtype))

    m = WorldModel(sc.crops, residual=True, seed=spec.seed)
    params = {f"p{i}": p for i, p in enumerate(m.residual.parameters())}
    scale = obs_tgt.std(dim=1, keepdim=True).clamp(min=1e-6)

    def loss() -> torch.Tensor:
        tr2 = ObsTrace(f0.days, ch, spec.obs_every)
        m.rollout(sc.field, f0, sc.actions, observer=tr2)
        return (((tr2.result() - obs_tgt) / scale) ** 2).mean()

    res = fit(loss, params, epochs=spec.epochs, lr=spec.lr)
    rep = identify(m, sc.field, f0, sc.actions)
    return {"fit_loss": round(res.best_loss, 8),
            "r2_symbolic": rep["r2"],
            "equations": rep["equations"],
            "extra": list(spec.extra)}
