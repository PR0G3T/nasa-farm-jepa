"""E11 — latent scaling law.

Held-out latent-prediction error as a function of predictor capacity
(latent_dim x hidden): world-model imagination error is expected to
follow a power law in parameters. Fits the log-log slope across a small
sweep — a cheap quantification of how much capacity buys dynamics skill.
"""

from __future__ import annotations

import torch

from nasa_farm.dynamics.latent import ObservationEncoder
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import scene
from nasa_farm.experiments.specification import Specification
from nasa_farm.observe.trace import ObservationTrace
from nasa_farm.optimize.embedding import embedding_fit


@register("scale")
def run(specification: Specification) -> dict:
    torch.manual_seed(specification.seed)
    sc = scene(specification)
    ch = specification.observed_channels
    generation = torch.Generator().manual_seed(specification.seed + 71)

    def stream(f):
        traits = ObservationTrace(f.days, ch, specification.cadence)
        with torch.inference_mode():
            sc.truth.rollout(sc.field, f, sc.actions, observer=traits)
        return traits.result()

    cal = stream(sc.realization(0))
    cal = cal * (1.0 + specification.noise * torch.randn(
        cal.shape, generator=generation, dtype=cal.dtype))
    held = stream(sc.realization(-1)) if sc.forcing_time.batch > 1 else cal

    rows = []
    for k in (4, 8, 16):
        m = WorldModel(sc.crops, residual=True, latent_dimension=k,
                       latent_hidden=64, seed=specification.seed)
        encoder = ObservationEncoder(observed_dimension=len(ch), z_dim=k)
        rep = embedding_fit(m, encoder, sc.field, sc.realization(0), sc.actions, ch,
                       every=specification.cadence, ema=specification.ema,
                       sig_weight=specification.sig_weight,
                       epochs=max(20, specification.epochs // 3), targets=cal)
        n_par = sum(p.numel() for p in m.latent.parameters())
        rows.append((k, n_par, rep["best_loss"]))
    # log-log slope of held-out-consistent best loss vs capacity
    xs = torch.log(torch.tensor([r[1] for r in rows], dtype=torch.float64))
    ys = torch.log(torch.tensor([max(r[2], 1e-12) for r in rows],
                                dtype=torch.float64))
    slope = float(((xs - xs.mean()) * (ys - ys.mean())).sum()
                  / (xs - xs.mean()).pow(2).sum().clamp(min=1e-12))
    return {"sizes": [{"latent_dim": k, "params": n, "best_loss": b}
                      for k, n, b in rows],
            "power_slope": round(slope, 4)}
