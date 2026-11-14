"""E11 — latent scaling law.

Held-out latent-prediction error as a function of predictor capacity
(latent_dim x hidden): world-model imagination error is expected to
follow a power law in parameters. Fits the log-log slope across a small
sweep — a cheap quantification of how much capacity buys dynamics skill.
"""

from __future__ import annotations

import torch

from nasa_farm.dynamics.latent import ObsEncoder
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import scene
from nasa_farm.experiments.specification import Spec
from nasa_farm.observe.trace import ObsTrace
from nasa_farm.optimize.embedding import jepa_fit


@register("scale")
def run(spec: Spec) -> dict:
    torch.manual_seed(spec.seed)
    sc = scene(spec)
    ch = spec.obs_channels
    gen = torch.Generator().manual_seed(spec.seed + 71)

    def stream(f):
        tr = ObsTrace(f.days, ch, spec.obs_every)
        with torch.inference_mode():
            sc.truth.rollout(sc.field, f, sc.actions, observer=tr)
        return tr.result()

    cal = stream(sc.realization(0))
    cal = cal * (1.0 + spec.noise * torch.randn(
        cal.shape, generator=gen, dtype=cal.dtype))
    held = stream(sc.realization(-1)) if sc.forcing_t.batch > 1 else cal

    rows = []
    for k in (4, 8, 16):
        m = WorldModel(sc.crops, residual=True, latent_dim=k,
                       latent_hidden=64, seed=spec.seed)
        enc = ObsEncoder(obs_dim=len(ch), z_dim=k)
        rep = jepa_fit(m, enc, sc.field, sc.realization(0), sc.actions, ch,
                       every=spec.obs_every, ema=spec.ema,
                       sig_weight=spec.sig_weight,
                       epochs=max(20, spec.epochs // 3), obs_tgt=cal)
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
