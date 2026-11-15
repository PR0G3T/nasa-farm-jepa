"""E11 — latent scaling law.

Held-out latent-prediction error as a function of predictor capacity
(latent_dim x hidden): world-model imagination error is expected to
follow a power law in parameters. Each size trains on the calibration
stream only; the reported loss is the one-step latent prediction error
recomputed on the last held-out realization — the slope quantifies how
much capacity buys *out-of-sample* dynamics skill.
"""

from __future__ import annotations

import torch

from nasa_farm.dynamics.latent import ObservationEncoder
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import scene
from nasa_farm.experiments.specification import Specification
from nasa_farm.observe.trace import EmbeddingTrace, ObservationTrace
from nasa_farm.optimize.embedding import embedding_fit, seeded


@register("scale")
def run(specification: Specification) -> dict:
    torch.manual_seed(specification.seed)
    scenario = scene(specification)
    ch = specification.observed_channels
    generation = torch.Generator().manual_seed(specification.seed + 71)

    def stream(f):
        traits = ObservationTrace(f.days, ch, specification.cadence)
        with torch.inference_mode():
            scenario.truth.rollout(scenario.field, f, scenario.actions, observer=traits)
        return traits.result()

    cal = stream(scenario.realization(0))
    cal = cal * (
        1.0 + specification.noise * torch.randn(cal.shape, generator=generation, dtype=cal.dtype)
    )
    held_i = list(scenario.held_indices())[-1]
    f_held, held = scenario.realization(held_i), stream(scenario.realization(held_i))

    def held_loss(m, encoder, rep) -> float:
        """One-step latent prediction error on the held-out stream."""
        with torch.inference_mode():
            f = seeded(
                scenario.field, m.latent.seed(encoder(held[:, 0])), m.latent.hidden_dimension
            )
            traits = EmbeddingTrace(f_held.days, m.latent.dim, ch, every=specification.cadence)
            m.rollout(f, f_held, scenario.actions, observer=traits)
            _, z_h = traits.result()
            z_t = rep["tgt"](held)
            return float((z_h[:, 1:] - z_t[:, 1:]).pow(2).mean())

    rows = []
    for k in (4, 8, 16):
        m = WorldModel(
            scenario.crops,
            residual=True,
            latent_dimension=k,
            latent_hidden=64,
            seed=specification.seed,
        )
        encoder = ObservationEncoder(observed_dimension=len(ch), z_dim=k)
        rep = embedding_fit(
            m,
            encoder,
            scenario.field,
            scenario.realization(0),
            scenario.actions,
            ch,
            every=specification.cadence,
            ema=specification.ema,
            sig_weight=specification.sig_weight,
            epochs=max(20, specification.epochs // 3),
            targets=cal,
        )
        n_par = sum(p.numel() for p in m.latent.parameters())
        rows.append((k, n_par, rep["best_loss"], held_loss(m, encoder, rep)))
    # log-log slope of held-out prediction loss vs capacity
    xs = torch.log(torch.tensor([r[1] for r in rows], dtype=torch.float64))
    ys = torch.log(torch.tensor([max(r[3], 1e-12) for r in rows], dtype=torch.float64))
    slope = float(
        ((xs - xs.mean()) * (ys - ys.mean())).sum() / (xs - xs.mean()).pow(2).sum().clamp(min=1e-12)
    )
    return {
        "sizes": [
            {"latent_dim": k, "params": n, "cal_loss": b, "held_loss": h} for k, n, b, h in rows
        ],
        "power_slope": round(slope, 4),
    }
