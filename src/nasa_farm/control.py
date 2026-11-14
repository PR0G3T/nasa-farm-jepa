"""Model-predictive control over daily irrigation / fertilization.

MPPI (Model-Predictive Path Integral): a mean action sequence is perturbed by
colored (Ornstein–Uhlenbeck) noise, candidates are scored by rollout reward
averaged over forcing scenarios, and the mean shifts toward a rank-weighted
recombination of the elite quartile. The incumbent is re-scored every round
and the best mean ever evaluated is returned — the result can never be worse
than `base`. No gradients needed — robust to ensemble forcing, and the reward
is any metric mix from `heads`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.forcing import ou_noise
from nasa_farm.heads import METRICS, SeasonAccumulator
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing

DEFAULT_WEIGHTS: dict[str, float] = {
    "yield_kg": 1.0, "water_used": -0.3, "n_leached": -2.0,
    "stress_days": -3.0, "frost_days": -50.0, "heat_dd": -2.0,
}


@dataclass(frozen=True, slots=True)
class ControlResult:
    actions: Actions
    reward: float           # best mean-scenario reward seen (>= base's)
    history: tuple[float, ...]  # running best, one entry per iteration


def metric_reward(metrics: torch.Tensor,
                  weights: dict[str, float] | None = None) -> torch.Tensor:
    """(S, E, H, M) → (S,) reward: weighted metric sum over years, mean over
    forcing scenarios."""
    weights = weights or DEFAULT_WEIGHTS
    idx = {n: i for i, n in enumerate(METRICS)}
    r = torch.zeros(metrics.shape[0], metrics.shape[1], dtype=DTYPE)
    for name, w in weights.items():
        r = r + w * metrics[..., idx[name]].sum(dim=-1)
    return r.mean(1)


def mppi(model, field: Field, forcing: Forcing, base: Actions,
         iters: int = 8, samples: int = 96, rho: float = 0.7,
         sigma: tuple[float, float] = (4.0, 10.0),
         bounds: tuple[tuple[float, float], tuple[float, float]] =
         ((0.0, 25.0), (0.0, 10.0)),
         weights: dict[str, float] | None = None,
         reward=None, seed: int = 0) -> ControlResult:
    """Optimize continuous irrigation (mm/day) and N (kg/ha/day) channels.

    `base` supplies sow/harvest structure; forcing may be an ensemble (E>1) —
    the reward is the scenario mean, so the schedule is robust by construction.
    `reward` overrides `metric_reward` for custom objectives.
    """
    days = forcing.days
    years = max(1, days // 365)
    E = forcing.batch
    gen = torch.Generator().manual_seed(seed)
    mu = torch.stack([base.irrigation[0], base.n_fert[0]])  # (2, T)
    best_r, best_mu = -math.inf, mu
    hist: list[float] = []

    for _ in range(iters):
        eps = torch.stack([ou_noise(samples, days, rho, gen),
                           ou_noise(samples, days, rho, gen)])      # (2, S, T)
        cand = torch.stack([(mu[c] + eps[c] * sigma[c]).clamp(*bounds[c])
                            for c in range(2)])                    # (2, S, T)
        cand[:, 0] = mu  # incumbent: the mean itself is scored each round
        b = samples * E
        acts = Actions(
            sow=base.sow.expand(b, days),
            irrigation=cand[0].repeat_interleave(E, dim=0),
            n_fert=cand[1].repeat_interleave(E, dim=0))
        fld, frc = field.repeat(b), forcing.repeat(samples)
        acc = SeasonAccumulator(b, years, device=fld.state.sw1.device)
        with torch.inference_mode():
            model.rollout(fld, frc, acts, observer=acc)
        m = acc.result().view(samples, E, years, -1)
        r = reward(m) if reward is not None else metric_reward(m, weights)
        r = r.view(samples, -1).mean(1)
        if float(r[0]) > best_r:
            best_r, best_mu = float(r[0]), mu
        hist.append(best_r)                  # best reward found so far
        # Rank-based elite recombination (CMA-style): only the top quartile
        # moves the mean — scale-free and immune to the drift a flat softmax
        # shows when the optimum sits on a bound.
        elite = max(1, samples // 4)
        order = r.argsort(descending=True)
        w = torch.zeros(samples, dtype=DTYPE)
        w[order[:elite]] = (math.log(elite + 0.5)
                            - torch.log(torch.arange(1, elite + 1, dtype=DTYPE)))
        w = w / w.sum()
        mu = mu + (w.unsqueeze(0).unsqueeze(-1) * (cand - mu.unsqueeze(1))).sum(1)

    acts = Actions(sow=base.sow, irrigation=best_mu[0].clamp(*bounds[0]),
                   n_fert=best_mu[1].clamp(*bounds[1]))
    return ControlResult(acts, best_r, tuple(hist))
