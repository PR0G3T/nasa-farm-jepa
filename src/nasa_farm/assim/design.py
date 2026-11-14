"""Observation-schedule design — Bayesian experimental design for the EnKF.

For each candidate (day, channel), the expected information gain is the
log-ratio of prior predictive spread to instrument noise,
0.5·log det(C_yy + R) − 0.5·log det R, evaluated on one rollout of the
state ensemble. Greedy ranking then picks the measurements that shrink
posterior spread most per acquisition.
"""

from __future__ import annotations

import torch

from nasa_farm.observe import observables
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


def eig(obs_pred: torch.Tensor, r: torch.Tensor | float) -> torch.Tensor:
    """Expected information gain of one measurement channel.

    obs_pred: (B,) ensemble of predicted observables; r: error variance.
    gain = 0.5·log(1 + C_yy/R) for a scalar channel — monotone in ensemble
    spread, so days where members disagree rank highest.
    """
    r_t = torch.as_tensor(r, dtype=obs_pred.dtype, device=obs_pred.device)
    cyy = obs_pred.var(unbiased=True)
    return 0.5 * torch.log1p(cyy / r_t.clamp(min=1e-12))


def design_schedule(model, field: Field, forcing: Forcing, actions: Actions,
                    days: tuple[int, ...], channels: tuple[str, ...],
                    r: torch.Tensor | float = 1.0,
                    n_pick: int | None = None,
                    ) -> list[tuple[int, str, float]]:
    """Rank (day, channel) candidates by expected information gain.

    One ensemble rollout caches states at candidate days; gains are
    evaluated channel-wise — cheap because each channel's gain is scalar.
    Returns [(day, channel, gain)] sorted descending.
    """
    cand = set(days)
    snap: dict[int, object] = {}

    def cache(t: int, state, fluxes) -> None:
        if t in cand:
            snap[t] = state

    with torch.inference_mode():
        model.rollout(field, forcing, actions, observer=cache)

    r_t = torch.as_tensor(r)
    ranked = []
    for t in days:
        if t not in snap:
            continue
        y = observables(snap[t], channels)
        for i, c in enumerate(channels):
            g = float(eig(y[:, i], r_t if r_t.dim() == 0 else r_t[i]))
            ranked.append((t, c, g))
    ranked.sort(key=lambda x: -x[2])
    return ranked[:n_pick] if n_pick is not None else ranked
