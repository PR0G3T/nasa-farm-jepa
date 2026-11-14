"""Ensemble Kalman assimilation — the non-variational reference method.

The batch dimension IS the ensemble: `enkf_update` runs the stochastic
analysis on (B,) state vectors through the ensemble covariance, and
`assimilate` interleaves analysis with the shared rollout, so a
measurement schedule drives the whole ensemble without any Python loop
over members.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nasa_farm.observe import CHANNELS, observables
from nasa_farm.state.field import Field, FieldState
from nasa_farm.types import Actions, Forcing

STATE_VARS: tuple[str, ...] = ("sw1", "sw2", "n_min", "lai", "biomass")
"""State components the analysis updates through cross-covariance."""


def enkf_update(state: FieldState, obs_pred: torch.Tensor,
                obs: torch.Tensor, r: torch.Tensor | float,
                var_names: tuple[str, ...] = STATE_VARS,
                perturb: bool = True,
                gen: torch.Generator | None = None) -> FieldState:
    """Stochastic EnKF analysis.

    state: (B,) ensemble; obs_pred: (B, C) predicted observations;
    obs: (C,) measurement; r: observation-error variance (scalar or (C,)).
    Members' observations are perturbed ~N(0, R) — the stochastic form
    whose posterior covariance is exact under Gaussianity.
    """
    b = state.batch
    x = torch.stack([getattr(state, n) for n in var_names], dim=-1)  # (B,V)
    xm, ym = x.mean(0), obs_pred.mean(0)
    xa, ya = x - xm, obs_pred - ym
    cxy = xa.t() @ ya / (b - 1)                     # (V,C)
    r_t = torch.as_tensor(r, dtype=x.dtype, device=x.device)
    cyy = ya.t() @ ya / (b - 1) \
        + torch.diag(r_t.expand(obs.numel()))
    k = cxy @ torch.linalg.inv(cyy)                 # (V,C)
    if perturb:
        noise = torch.randn(b, obs.numel(), generator=gen,
                            dtype=x.dtype, device=x.device) * r_t.sqrt()
        innov = (obs + noise) - obs_pred
    else:
        innov = obs - obs_pred
    xn = (x + innov @ k.t()).clamp(min=0.0)
    return state.replace(**{n: xn[:, i] for i, n in enumerate(var_names)})


@dataclass(frozen=True, slots=True)
class Observation:
    """One analysis point: day index, channel names, values, error std."""

    t: int
    channels: tuple[str, ...]
    values: torch.Tensor        # (C,)
    std: torch.Tensor | float   # (C,) or scalar

    def __post_init__(self) -> None:
        for c in self.channels:
            if c not in CHANNELS:
                raise KeyError(f"unknown observable channel: {c}")


def assimilate(model, field: Field, forcing: Forcing, actions: Actions,
               schedule: list[Observation],
               var_names: tuple[str, ...] = STATE_VARS,
               perturb: bool = True, seed: int = 0) -> Field:
    """Online EnKF: rollout an ensemble field (B = members), pausing at each
    scheduled day to analyze predicted observables vs the measurement.

    Forcing and actions are shared across members (perfect-forcing form);
    spread comes from the initial ensemble and/or soil.sample(). The step
    itself runs under inference_mode — EnKF needs no gradients.
    """
    gen = torch.Generator().manual_seed(seed)
    t0 = 0
    with torch.inference_mode():
        for o in sorted(schedule, key=lambda x: x.t):
            field = _slice_rollout(model, field, forcing, actions, t0, o.t)
            pred = observables(field.state, o.channels)
            field = _replace_state(
                field, enkf_update(field.state, pred, o.values,
                                   o.std ** 2, var_names, perturb, gen))
            t0 = o.t
        field = _slice_rollout(model, field, forcing, actions, t0,
                               forcing.days)
    return field


def _replace_state(field: Field, state: FieldState) -> Field:
    return field.__class__(state=state, soil=field.soil, lat=field.lat,
                           elev=field.elev)


def _slice_rollout(model, field: Field, forcing: Forcing, actions: Actions,
                   t0: int, t1: int) -> Field:
    for t in range(t0, t1):
        field, _ = model.step(field, actions.day(t), forcing.day(t))
    return field
