"""Ensemble Kalman assimilation — the non-variational reference method.

The batch dimension IS the ensemble: `enkf_update` runs the stochastic
analysis on (B,) state vectors through the ensemble covariance, and
`assimilate` interleaves analysis with the shared rollout, so a
measurement schedule drives the whole ensemble without any Python loop
over members.

Analysis variables are pseudo-channels, not storage fields: `sw0` is the
surface layer and `swr` the summed root zone below it — an update on `swr`
is redistributed over layers proportionally to current content, which is
both layer-count-agnostic and storage-consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nasa_farm.observe import CHANNELS, observables
from nasa_farm.state.field import Field, FieldState
from nasa_farm.types import Actions, Forcing

STATE_VARS: tuple[str, ...] = ("sw0", "swr", "n_min", "lai", "biomass")
"""State components the analysis updates through cross-covariance."""


def _get_var(state: FieldState, name: str) -> torch.Tensor:
    if name == "sw0":
        return state.sw[:, 0]
    if name == "swr":
        return state.sw[:, 1:].sum(-1)
    return getattr(state, name)


def _put_vars(state: FieldState, names: tuple[str, ...],
              x: torch.Tensor) -> FieldState:
    kw: dict[str, torch.Tensor] = {}
    sw = state.sw.clone()
    for i, n in enumerate(names):
        if n == "sw0":
            sw[:, 0] = x[:, i]
        elif n == "swr" and state.n_layers > 1:
            root = sw[:, 1:]
            w = root / root.sum(-1, keepdim=True).clamp(min=1e-9)
            sw[:, 1:] = (root
                         + (x[:, i] - root.sum(-1)).unsqueeze(-1) * w).clamp(min=0.0)
        elif n == "swr":
            pass
        else:
            kw[n] = x[:, i]
    return state.replace(sw=sw, **kw)


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
    x = torch.stack([_get_var(state, n) for n in var_names], dim=-1)  # (B,V)
    xm, ym = x.mean(0), obs_pred.mean(0)
    xa, ya = x - xm, obs_pred - ym
    cxy = xa.t() @ ya / (b - 1)                     # (V,C)
    r_t = torch.as_tensor(r, dtype=x.dtype, device=x.device)
    cyy = ya.t() @ ya / (b - 1) \
        + torch.diag(r_t.expand(obs.numel()))
    k = torch.linalg.solve(cyy, cxy.t()).t()        # (V,C) — never invert
    if perturb:
        noise = torch.randn(b, obs.numel(), generator=gen,
                            dtype=x.dtype, device=x.device) * r_t.sqrt()
        innov = (obs + noise) - obs_pred
    else:
        innov = obs - obs_pred
    xn = (x + innov @ k.t()).clamp(min=0.0)
    return _put_vars(state, var_names, xn)


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
               update=None,
               perturb: bool = True, seed: int = 0) -> Field:
    """Online EnKF: rollout an ensemble field (B = members), pausing at each
    scheduled day to analyze predicted observables vs the measurement.
    `update` swaps the analysis step (default enkf_update; pass ensf_update
    for score filtering).

    Forcing and actions are shared across members (perfect-forcing form);
    spread comes from the initial ensemble and/or soil.sample(). The step
    itself runs under inference_mode — EnKF needs no gradients.
    """
    update = update or enkf_update
    gen = torch.Generator().manual_seed(seed)
    t0 = 0
    with torch.inference_mode():
        for o in sorted(schedule, key=lambda x: x.t):
            field = _slice_rollout(model, field, forcing, actions, t0, o.t)
            pred = observables(field.state, o.channels)
            kw = {"r": o.std ** 2, "var_names": var_names}
            try:
                st = update(field.state, pred, o.values, **kw,
                            perturb=perturb, gen=gen)
            except TypeError:
                st = update(field.state, pred, o.values, **kw)
            field = _replace_state(field, st)
            t0 = o.t
        field = _slice_rollout(model, field, forcing, actions, t0,
                               forcing.days)
    return field


def _replace_state(field: Field, state: FieldState) -> Field:
    return field.__class__(state=state, soil=field.soil, lat=field.lat,
                           elev=field.elev, ctx=field.ctx)


def _slice_rollout(model, field: Field, forcing: Forcing, actions: Actions,
                   t0: int, t1: int) -> Field:
    for t in range(t0, t1):
        field, _ = model.step(field, actions.day(t), forcing.day(t))
    return field
