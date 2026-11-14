"""Observation targets and the differentiable trajectory loss.

`ObservedTrajectory` is the exact contract real observations will satisfy:
per-year values on the same channels `heads.METRICS` produces, with NaN for
unobserved cells and a per-channel weight. The loss is a scaled χ² — fully
differentiable, so gradients flow back through the rollout into whichever
parameters produced the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.heads import METRICS, SeasonAccumulator
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


def season_metrics(model, field: Field, forcing: Forcing,
                   actions: Actions) -> torch.Tensor:
    """Rollout → (B, H, M) seasonal metrics, *keeping the graph*.

    Unlike `plan._rollout_metrics` (inference_mode for planning), this is the
    data-assimilation path: gradients must reach the model parameters.
    """
    acc = SeasonAccumulator(field.batch, forcing.days // 365,
                            device=field.state.sw1.device)
    model.rollout(field, forcing, actions, observer=acc)
    return acc.result()


@dataclass(frozen=True, slots=True)
class ObservedTrajectory:
    """(H, M) observed seasonal metrics; NaN = unobserved."""

    values: torch.Tensor
    channel_scale: torch.Tensor | None = None   # (M,) χ² normalizer

    def observed_mask(self) -> torch.Tensor:
        return torch.isfinite(self.values)


def trajectory_loss(pred: torch.Tensor, obs: ObservedTrajectory,
                    weights: torch.Tensor | None = None) -> torch.Tensor:
    """χ² over observed cells. pred (B, H, M) or (H, M); returns scalar."""
    p = pred.squeeze(0) if pred.dim() == 3 else pred
    mask = obs.observed_mask()
    scale = obs.channel_scale
    if scale is None:
        scale = obs.values[mask].abs().mean().clamp(min=1.0)
    r = (p - torch.nan_to_num(obs.values)) / scale
    r = r * mask
    if weights is not None:
        r = r * weights.unsqueeze(0) if weights.dim() == 1 else r * weights
    return (r ** 2).sum() / mask.sum().clamp(min=1)


def synthetic_observation(model, field: Field, forcing: Forcing,
                          actions: Actions,
                          channels: tuple[str, ...] = ("yield_kg",),
                          noise: float = 0.0, seed: int = 0) -> ObservedTrajectory:
    """Reference 'dataset' built from a rollout — the same container real
    SMAP/HLS/yield series will fill. `channels` are METRICS names; all other
    cells stay NaN. Optional Gaussian noise emulates observational error.
    """
    with torch.inference_mode():
        pred = season_metrics(model, field, forcing, actions)[0]  # (H, M)
    idx = {n: i for i, n in enumerate(METRICS)}
    values = torch.full_like(pred, float("nan"))
    for c in channels:
        values[:, idx[c]] = pred[:, idx[c]]
    if noise > 0:
        gen = torch.Generator().manual_seed(seed)
        obs_cells = torch.isfinite(values)
        values = values + torch.randn(values.shape, generator=gen,
                                      dtype=values.dtype) \
            * noise * values.abs().clamp(min=1.0) * obs_cells
    return ObservedTrajectory(values=values)


@dataclass(frozen=True, slots=True)
class ObservedSeries:
    """(N, C) satellite observation timeseries; NaN = unobserved.

    Same contract as `ObservedTrajectory` but at sub-seasonal cadence on
    `observe.CHANNELS` names (ndvi, s0v, tb_v, sm1, ...)."""

    values: torch.Tensor
    channel_scale: torch.Tensor | None = None   # (C,) χ² normalizer

    def observed_mask(self) -> torch.Tensor:
        return torch.isfinite(self.values)


def obs_series(model, field: Field, forcing: Forcing, actions: Actions,
               channels: tuple[str, ...] = ("ndvi", "sm1"),
               every: int = 5) -> torch.Tensor:
    """Rollout → (B, N, C) predicted observables, *keeping the graph*."""
    from nasa_farm.observe.trace import ObsTrace
    tr = ObsTrace(forcing.days, channels, every)
    model.rollout(field, forcing, actions, observer=tr)
    return tr.result()


def series_loss(pred: torch.Tensor, obs: ObservedSeries,
                weights: torch.Tensor | None = None) -> torch.Tensor:
    """χ² over observed cells. pred (B, N, C) or (N, C); returns scalar."""
    p = pred.squeeze(0) if pred.dim() == 3 else pred
    mask = obs.observed_mask()
    scale = obs.channel_scale
    if scale is None:
        scale = obs.values[mask].abs().mean().clamp(min=1.0)
    r = (p - torch.nan_to_num(obs.values)) / scale * mask
    if weights is not None:
        r = r * weights.unsqueeze(0) if weights.dim() == 1 else r * weights
    return (r ** 2).sum() / mask.sum().clamp(min=1)
