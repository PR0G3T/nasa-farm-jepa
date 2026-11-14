"""Emulation pretraining — the NeuralCrop recipe.

A student model (this world model) first learns to imitate a richer teacher
(process-based GGCM, or a perturbed-trait clone of itself for smoke tests)
across a forcing ensemble, then fine-tunes on observations. Pretraining
across many sites/years is what makes the residual generalize outside the
calibration window.
"""

from __future__ import annotations

import torch

from nasa_farm.optimize.targets import (
    ObservedTrajectory, season_metrics, trajectory_loss,
)
from nasa_farm.optimize.trainer import fit
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


def emulate(teacher, fields: list[Field], forcings: list[Forcing],
            actions: list[Actions], channels: tuple[str, ...],
            ) -> list[ObservedTrajectory]:
    """Teacher rollouts → observation targets, one per (field, forcing) pair."""
    from nasa_farm.heads import METRICS
    idx = {n: i for i, n in enumerate(METRICS)}
    obs = []
    with torch.inference_mode():
        for f, w, a in zip(fields, forcings, actions):
            pred = season_metrics(teacher, f, w, a)[0]
            values = torch.full_like(pred, float("nan"))
            for c in channels:
                values[:, idx[c]] = pred[:, idx[c]]
            obs.append(ObservedTrajectory(values=values))
    return obs


def pretrain(student, fields: list[Field], forcings: list[Forcing],
             actions: list[Actions], targets: list[ObservedTrajectory],
             params: dict[str, torch.Tensor], **fit_kw):
    """Fit `params` so the student reproduces the teacher's metrics over the
    whole site×year set — mean trajectory χ² across all rollouts."""
    def loss():
        tot = torch.zeros((), dtype=fields[0].state.sw.dtype)
        for f, w, a, o in zip(fields, forcings, actions, targets):
            tot = tot + trajectory_loss(season_metrics(student, f, w, a), o)
        return tot / len(targets)
    return fit(loss, params, **fit_kw)
