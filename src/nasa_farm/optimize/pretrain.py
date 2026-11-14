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
    index = {n: i for i, n in enumerate(METRICS)}
    observed = []
    with torch.inference_mode():
        for f, w, a in zip(fields, forcings, actions):
            predicted = season_metrics(teacher, f, w, a)[0]
            values = torch.full_like(predicted, float("nan"))
            for c in channels:
                values[:, index[c]] = predicted[:, index[c]]
            observed.append(ObservedTrajectory(values=values))
    return observed


def pretrain(student, fields: list[Field], forcings: list[Forcing],
             actions: list[Actions], targets: list[ObservedTrajectory],
             parameters: dict[str, torch.Tensor], **fit_kw):
    """Fit `params` so the student reproduces the teacher's metrics over the
    whole site×year set — mean trajectory χ² across all rollouts."""
    def loss():
        total = torch.zeros((), dtype=fields[0].state.sw.dtype)
        for f, w, a, o in zip(fields, forcings, actions, targets):
            total = total + trajectory_loss(season_metrics(student, f, w, a), o)
        return total / len(targets)
    return fit(loss, parameters, **fit_kw)
