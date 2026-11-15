"""Shared hybrid pipeline — JEPA latent feeding a ctx-conditioned residual.

The jepa experiment (E6) evaluates the arms this module trains; the assim
experiment (E4) carries the same fitted modules inside its member
ensemble. One code path so every experiment trains the same object the
same way.
"""

from __future__ import annotations

import torch

from nasa_farm.observe.trace import EmbeddingTrace
from nasa_farm.optimize.trainer import FitResult, fit
from nasa_farm.state.field import Field


def trace(
    model,
    field: Field,
    forcing,
    actions,
    channels: tuple[str, ...],
    *,
    every: int,
    z_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Rollout with EmbeddingTrace → (obs (B,N,C), z (B,N,K)|None)."""
    traits = EmbeddingTrace(forcing.days, z_dim, channels, every=every)
    model.rollout(field, forcing, actions, observer=traits)
    return traits.result()


def obs_fit(
    model,
    field: Field,
    forcing,
    actions,
    targets: torch.Tensor,
    channels: tuple[str, ...],
    *,
    every: int,
    epochs: int,
    rate: float,
) -> FitResult:
    """Supervised obs-space fit of the residual — normalized MSE on the
    trace cadence, so JEPA and non-JEPA arms get identical budgets."""
    parameters = {f"p{i}": p for i, p in enumerate(model.residual.parameters())}
    scale = targets.std(dim=1, keepdim=True).clamp(min=1e-6)
    z_dim = model.latent.dim if model.latent else 0

    def loss() -> torch.Tensor:
        observed, _ = trace(model, field, forcing, actions, channels, every=every, z_dim=z_dim)
        return (((observed - targets) / scale) ** 2).mean()

    return fit(loss, parameters, epochs=epochs, rate=rate)
