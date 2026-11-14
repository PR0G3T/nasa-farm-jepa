"""ObsTrace — rollout observer that records predicted satellite observables
every `every` days into a preallocated (B, N, C) buffer.

Keeps the autograd graph (channels are differentiable functions of state),
so predicted-observation residuals flow gradients into dynamics.
"""

from __future__ import annotations

import torch

from nasa_farm.observe import CHANNELS, observables


class ObsTrace:
    def __init__(self, days: int, channels: tuple[str, ...] = ("ndvi",),
                 every: int = 5) -> None:
        for c in channels:
            if c not in CHANNELS:
                raise KeyError(f"unknown observable channel: {c}")
        self.channels, self.every = channels, every
        self.n = days // every + (1 if days % every else 0)
        self.rec: torch.Tensor | None = None

    def __call__(self, t: int, state, fluxes: dict) -> None:
        if t % self.every != 0:
            return
        if self.rec is None:
            self.rec = state.sw.new_zeros(state.batch, self.n,
                                          len(self.channels))
        self.rec[:, t // self.every] = observables(state, self.channels)

    def result(self) -> torch.Tensor:
        """(B, N, C) predicted observation series."""
        return self.rec


class JEPATrace:
    """Records (observations, latent z) pairs — everything a JEPA fit needs.

    `obs` feeds the EMA target encoder; `z` is the predictor's own
    trajectory as advanced inside `WorldModel.step`. One observer, one
    rollout, two preallocated buffers — no trajectory materialized."""

    def __init__(self, days: int, z_dim: int,
                 channels: tuple[str, ...] = ("ndvi",), every: int = 5) -> None:
        for c in channels:
            if c not in CHANNELS:
                raise KeyError(f"unknown observable channel: {c}")
        self.channels, self.every, self.z_dim = channels, every, z_dim
        self.n = days // every + (1 if days % every else 0)
        self.obs: torch.Tensor | None = None
        self.z: torch.Tensor | None = None

    def __call__(self, t: int, state, fluxes: dict) -> None:
        if t % self.every != 0:
            return
        if self.obs is None:
            self.obs = state.sw.new_zeros(state.batch, self.n,
                                          len(self.channels))
            self.z = state.sw.new_zeros(state.batch, self.n, self.z_dim)
        i = t // self.every
        self.obs[:, i] = observables(state, self.channels)
        if state.z is not None and self.z_dim:
            self.z[:, i] = state.z[:, :self.z_dim]

    def result(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(obs (B, N, C), z (B, N, K))."""
        return self.obs, self.z
