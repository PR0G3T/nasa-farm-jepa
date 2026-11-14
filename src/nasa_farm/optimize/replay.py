"""Persistent test-time training of the latent predictor.

Deployment streams reveal (feats, z, zh → z_next) transitions continuously.
A dense replay buffer keeps them and takes small SGD steps on the predictor
alone, so the latent dynamics track slow regime drift without ever touching
the physics or the encoder. `optimize.adapt` refits the environment (z0/ctx);
this module refits the dynamics model itself — complementary layers.
"""

from __future__ import annotations

from collections import deque

import torch

from nasa_farm.dynamics.latent import regularization


class Replay:
    """Ring-buffer replay + one-step latent-prediction updates."""

    def __init__(self, model, cap: int = 512, batch: int = 32,
                 rate: float = 3e-3, sig_weight: float = 0.1,
                 seed: int = 0) -> None:
        if model.latent is None:
            raise ValueError("TTT needs a model built with latent_dim > 0")
        self.latent = model.latent
        self.buffer: deque[tuple] = deque(maxlen=cap)
        self.batch = batch
        self.sig_weight = sig_weight
        self.opt = torch.optim.SGD(model.latent.parameters(), lr=rate)
        self.generation = torch.Generator().manual_seed(seed)
        self.history: list[float] = []

    def push(self, features: torch.Tensor, z: torch.Tensor,
             zh: torch.Tensor, target: torch.Tensor) -> None:
        """Store one transition (detached rows). Shapes (B, ·)."""
        for i in range(features.shape[0]):
            self.buffer.append((features[i].detach(), z[i].detach(),
                             zh[i].detach(), target[i].detach()))

    def step(self, n: int = 1) -> float:
        """n minibatch updates on the predictor → mean loss."""
        if len(self.buffer) < 2:
            return float("nan")
        losses = []
        for _ in range(n):
            index = torch.randperm(len(self.buffer), generator=self.generation)[:self.batch]
            rows = [self.buffer[i] for i in index.tolist()]
            f = torch.stack([r[0] for r in rows])
            z = torch.stack([r[1] for r in rows])
            zh = torch.stack([r[2] for r in rows])
            target = torch.stack([r[3] for r in rows])
            self.opt.zero_grad(set_to_none=True)
            predicted, _ = self.latent(f, z, zh)
            loss = (predicted - target).pow(2).mean() \
                + self.sig_weight * regularization(predicted)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.latent.parameters(), 1.0)
            self.opt.step()
            losses.append(float(loss.detach()))
        v = sum(losses) / len(losses)
        self.history.append(v)
        return v
