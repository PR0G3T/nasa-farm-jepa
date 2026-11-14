"""SWAG — diagonal Gaussian posterior over network weights from iterates.

Collect the last-K optimization iterates, fit mean + diagonal covariance,
then sample weights for epistemic-uncertainty predictions. No second-order
information, no extra training runs — the cheapest defensible epistemic
layer for the residual/latent nets.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Swag:
    """Diagonal-moment accumulator over a flat parameter vector."""

    def __init__(self, net: nn.Module) -> None:
        self.net = net
        self.n = 0
        self.mean: torch.Tensor | None = None
        self.sq: torch.Tensor | None = None

    @staticmethod
    def _flat(net: nn.Module) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in net.parameters()])

    def collect(self) -> None:
        """Append current parameters as one posterior sample."""
        v = self._flat(self.net)
        self.mean = v if self.mean is None else (self.mean * self.n + v) / (self.n + 1)
        self.sq = (v ** 2) if self.sq is None else (self.sq * self.n + v ** 2) / (self.n + 1)
        self.n += 1

    @property
    def var(self) -> torch.Tensor:
        return (self.sq - self.mean ** 2).clamp(min=0.0)

    def sample(self, gen: torch.Generator | None = None,
               scale: float = 1.0) -> torch.Tensor:
        """Draw a weight vector ~ N(mean, diag(var)·scale²)."""
        return self.mean + scale * self.var.sqrt() * torch.randn(
            self.mean.shape, generator=gen,
            dtype=self.mean.dtype, device=self.mean.device)

    def load(self, flat: torch.Tensor) -> None:
        """Write a flat vector back into the network's parameters."""
        off = 0
        with torch.no_grad():
            for p in self.net.parameters():
                n = p.numel()
                p.copy_(flat[off:off + n].view_as(p))
                off += n

    def predict(self, x: torch.Tensor, n: int = 20,
                gen: torch.Generator | None = None) -> torch.Tensor:
        """(n, B, O) prediction ensemble under sampled weights."""
        cur = self._flat(self.net)
        outs = []
        for _ in range(n):
            self.load(self.sample(gen))
            outs.append(self.net(x))
        self.load(cur)
        return torch.stack(outs)
