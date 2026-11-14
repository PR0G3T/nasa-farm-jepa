"""Observation-error estimation and adaptive inflation for the EnKF.

`RStat` implements the Desroziers et al. (2005) diagnostic: under linear
least-variance estimation, E[(y − Hx_b)(y − Hx_a)ᵀ] = R — background ×
analysis innovations in observation space recover the observation-error
covariance nearly free, including cross-channel correlations a diagonal
guess would miss. Running-mean accumulation makes it online.

`aoei` is the Minamide–Zhang adaptive observation-error inflation: a
flow-dependent check that raises R just enough to keep the prior +
likelihood consistent when the day's innovation exceeds what the ensemble
spread + R can explain. R' = R while d² ≤ var_ens + R, else d² − var_ens.
"""

from __future__ import annotations

import torch


class Statistic:
    """Online Desroziers R estimator; `estimate` → (C, C) covariance."""

    def __init__(self, decay: float = 0.0) -> None:
        """decay=0 → plain running mean; >0 → exponential forgetting."""
        self.decay = decay
        self.n = 0.0
        self.accumulate: torch.Tensor | None = None

    def update(self, d_bg: torch.Tensor, d_an: torch.Tensor) -> None:
        """d_bg = y − Hx̄_b, d_an = y − Hx̄_a, both (C,) observation space."""
        p = torch.outer(d_bg, d_an)
        if self.accumulate is None:
            self.accumulate, self.n = torch.zeros_like(p), 0.0
        self.accumulate.mul_(1.0 - self.decay).add_(p)
        self.n = self.n * (1.0 - self.decay) + 1.0

    @property
    def estimate(self) -> torch.Tensor | None:
        return None if self.accumulate is None else self.accumulate / max(self.n, 1.0)

    @property
    def var(self) -> torch.Tensor | None:
        """Diagonal of the estimate — drop-in replacement for std²."""
        e = self.estimate
        return None if e is None else e.diagonal().clamp(min=1e-12)


def inflate(innov: torch.Tensor, var_ens: torch.Tensor,
         r: torch.Tensor | float) -> torch.Tensor:
    """Adaptive observation-error inflation, per channel.

    innov (C,) = y − Hx̄_b; var_ens (C,) = ensemble variance of predicted
    obs; r (C,) or scalar baseline variance → inflated variance ≥ r.
    """
    r_t = torch.as_tensor(r, dtype=innov.dtype, device=innov.device)
    return torch.maximum(r_t.expand_as(innov), innov ** 2 - var_ens)
