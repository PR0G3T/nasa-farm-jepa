"""Distribution-free uncertainty quantification: split conformal prediction.

Calibration scores s_i = |y_i − ŷ_i| on held-out data give a margin q̂ such
that P(y ∈ ŷ ± q̂) ≥ 1−α under exchangeability — exact finite-sample validity
with zero distribution assumptions. `weighted` supports non-exchangeable
(covariate-shifted) calibration via importance weights.
"""

from __future__ import annotations

import math

import torch


def weighted_quantile(x: torch.Tensor, q: float,
                      w: torch.Tensor | None = None) -> torch.Tensor:
    """q-quantile of x; w importance-weights each sample."""
    x, idx = torch.sort(x.flatten())
    w = torch.ones_like(x) if w is None else w.flatten()[idx]
    cw = w.cumsum(0) / w.sum()
    i = torch.searchsorted(cw, torch.as_tensor(q, dtype=x.dtype))
    return x[i.clamp(max=x.numel() - 1)]


def conf_level(scores: torch.Tensor, alpha: float = 0.1,
               weights: torch.Tensor | None = None) -> torch.Tensor:
    """Conformal margin: ceil((n+1)(1−α))/n empirical quantile of |residuals|."""
    n = scores.numel()
    q = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    return weighted_quantile(scores, q, weights)


def interval(pred: torch.Tensor, scores: torch.Tensor, alpha: float = 0.1,
             weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric conformal band around point predictions."""
    q = conf_level(scores, alpha, weights)
    return pred - q, pred + q


def coverage(y: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> float:
    """Empirical fraction of y inside [lo, hi]."""
    inside = (y >= lo) & (y <= hi)
    return float(inside.to(torch.float64).mean())


class SplitConformal:
    """Per-channel split-conformal calibrator.

    fit(pred_cal, y_cal) stores scores per channel;
    bound(pred) → (lo, hi) at the configured alpha.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.scores: torch.Tensor | None = None

    def fit(self, pred: torch.Tensor, y: torch.Tensor) -> "SplitConformal":
        self.scores = (pred - y).abs()
        return self

    def bound(self, pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = torch.stack([conf_level(self.scores[..., c], self.alpha)
                         for c in range(self.scores.shape[-1])])
        return pred - q, pred + q
