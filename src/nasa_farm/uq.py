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


class TrajectoryConformal:
    """Joint conformal band over a whole trajectory.

    The score is the sup-norm of scaled per-day residuals,
    s_i = max_t |y_it − ŷ_it| / scale_t, so the returned band covers the
    entire predicted trajectory simultaneously at level 1−α (not just
    pointwise)."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.scale: torch.Tensor | None = None
        self.scores: torch.Tensor | None = None

    def fit(self, pred: torch.Tensor, y: torch.Tensor,
            scale: torch.Tensor | None = None) -> "TrajectoryConformal":
        """pred/y: (N, T[, C]) calibration trajectories."""
        self.scale = (torch.ones_like(pred[0]) if scale is None
                      else scale.clamp(min=1e-9))
        self.scores = ((pred - y).abs() / self.scale).amax(
            dim=tuple(range(1, pred.dim())))
        return self

    def bound(self, pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = conf_level(self.scores, self.alpha)
        m = q * self.scale
        return pred - m, pred + m


def aci(alpha_t: float, err: bool, target: float = 0.1,
        gamma: float = 0.02) -> float:
    """Adaptive conformal inference step: after a miss the miscoverage
    level drops (intervals widen), after a cover it rises (intervals
    narrow) — online calibration under distribution shift."""
    return alpha_t + gamma * (target - float(err))


class LocalConformal:
    """Localized split conformal: kernel weights around each test point.

    Standard conformal assumes all calibration residuals are exchangeable
    with the test residual; heteroscedastic or shifted settings violate
    that locally. Here the calibration set is reweighted by a Gaussian
    kernel on feature distance (Guan's localizer), so each test point
    calibrates against its similar neighbors — marginal coverage is kept
    by the weighted-quantile machinery and conditional coverage improves
    where similarity is informative.
    """

    def __init__(self, alpha: float = 0.1, bw: float | None = None,
                 floor: float = 0.05) -> None:
        self.alpha, self.bw, self.floor = alpha, bw, floor
        self.scores: torch.Tensor | None = None
        self.x: torch.Tensor | None = None

    def fit(self, pred: torch.Tensor, y: torch.Tensor,
            x: torch.Tensor) -> "LocalConformal":
        """x: (N, F) calibration features (any distance-meaningful space)."""
        self.scores = (pred - y).abs().flatten()
        self.x = x.reshape(x.shape[0], -1)
        if self.bw is None:                     # median-heuristic bandwidth
            d = torch.cdist(self.x, self.x)
            pos = d[d > 0]
            self.bw = float(pos.median().clamp(min=1e-6)) \
                if pos.numel() else 1.0
        return self

    def bound(self, pred: torch.Tensor, x: torch.Tensor,
              ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-point band: q_j = weighted conformal level near x_j."""
        xt = x.reshape(x.shape[0], -1)
        d = torch.cdist(xt, self.x)                          # (M, N)
        w = torch.exp(-0.5 * (d / self.bw) ** 2) + self.floor
        n = self.scores.numel()
        q = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        qs = torch.stack([weighted_quantile(self.scores, q, w[j])
                          for j in range(xt.shape[0])]).reshape(pred.shape)
        return pred - qs, pred + qs


class PanelConformal:
    """Online weighted conformal for panels — W-TQA-style calibration.

    At each step the target unit borrows the other units' most recent
    conformity scores, weighted by residual similarity (softmax of
    η·corr over the rolling window) plus a uniform floor — coverage
    holds marginally and tracks regime shifts the pooled split cannot.
    The miscoverage level adapts online via ACI after every outcome.
    """

    def __init__(self, alpha: float = 0.1, eta: float = 3.0,
                 gamma: float = 0.02, floor: float = 0.1,
                 target: float = 0.1) -> None:
        self.alpha_t, self.eta, self.gamma = alpha, eta, gamma
        self.floor, self.target = floor, target

    def bound(self, pred: float | torch.Tensor,
              peers: torch.Tensor, sim: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor]:
        """peers (K,) other units' latest scores; sim (K,) similarity ∈ℝ."""
        w = torch.softmax(self.eta * sim, 0) + self.floor / peers.numel()
        q = conf_level(peers, self.alpha_t, w)
        return pred - q, pred + q

    def update(self, err: bool) -> float:
        """Feed the realized miss → adapted alpha for the next round."""
        self.alpha_t = aci(self.alpha_t, err, self.target, self.gamma)
        return self.alpha_t


def enbpi(pred: torch.Tensor, residuals: torch.Tensor,
          alpha: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensemble-batch prediction intervals: per-day band from the quantiles
    of a rolling residual window (bootstrap-after-fit UQ that needs no
    retraining). residuals: (N, T)."""
    lo = torch.quantile(residuals, alpha / 2.0, dim=0)
    hi = torch.quantile(residuals, 1.0 - alpha / 2.0, dim=0)
    return pred + lo, pred + hi


def spw_perturb(model, scale: float = 0.05,
                gen: torch.Generator | None = None) -> dict:
    """Stochastically perturbed weights: zero-cost ensemble member.

    Gaussian noise on every learned tensor, scaled per-tensor by its std —
    a weight-space analogue of forecast perturbations. Returns the backup
    dict; `spw_restore` puts the originals back.
    """
    backup = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            backup[n] = p.detach().clone()
            p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype,
                               device=p.device) * scale
                   * p.std().clamp(min=1e-6))
    return backup


def spw_restore(model, backup: dict) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(backup[n])


def spw_rollout(model, field, forcing, actions, members: int = 8,
                scale: float = 0.05, seed: int = 0) -> torch.Tensor:
    """(S, B, H, M) seasonal metrics under weight perturbations — ensemble
    spread at the cost of S rollouts and zero extra training."""
    from nasa_farm.heads import SeasonAccumulator
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(members):
        backup = spw_perturb(model, scale, gen)
        with torch.inference_mode():
            acc = SeasonAccumulator(field.batch, max(1, forcing.days // 365),
                                    device=field.state.sw.device)
            model.rollout(field, forcing, actions, observer=acc)
            out.append(acc.result())
        spw_restore(model, backup)
    return torch.stack(out)
