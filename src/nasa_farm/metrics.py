"""Scoring rules — deterministic and probabilistic evaluation.

Every function is NaN-aware (a pair with either side unobserved is dropped,
matching the `ObservedTrajectory` contract), returns a float64 scalar, and
accepts any broadcastable shapes — the same calls score a (H,) year series,
a (B, H, M) seasonal tensor sliced per channel, or a single value.

Deterministic suite (ag-model validation standard): rmse, nrmse, mae, mbe,
r2 (squared Pearson), nse (Nash–Sutcliffe modeling efficiency), kge (2009
and 2012 forms), d (Willmott agreement index), ccc (Lin concordance).

Probabilistic suite (ensemble-verification standard): fair ensemble CRPS
(Ferro et al. 2008) in the O(m log m) sorted form, randomized PIT, PICP,
Winkler interval score, pinball loss, spread–skill diagnostics.

`score` / `score_ens` bundle the full suite into one dict — the single
call every runner in `exp/` makes per evaluated channel.
"""

from __future__ import annotations

import math

import torch

from nasa_farm.constants import DTYPE

_NAN = torch.tensor(float("nan"), dtype=DTYPE)
_EPS = 1e-12


def _finite(pred: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast, flatten, and drop every pair with a non-finite side."""
    p, o = torch.broadcast_tensors(pred, obs)
    p, o = p.reshape(-1).to(DTYPE), o.reshape(-1).to(DTYPE)
    m = torch.isfinite(p) & torch.isfinite(o)
    return p[m], o[m]


def _finite_ens(ens: torch.Tensor,
                obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ens (..., M) member-last, obs (...) → flat (N, M) / (N,) rows; a row
    survives only if the obs and *all* its members are finite (a corrupt
    member corrupts the score)."""
    e = ens.reshape(-1, ens.shape[-1]).to(DTYPE)
    o = torch.broadcast_to(obs, ens.shape[:-1]).reshape(-1).to(DTYPE)
    m = torch.isfinite(o) & torch.isfinite(e).all(-1)
    return e[m], o[m]


# ---------------------------------------------------------------------------
# Deterministic scores
# ---------------------------------------------------------------------------

def mse(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    p, o = _finite(pred, obs)
    return ((p - o) ** 2).mean() if p.numel() else _NAN


def rmse(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return mse(pred, obs).sqrt()


def mae(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    p, o = _finite(pred, obs)
    return (p - o).abs().mean() if p.numel() else _NAN


def mbe(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Mean bias error — signed, positive = overestimation."""
    p, o = _finite(pred, obs)
    return (p - o).mean() if p.numel() else _NAN


def nrmse(pred: torch.Tensor, obs: torch.Tensor,
          norm: str = "auto") -> torch.Tensor:
    """RMSE normalized to be scale-free: /mean|obs| by default, falling back
    to the obs std when the mean is ~0 (anomaly channels like ΔSOC)."""
    p, o = _finite(pred, obs)
    if not p.numel():
        return _NAN
    if norm == "auto":
        norm = "mean" if o.abs().mean() > _EPS else "sd"
    den = {"mean": lambda: o.abs().mean(),
           "sd": lambda: o.std() if o.numel() > 1 else o.new_ones(()),
           "range": lambda: o.max() - o.min()}[norm]()
    return ((p - o) ** 2).mean().sqrt() / den.clamp(min=_EPS)


def _pearson(p: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    pc, oc = p - p.mean(), o - o.mean()
    den = pc.norm() * oc.norm()
    return (pc * oc).sum() / den.clamp(min=_EPS)


def r2(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Squared Pearson correlation — the crop-model R² convention."""
    p, o = _finite(pred, obs)
    if p.numel() < 2:
        return _NAN
    return _pearson(p, o) ** 2


def nse(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Nash–Sutcliffe / modeling efficiency: 1 perfect, 0 = obs-mean skill,
    <0 worse than climatology. EF > 0.70 marks an adequate crop model."""
    p, o = _finite(pred, obs)
    if p.numel() < 2:
        return _NAN
    sst = ((o - o.mean()) ** 2).sum()
    return 1.0 - ((p - o) ** 2).sum() / sst.clamp(min=_EPS)


def kge(pred: torch.Tensor, obs: torch.Tensor,
        version: str = "2012") -> torch.Tensor:
    """Kling–Gupta efficiency. '2009': variability term = σ ratio;
    '2012' (Kling et al.): variability term = CV ratio."""
    p, o = _finite(pred, obs)
    if p.numel() < 2:
        return _NAN
    r = _pearson(p, o)
    if version == "2009":
        v = p.std() / o.std().clamp(min=_EPS)
    else:
        cv = lambda x: x.std() / x.mean().abs().clamp(min=_EPS)
        v = cv(p) / cv(o).clamp(min=_EPS)
    beta = p.mean() / o.mean().clamp(min=_EPS)
    return 1.0 - ((r - 1) ** 2 + (v - 1) ** 2 + (beta - 1) ** 2).sqrt()


def d_index(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Willmott's index of agreement — 1 perfect, 0 no agreement."""
    p, o = _finite(pred, obs)
    if not p.numel():
        return _NAN
    num = ((p - o) ** 2).sum()
    den = ((p - o.mean()).abs() + (o - o.mean()).abs()) ** 2
    return 1.0 - num / den.sum().clamp(min=_EPS)


def ccc(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Lin's concordance correlation — precision × accuracy in one score."""
    p, o = _finite(pred, obs)
    if p.numel() < 2:
        return _NAN
    num = 2.0 * ((p - p.mean()) * (o - o.mean())).mean()
    den = p.var(correction=0) + o.var(correction=0) \
        + (p.mean() - o.mean()) ** 2
    return num / den.clamp(min=_EPS)


# ---------------------------------------------------------------------------
# Probabilistic scores
# ---------------------------------------------------------------------------

def crps(ens: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Fair ensemble CRPS (Ferro et al. 2008): mean |X−y| minus the
    member-diversity term — a proper score reducing to MAE for m = 1.

    Sorted O(m log m) form: Σ_{i≠j}|xᵢ−xⱼ| = 2·Σᵢ(2i−m+1)·x₍ᵢ₎, so no
    pairwise (m, m) distance matrix is ever materialized.
    """
    e, o = _finite_ens(ens, obs)
    if not e.numel():
        return _NAN
    x = e.sort(-1).values
    m = x.shape[-1]
    ae = (x - o.unsqueeze(-1)).abs().mean(-1)
    if m == 1:
        return ae.mean()
    i = torch.arange(m, dtype=DTYPE, device=x.device)
    spread = ((2 * i - m + 1) * x).sum(-1) / (m * (m - 1))
    return (ae - spread).mean()


def pit(ens: torch.Tensor, obs: torch.Tensor,
        gen: torch.Generator | None = None) -> torch.Tensor:
    """Randomized probability integral transform, one value per obs in
    [0, 1]. Uniform under a calibrated ensemble; U-shaped under
    overdispersion, domed under underdispersion, skewed under bias."""
    e, o = _finite_ens(ens, obs)
    m = e.shape[-1]
    lo = (e < o.unsqueeze(-1)).sum(-1)
    hi = (e <= o.unsqueeze(-1)).sum(-1)
    u = torch.rand(o.shape, generator=gen, dtype=DTYPE, device=o.device)
    return (lo + u * (hi - lo)) / m


def pit_hist(ens: torch.Tensor, obs: torch.Tensor, bins: int = 10,
             gen: torch.Generator | None = None) -> torch.Tensor:
    """Reliability histogram of PIT values — (bins,) counts."""
    u = pit(ens, obs, gen)
    return torch.histc(u, bins=bins, min=0.0, max=1.0)


def interval(ens: torch.Tensor,
             alpha: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Central (1−α) quantile band of the member ensemble → (lo, hi)."""
    e = ens.to(DTYPE)
    return (torch.quantile(e, alpha / 2.0, dim=-1),
            torch.quantile(e, 1.0 - alpha / 2.0, dim=-1))


def picp(lo: torch.Tensor, hi: torch.Tensor,
         obs: torch.Tensor) -> torch.Tensor:
    """Prediction-interval coverage probability — the calibration metric:
    should equal 1−α on a calibrated forecaster."""
    l, h, o = torch.broadcast_tensors(lo, hi, obs)
    m = torch.isfinite(o.reshape(-1))
    l, h, o = l.reshape(-1)[m], h.reshape(-1)[m], o.reshape(-1)[m]
    return ((o >= l) & (o <= h)).to(DTYPE).mean() if o.numel() else _NAN


def interval_score(lo: torch.Tensor, hi: torch.Tensor, obs: torch.Tensor,
                   alpha: float = 0.1) -> torch.Tensor:
    """Winkler interval score — rewards sharpness, penalizes misses by
    2/α per unit beyond the band. Lower is better."""
    l, h, o = torch.broadcast_tensors(lo, hi, obs)
    m = torch.isfinite(o.reshape(-1))
    l, h, o = l.reshape(-1)[m], h.reshape(-1)[m], o.reshape(-1)[m]
    if not o.numel():
        return _NAN
    s = (h - l) + (2.0 / alpha) * ((l - o).clamp(min=0.0)
                                   + (o - h).clamp(min=0.0))
    return s.mean()


def mean_width(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Sharpness — mean band width; only comparable at equal coverage."""
    l, h = torch.broadcast_tensors(lo, hi)
    m = torch.isfinite(l.reshape(-1)) & torch.isfinite(h.reshape(-1))
    return (h.reshape(-1)[m] - l.reshape(-1)[m]).mean() if m.any() else _NAN


def pinball(pred_q: torch.Tensor, obs: torch.Tensor,
            tau: float = 0.5) -> torch.Tensor:
    """Quantile (pinball) loss for a τ-quantile prediction."""
    p, o = _finite(pred_q, obs)
    if not p.numel():
        return _NAN
    d = o - p
    return torch.maximum(torch.tensor(tau, dtype=DTYPE) * d,
                         (tau - 1.0) * d).mean()


def energy_score(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fair energy score (Gneiting & Raftery 2007) — the proper multivariate
    scoring rule. pred (K, B, V) members, y (B, V) target; the k/(k−1)
    correction keeps it unbiased at small ensemble size."""
    k = pred.shape[0]
    t1 = (pred - y.unsqueeze(0)).norm(dim=-1).mean(0)
    d = (pred.unsqueeze(0) - pred.unsqueeze(1)).norm(dim=-1)
    t2 = 0.5 * d.mean(dim=(0, 1)) * (k / max(1, k - 1))
    return (t1 - t2).mean()


def spread(ens: torch.Tensor) -> torch.Tensor:
    """Mean ensemble spread (member std)."""
    e = ens.to(DTYPE)
    return e.std(-1).nanmean() if e.shape[-1] > 1 else e.new_zeros(())


def spread_skill(ens: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """RMSE/spread ratio — ≈1 for a calibrated ensemble, >1 underdispersed."""
    s = spread(ens)
    return rmse(ens.mean(-1), obs) / s.clamp(min=_EPS)


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------

def _num(x: torch.Tensor) -> float | None:
    """Metric → JSON-clean float; None when undefined (non-finite)."""
    v = float(x)
    return v if math.isfinite(v) else None


def score(pred: torch.Tensor, obs: torch.Tensor) -> dict[str, float | None]:
    """Full deterministic suite for one (pred, obs) pair."""
    p, o = _finite(pred, obs)
    return {
        "rmse": _num(rmse(p, o)), "nrmse": _num(nrmse(p, o)),
        "mae": _num(mae(p, o)), "mbe": _num(mbe(p, o)),
        "r2": _num(r2(p, o)), "nse": _num(nse(p, o)),
        "kge09": _num(kge(p, o, "2009")), "kge12": _num(kge(p, o)),
        "d": _num(d_index(p, o)), "ccc": _num(ccc(p, o)),
        "n": int(p.numel()),
    }


def score_ens(ens: torch.Tensor, obs: torch.Tensor,
              alphas: tuple[float, ...] = (0.1,), bins: int = 10,
              gen: torch.Generator | None = None) -> dict:
    """Full suite for an (N, M) ensemble: deterministic scores on the member
    mean plus CRPS, per-alpha PICP/interval-score/width, spread–skill and a
    PIT reliability histogram."""
    out: dict = score(ens.mean(-1), obs)
    out["crps"] = _num(crps(ens, obs))
    out["spread"] = _num(spread(ens))
    out["spread_skill"] = _num(spread_skill(ens, obs))
    for a in alphas:
        lo, hi = interval(ens, a)
        tag = f"{a:g}"
        out[f"picp_{tag}"] = _num(picp(lo, hi, obs))
        out[f"is_{tag}"] = _num(interval_score(lo, hi, obs, a))
        out[f"width_{tag}"] = _num(mean_width(lo, hi))
    out["pit"] = [int(c) for c in pit_hist(ens, obs, bins, gen)]
    return out
