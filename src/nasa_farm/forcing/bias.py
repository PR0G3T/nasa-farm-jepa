"""Quantile-mapping bias correction — map model/forecast quantiles onto the
reference (reanalysis/observed) distribution, per variable and seasonal
window. The standard post-processor for S2S products feeding the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.types import Forcing

QMAP_VARS: tuple[str, ...] = ("tmin", "tmax", "srad", "precip", "rh", "u2")


@dataclass(frozen=True, slots=True)
class QMap:
    """Per-(variable, window) model and reference quantile levels."""

    q_model: torch.Tensor   # (V, W, Q)
    q_ref: torch.Tensor     # (V, W, Q)
    windows: int
    probs: torch.Tensor     # (Q,)


def fit_qmap(model: Forcing, ref: Forcing, windows: int = 12,
             n_q: int = 17) -> QMap:
    """Learn the transfer map: for each variable and day-of-year window,
    sort the pooled model and reference values into `n_q` quantiles."""
    probs = torch.linspace(0, 1, n_q, dtype=DTYPE)
    span = DAYS_PER_YEAR / windows
    qm, qr = [], []
    for k in QMAP_VARS:
        mv, rv = getattr(model, k)[0], getattr(ref, k)[0]
        dm = (model.doy[0] - 1).div(span).floor().long().clamp(0, windows - 1)
        row_m, row_r = [], []
        for w in range(windows):
            sel = dm == w
            row_m.append(torch.quantile(mv[sel].to(DTYPE), probs))
            row_r.append(torch.quantile(rv[sel].to(DTYPE), probs))
        qm.append(torch.stack(row_m))
        qr.append(torch.stack(row_r))
    return QMap(torch.stack(qm), torch.stack(qr), windows, probs)


def _interp(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """1-D linear interpolation per row: fp(xp) evaluated at x."""
    i = torch.searchsorted(xp, x.contiguous()).clamp(1, xp.shape[-1] - 1)
    x0, x1 = xp.gather(-1, i - 1), xp.gather(-1, i)
    f0, f1 = fp.gather(-1, i - 1), fp.gather(-1, i)
    w = ((x - x0) / (x1 - x0).clamp(min=1e-9)).clamp(0.0, 1.0)
    return f0 + w * (f1 - f0)


def apply_qmap(f: Forcing, qm: QMap) -> Forcing:
    """Bias-correct a forcing: each value is mapped through the quantile
    curve of its (variable, doy window)."""
    out = {}
    span = DAYS_PER_YEAR / qm.windows
    w = (f.doy[0] - 1).div(span).floor().long().clamp(0, qm.windows - 1)
    for vi, k in enumerate(QMAP_VARS):
        rows = []
        for wi in range(qm.windows):
            sel = w == wi
            if not sel.any():
                continue
            x = getattr(f, k)[0, sel]
            xp, fp = qm.q_model[vi, wi], qm.q_ref[vi, wi]
            rows.append((sel, _interp(xp.view(1, -1), fp.view(1, -1),
                                      x.view(1, -1))[0]))
        col = getattr(f, k)[0].clone()
        for sel, vals in rows:
            col[sel] = vals.to(col.dtype)
        out[k] = col.unsqueeze(0)
    return Forcing(tmin=out["tmin"], tmax=torch.maximum(out["tmax"], out["tmin"]),
                   srad=out["srad"].clamp(min=0.5),
                   precip=out["precip"].clamp(min=0.0),
                   rh=out["rh"].clamp(0.0, 100.0),
                   u2=out["u2"].clamp(min=0.0), doy=f.doy, co2=f.co2)
