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
class Mapping:
    """Per-(variable, window) model and reference quantile levels."""

    q_model: torch.Tensor  # (V, W, Q)
    q_ref: torch.Tensor  # (V, W, Q)
    windows: int
    probs: torch.Tensor  # (Q,)


def fit_mapping(model: Forcing, reference: Forcing, windows: int = 12, n_q: int = 17) -> Mapping:
    """Learn the transfer map: for each variable and day-of-year window,
    sort the pooled model and reference values into `n_q` quantiles."""
    probs = torch.linspace(0, 1, n_q, dtype=DTYPE)
    span = DAYS_PER_YEAR / windows
    qm, qr = [], []
    for k in QMAP_VARS:
        mv, rv = getattr(model, k)[0], getattr(reference, k)[0]
        dm = (model.doy[0] - 1).div(span).floor().long().clamp(0, windows - 1)
        row_m, row_r = [], []
        for w in range(windows):
            sel = dm == w
            row_m.append(torch.quantile(mv[sel].to(DTYPE), probs))
            row_r.append(torch.quantile(rv[sel].to(DTYPE), probs))
        qm.append(torch.stack(row_m))
        qr.append(torch.stack(row_r))
    return Mapping(torch.stack(qm), torch.stack(qr), windows, probs)


def _interp(xp: torch.Tensor, fp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """1-D linear interpolation per row: fp(xp) evaluated at x."""
    i = torch.searchsorted(xp, x.contiguous()).clamp(1, xp.shape[-1] - 1)
    x0, x1 = xp.gather(-1, i - 1), xp.gather(-1, i)
    f0, f1 = fp.gather(-1, i - 1), fp.gather(-1, i)
    w = ((x - x0) / (x1 - x0).clamp(min=1e-9)).clamp(0.0, 1.0)
    return f0 + w * (f1 - f0)


def apply_mapping(f: Forcing, qm: Mapping) -> Forcing:
    """Bias-correct a forcing: each value is mapped through the quantile
    curve of its (variable, doy window)."""
    output = {}
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
            rows.append((sel, _interp(xp.view(1, -1), fp.view(1, -1), x.view(1, -1))[0]))
        column = getattr(f, k)[0].clone()
        for sel, values in rows:
            column[sel] = values.to(column.dtype)
        output[k] = column.unsqueeze(0)
    return Forcing(
        tmin=output["tmin"],
        tmax=torch.maximum(output["tmax"], output["tmin"]),
        srad=output["srad"].clamp(min=0.5),
        precip=output["precip"].clamp(min=0.0),
        rh=output["rh"].clamp(0.0, 100.0),
        u2=output["u2"].clamp(min=0.0),
        doy=f.doy,
        co2=f.co2,
    )


@dataclass(frozen=True, slots=True)
class Correction:
    """Probabilistic ensemble bias correction: per-(variable, window)
    mean offset and spread scale for a forecast ensemble — the ensemble
    analogue of `QMap` (calibrates the distribution, not each member
    independently)."""

    bias: torch.Tensor  # (V, W) — subtracted from every member
    gamma: torch.Tensor  # (V, W) — anomaly (spread) rescale
    windows: int


def fit_correction(fcst: Forcing, reference: Forcing, windows: int = 12) -> Correction:
    """fcst: (S, T) forecast ensemble; ref: (R, T) reference years on the
    same doy axis. Learns per-(variable, window) mean offset of the
    ensemble mean vs reference, and the spread ratio reference/ensemble
    (spread-skill recalibration)."""
    span = DAYS_PER_YEAR / windows
    w = (fcst.doy[0] - 1).div(span).floor().long().clamp(0, windows - 1)
    bias, gamma = [], []
    for k in QMAP_VARS:
        ensemble = getattr(fcst, k).to(DTYPE)  # (S, T)
        rv = getattr(reference, k).to(DTYPE)  # (R, T)
        eb = ensemble.mean(0)  # ensemble mean (T,)
        b_row, g_row = [], []
        for wi in range(windows):
            sel = w == wi
            if not sel.any():
                b_row.append(torch.zeros((), dtype=DTYPE))
                g_row.append(torch.ones((), dtype=DTYPE))
                continue
            ref_sel = rv[:, sel].flatten()
            b_row.append(eb[sel].mean() - ref_sel.mean())
            anom_ref = ref_sel - ref_sel.mean()
            anom_ens = (ensemble[:, sel] - eb[sel]).flatten()
            g_row.append(anom_ref.std() / anom_ens.std().clamp(min=1e-9))
        bias.append(torch.stack(b_row))
        gamma.append(torch.stack(g_row).clamp(0.25, 4.0))
    return Correction(torch.stack(bias), torch.stack(gamma), windows)


def apply_correction(f: Forcing, bc: Correction) -> Forcing:
    """Apply to each member: x' = x̄ − δ + γ·(x − x̄) — recenter and
    recalibrate spread, preserving member rank order (no re-sampling)."""
    span = DAYS_PER_YEAR / bc.windows
    w = (f.doy[0] - 1).div(span).floor().long().clamp(0, bc.windows - 1)
    output = {}
    for vi, k in enumerate(QMAP_VARS):
        x = getattr(f, k).to(DTYPE)  # (S, T)
        xm = x.mean(0, keepdim=True)
        d = bc.bias[vi][w].unsqueeze(0)
        g = bc.gamma[vi][w].unsqueeze(0)
        output[k] = xm - d + g * (x - xm)
    return Forcing(
        tmin=output["tmin"],
        tmax=torch.maximum(output["tmax"], output["tmin"]),
        srad=output["srad"].clamp(min=0.5),
        precip=output["precip"].clamp(min=0.0),
        rh=output["rh"].clamp(0.0, 100.0),
        u2=output["u2"].clamp(min=0.0),
        doy=f.doy,
        co2=f.co2,
    )
