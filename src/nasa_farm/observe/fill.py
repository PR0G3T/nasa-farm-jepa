"""Weighted Whittaker gap-filling — cloud/gap-aware series reconstruction.

The smoother solves (W + λDᵀD)z = Wy: a fidelity term per observation
weight (0 for masked/missing, 1 for clean retrievals, fractions for
uncertain QA) against a second-difference roughness penalty. λ=0
interpolates observations exactly; linear signals pass through at any λ
since D² annihilates them; gaps are filled by the smoothest curve the
observations permit.

`fill_series` maps irregular (dates, values[, weights]) onto the regular
daily grid the model consumes — duplicate retrievals merge by weighted
mean. `envelope` iterations reweight points below the fit downward
(upper-envelope Whittaker): atmospheric noise biases vegetation indices
down, not up, so the truth tracks the top of the scatter.
"""

from __future__ import annotations

import datetime as dt

import numpy as numpy
import torch

from nasa_farm.constants import DTYPE

RIDGE = 1e-8
"""Tiny diagonal term making the system positive-definite in all gaps."""


def _diff(t: int, order: int = 2) -> torch.Tensor:
    """(t−order, t) forward-difference matrix."""
    d = torch.eye(t, dtype=DTYPE)
    for _ in range(order):
        d = d[1:] - d[:-1]
    return d


def whittaker(y: torch.Tensor, w: torch.Tensor | None = None,
              lam: float = 100.0, order: int = 2) -> torch.Tensor:
    """(T,) series → smoothed (T,). w=0 cells are filled, not fit."""
    y = y.to(DTYPE)
    t = y.numel()
    w = torch.isfinite(y).to(DTYPE) if w is None else w.to(DTYPE)
    if float(w.sum()) == 0.0:
        return torch.full_like(y, float("nan"))
    d = _diff(t, order)
    a = torch.diag(w) + lam * d.t() @ d + RIDGE * torch.eye(t, dtype=DTYPE)
    b = w * torch.nan_to_num(y)
    return torch.linalg.solve(a, b)


def _reweight(y: torch.Tensor, z: torch.Tensor, w: torch.Tensor,
              floor: float = 0.2) -> torch.Tensor:
    """Upper-envelope update: points below the fit lose weight."""
    under = (torch.nan_to_num(y, nan=-torch.inf) < z) & (w > 0)
    return torch.where(under, w * floor, w)


def _day_of(dates: numpy.ndarray) -> numpy.ndarray:
    """yyyymmdd → ordinal day index within the series' own span."""
    d0 = dt.date(dates.min() // 10000, dates.min() // 100 % 100,
                 dates.min() % 100).toordinal()
    output = numpy.empty(len(dates), numpy.int64)
    for i, d in enumerate(dates):
        output[i] = (dt.date(d // 10000, d // 100 % 100, d % 100).toordinal()
                  - d0)
    return output


def fill_series(dates: numpy.ndarray, values: numpy.ndarray,
                weights: numpy.ndarray | None = None, lam: float = 100.0,
                envelope: int = 0) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Irregular obs → regular daily grid.

    Returns (day_index, filled, obs_weight): obs_weight=0 marks filled
    cells — downstream losses can mask or down-weight them.
    """
    dates = numpy.asarray(dates, numpy.int64)
    index = _day_of(dates)
    n = int(index.max()) + 1
    v = numpy.asarray(values, numpy.float64)
    w_in = numpy.ones_like(v) if weights is None else numpy.asarray(weights, numpy.float64)
    w_in = numpy.where(numpy.isfinite(v), w_in, 0.0)
    number = numpy.zeros(n); den = numpy.zeros(n)
    numpy.add.at(number, index, w_in * numpy.nan_to_num(v))
    numpy.add.at(den, index, w_in)
    w = den.clip(0.0, 1.0)
    y = numpy.where(den > 0, number / numpy.maximum(den, 1e-12), numpy.nan)
    wt, yt = torch.from_numpy(w), torch.from_numpy(y)
    for _ in range(envelope):
        z = whittaker(yt, wt, lam)
        wt = _reweight(yt, z, wt)
    z = whittaker(yt, wt, lam)
    return numpy.arange(n), z.numpy(), w
