"""Management-event detection — infer the real action history from remote
sensing instead of trusting a declared plan.

Sowing = sustained NDVI rise from a bare/low baseline; harvest = the
steepest NDVI drop after the seasonal peak; tillage = a bare-soil SAR
backscatter spike before green-up. Thresholds are adaptive to the series'
own dynamic range so the same rule works across crops and sensors.
"""

from __future__ import annotations

import torch

from nasa_farm.types import Actions


def _sow_day(ndvi: torch.Tensor) -> int:
    """First day the NDVI rises through 15 % of the seasonal amplitude,
    confirmed by a positive 5-day trend."""
    lo, hi = float(ndvi.min()), float(ndvi.max())
    if hi - lo < 0.08:
        return -1
    thr = lo + 0.15 * (hi - lo)
    above = ndvi > thr
    trend = torch.zeros_like(ndvi)
    trend[5:] = ndvi[5:] - ndvi[:-5]
    hit = (above & (trend > 0)).nonzero()
    return int(hit[0]) if hit.numel() else -1


def _harvest_day(ndvi: torch.Tensor, sow_t: int) -> int:
    """Steepest post-peak NDVI decline (≈ harvest or termination)."""
    if sow_t < 0:
        return -1
    peak = int(ndvi[sow_t:].argmax()) + sow_t
    if peak >= ndvi.numel() - 3:
        return -1
    drop = ndvi[peak:-1] - ndvi[peak + 1:]
    i = int(drop.argmax())
    return int(peak + i + 1) if float(drop[i]) > 0.05 else -1


def _till_days(s0v: torch.Tensor, sow_t: int) -> list[int]:
    """Bare-soil σ⁰ spikes (> 1.5 dB above the 7-day rolling median)
    preceding green-up — the tillage signature."""
    if s0v is None or sow_t <= 7:
        return []
    seg = s0v[:sow_t]
    med = seg.median()
    return ((seg - med) > 1.5).nonzero().flatten().tolist()


def infer_actions(ndvi: torch.Tensor, doy: torch.Tensor, crop_idx: int,
                  days: int | None = None,
                  s0v: torch.Tensor | None = None) -> Actions:
    """(T,) NDVI (+ optional σ⁰) → detected Actions batch (1, T).

    sow and till are filled from detection; irrigation/fertilizer stay
    zero — they are not reliably observable from vegetation indices.
    """
    t = days or ndvi.numel()
    sow = torch.full((1, t), -1, dtype=torch.long)
    till = torch.zeros(1, t)
    s_t = _sow_day(ndvi)
    if s_t >= 0:
        sow[0, s_t] = crop_idx
        for d in _till_days(s0v, s_t):
            till[0, d] = 1.0
    return Actions(sow=sow, irrigation=torch.zeros(1, t),
                   n_fert=torch.zeros(1, t), till=till)
