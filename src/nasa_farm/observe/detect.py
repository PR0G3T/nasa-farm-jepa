"""Management-event detection — infer the real action history from remote
sensing instead of trusting a declared plan.

Sowing = sustained NDVI rise from a bare/low baseline, refined to the
interferometric-coherence drop SAR shows at soil disturbance; harvest =
the steepest NDVI drop after the seasonal peak; tillage = a bare-soil SAR
backscatter spike before green-up; irrigation = the soil-water-balance
inversion (SM2RAIN): water entering the profile that rain cannot
explain. Thresholds are adaptive to the series' own dynamic range so the
same rules work across crops and sensors.
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


def _sow_coh(coh: torch.Tensor, s_t: int,
             back: int = 15, fwd: int = 5) -> int:
    """Refine an NDVI sowing estimate to the coherence dip SAR shows at
    soil disturbance: local minimum in [s−back, s+fwd] below the segment
    median, else keep the NDVI day."""
    if coh is None or s_t < 0:
        return s_t
    lo, hi = max(0, s_t - back), min(coh.numel(), s_t + fwd + 1)
    seg = torch.nan_to_num(coh[lo:hi], nan=torch.inf)
    if not torch.isfinite(seg).all():
        return s_t
    i = int(seg.argmin())
    return lo + i if float(seg[i]) < float(seg.median()) else s_t


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


def _rolling_median(x: torch.Tensor, window: int) -> torch.Tensor:
    """Centered rolling median; edges replicate the boundary values."""
    lo, hi = window // 2, window - 1 - window // 2
    xp = torch.cat([x[0].expand(lo), x, x[-1].expand(hi)])
    return xp.unfold(0, window, 1).median(-1).values


def infer_irrigation(sm: torch.Tensor, precip: torch.Tensor,
                     depth_mm: float = 200.0, loss: float = 0.08,
                     min_mm: float = 0.5, window: int = 15) -> torch.Tensor:
    """SM2RAIN inversion — daily irrigation (mm) from surface SM + rain.

    Water balance: Z·ds/dt = p + i − e(s) − g(s), so the water input the
    balance requires is Z·(Δs + k·s); whatever rain does not supply must
    be irrigation. The rolling-median background absorbs the constant
    maintenance flux (drainage at equilibrium is not an event), leaving
    irrigation as the positive anomaly: i = max(0, need − base − min_mm).
    k lumps ET+drainage as a fraction of storage per day.
    Inputs (T,): sm in m³/m³, precip in mm.
    """
    sm, precip = (torch.nan_to_num(x.to(torch.float64)) for x in (sm, precip))
    ds = torch.zeros_like(sm)
    ds[1:] = sm[1:] - sm[:-1]
    need = depth_mm * (ds + loss * sm) - precip
    base = _rolling_median(need, min(window, sm.numel()))
    return (need - base - min_mm).clamp(min=0.0)


def infer_actions(ndvi: torch.Tensor, doy: torch.Tensor, crop_idx: int,
                  days: int | None = None,
                  s0v: torch.Tensor | None = None,
                  sm: torch.Tensor | None = None,
                  precip: torch.Tensor | None = None,
                  coh: torch.Tensor | None = None) -> Actions:
    """(T,) NDVI (+ σ⁰, sm/precip, coherence) → detected Actions (1, T).

    sow is NDVI-derived then refined to the InSAR-coherence disturbance
    dip when `coh` is given; till from the σ⁰ spike; irrigation from the
    SM2RAIN inversion when `sm` and `precip` are supplied; fertilizer
    stays zero — not reliably observable from space.
    """
    t = days or ndvi.numel()
    sow = torch.full((1, t), -1, dtype=torch.long)
    till = torch.zeros(1, t)
    s_t = _sow_coh(coh, _sow_day(ndvi))
    if s_t >= 0:
        sow[0, s_t] = crop_idx
        for d in _till_days(s0v, s_t):
            till[0, d] = 1.0
    irr = (infer_irrigation(sm, precip) if sm is not None
           and precip is not None else torch.zeros(t))
    return Actions(sow=sow, irrigation=irr.unsqueeze(0).to(torch.float64),
                   n_fert=torch.zeros(1, t, dtype=torch.float64), till=till)
