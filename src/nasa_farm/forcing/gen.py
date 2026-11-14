"""Generative weather forcing — conditional flow matching on daily vectors.

`WxGen` is a velocity net: conditioned on (seasonal phase, previous-day
normalized weather, flow time τ), it transports standard noise to the next
day's normalized weather by integrating the learned ODE dx/dτ = v. CFM
regresses the analytic velocity x1 − x0 along the linear interpolant —
simulation-free training that recovers the full joint conditional, so
sampled members carry the true precip × radiation × humidity dependence
compound stress needs. Sampling draws x0 ~ N(0, I) per member per day and
integrates `steps` Euler steps — still autoregressive and seeded.

The sampler clamps back to the physical forcing contract (precip ≥ 0,
tmax ≥ tmin, rh ∈ [0, 100]). `metrics.energy_score` remains the ensemble
diagnostic and optional fine-tune objective.

Zero external data required: `fit_gen` trains on whatever history the
caller has — synthetic years out of the box, POWER series when available.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.metrics import energy_score
from nasa_farm.types import Forcing

W_VARS: tuple[str, ...] = ("tmin", "tmax", "srad", "precip", "rh", "u2")
W_SCALE = torch.tensor([30.0, 40.0, 30.0, 50.0, 100.0, 10.0], dtype=DTYPE)
NV = len(W_VARS)


def _series_tensor(series: list[Forcing]) -> torch.Tensor:
    """list of Forcing → (N, T, 6) normalized day vectors."""
    cols = []
    for f in series:
        v = torch.stack([getattr(f, k)[0].to(DTYPE) for k in W_VARS], -1)
        cols.append(v)
    return torch.stack(cols) / W_SCALE


class WxGen(nn.Module):
    """CFM velocity net: (phase, v_{t−1}, x_τ, τ) → dx/dτ toward v_t."""

    def __init__(self, hidden: int = 128, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = 2 + NV + NV + 1
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, NV)]
        self.net = nn.Sequential(*layers)
        self.to(DTYPE)

    def forward(self, phase: torch.Tensor, prev: torch.Tensor,
                x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """phase (B,2), prev (B,6), x (B,6), tau (B,1) → velocity (B,6)."""
        return self.net(torch.cat([phase, prev, x, tau], dim=-1))


def _phase(doy: torch.Tensor) -> torch.Tensor:
    a = 2.0 * math.pi * (doy.to(DTYPE) - 1.0) / DAYS_PER_YEAR
    return torch.stack([a.sin(), a.cos()], dim=-1)


def fit_gen(series: list[Forcing], epochs: int = 400, lr: float = 2e-3,
            hidden: int = 128, members: int = 8,
            seed: int = 0, model: WxGen | None = None) -> WxGen:
    """CFM training over every (day, series-row) transition, vectorized:
    one batch covers all (N × T−1) pairs per epoch."""
    v = _series_tensor(series)
    n, t = v.shape[0], v.shape[1]
    doy = series[0].doy[0]
    m = model or WxGen(hidden)
    if seed is not None:
        torch.manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    prev = v[:, :-1].reshape(-1, NV)                          # (N·(T−1), 6)
    x1 = v[:, 1:].reshape(-1, NV)
    ph = _phase(doy[1:]).unsqueeze(0).expand(n, -1, -1).reshape(-1, 2)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        x0 = torch.randn_like(x1)
        tau = torch.rand(len(x1), 1, generator=gen, dtype=DTYPE)
        xt = (1.0 - tau) * x0 + tau * x1
        loss = torch.nn.functional.mse_loss(m(ph, prev, xt, tau), x1 - x0)
        loss.backward()
        opt.step()
    return m


def _clamp(v: torch.Tensor) -> torch.Tensor:
    """Normalized vector → physical bounds (tmax ≥ tmin handled by caller)."""
    out = v.clone()
    out[..., 2] = out[..., 2].clamp(min=0.5 / W_SCALE[2].item())   # srad
    out[..., 3] = out[..., 3].clamp(min=0.0)                        # precip
    out[..., 4] = out[..., 4].clamp(min=0.0, max=1.0)               # rh
    out[..., 5] = out[..., 5].clamp(min=0.0)                        # u2
    return out


@torch.no_grad()
def sample_gen(m: WxGen, days: int, n: int, start_doy: int = 1,
               seed: int = 0, steps: int = 8,
               prev: torch.Tensor | None = None) -> Forcing:
    """(n, days) ensemble of forcing trajectories, reproducible by seed."""
    gen = torch.Generator().manual_seed(seed)
    ph = lambda d: _phase(torch.full((n,), float(d), dtype=DTYPE))
    v = (prev if prev is not None
         else torch.tensor([0.0, 0.1, 0.5, 0.0, 0.7, 0.2], dtype=DTYPE)
         .expand(n, -1).clone())
    cols = []
    d = start_doy
    taus = torch.linspace(0.0, 1.0, steps + 1, dtype=DTYPE)
    for _ in range(days):
        x = torch.randn(n, NV, generator=gen, dtype=DTYPE)
        for k in range(steps):
            tau = taus[k].expand(n, 1)
            x = x + m(ph(d), v, x, tau) * (1.0 / steps)
        v = _clamp(x)
        cols.append(v)
        d = d % DAYS_PER_YEAR + 1
    w = torch.stack(cols, dim=1) * W_SCALE                    # (n, T, 6)
    tmin = torch.minimum(w[..., 0], w[..., 1])
    tmax = torch.maximum(w[..., 0], w[..., 1])
    doy = ((torch.arange(days, dtype=DTYPE) + start_doy - 1)
           % DAYS_PER_YEAR + 1).expand(n, -1)
    return Forcing(tmin=tmin, tmax=tmax, srad=w[..., 2],
                   precip=w[..., 3], rh=w[..., 4], u2=w[..., 5], doy=doy)
