"""Learned generative forcing — conditional flow matching over daily weather.

A neural vector field transports a Gaussian base to the distribution of the
next day's normalized weather vector, conditioned on the previous day and
the day-of-year phase. Sampling integrates the learned ODE by Euler steps,
then auto-regresses day by day — temporally coherent ensemble members
instead of independent climatological noise.

Zero external data required: `fit_flow` trains on whatever history the
caller supplies (POWER series, reanalysis, or synthetic years for tests).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.types import Forcing

FLOW_VARS: tuple[str, ...] = ("tmin", "tmax", "srad", "precip", "rh", "u2")
V = len(FLOW_VARS)


class FlowGen(nn.Module):
    """Conditional vector field + climatological normalizer."""

    def __init__(self, hidden: int = 128, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(V + 1 + V + 2, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, V)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(V, dtype=torch.float64))
        self.register_buffer("sd", torch.ones(V, dtype=torch.float64))
        self.to(torch.float64)

    def velocity(self, x: torch.Tensor, tau: torch.Tensor,
                 ctx: torch.Tensor) -> torch.Tensor:
        """u(x_t, τ | ctx) — target direction toward the data sample."""
        return self.net(torch.cat([x, tau.unsqueeze(-1), ctx], dim=-1))

    def _ctx(self, doy: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        ph = 2.0 * math.pi * doy / DAYS_PER_YEAR
        return torch.cat([torch.sin(ph).unsqueeze(-1),
                          torch.cos(ph).unsqueeze(-1), prev], dim=-1)

    def normalize(self, v: torch.Tensor) -> torch.Tensor:
        return (v - self.mu) / self.sd

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sd + self.mu


def _series_tensor(series: list[Forcing]) -> torch.Tensor:
    """list of years → (N, T, V) in physical units."""
    return torch.stack([
        torch.stack([getattr(y, k)[0] for k in FLOW_VARS], dim=-1)
        for y in series])


def fit_flow(series: list[Forcing], epochs: int = 400, lr: float = 2e-3,
             hidden: int = 128, seed: int = 0,
             model: FlowGen | None = None) -> FlowGen:
    """Train the conditional vector field on daily transitions.

    Loss: E‖u(x_τ, τ, ctx) − (x1 − x0)‖² over τ~U(0,1), x0~N(0,I),
    x_τ = τ·x1 + (1−τ)·x0 — standard flow-matching regression.
    """
    data = _series_tensor(series).to(DTYPE)
    n, t, _ = data.shape
    m = model or FlowGen(hidden)
    m.mu.copy_(data.reshape(-1, V).mean(0))
    m.sd.copy_(data.reshape(-1, V).std(0).clamp(min=1e-3))
    x = m.normalize(data)
    prev = torch.cat([x[:, -1:], x[:, :-1]], dim=1)  # day t−1 (wraps)
    doy = torch.arange(1, t + 1, dtype=DTYPE).expand(n, t)

    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        tau = torch.rand(n, t, generator=g, dtype=DTYPE)
        x0 = torch.randn(n, t, V, generator=g, dtype=DTYPE)
        x_t = tau.unsqueeze(-1) * x + (1.0 - tau.unsqueeze(-1)) * x0
        ctx = torch.stack([m._ctx(doy[:, i], prev[:, i])
                           for i in range(t)], dim=1)
        u = m.velocity(x_t.reshape(-1, V), tau.reshape(-1),
                       ctx.reshape(-1, ctx.shape[-1]))
        loss = (u - (x - x0).reshape(-1, V)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m


@torch.no_grad()
def sample_flow(m: FlowGen, days: int, n: int, start_doy: int = 1,
                steps: int = 8, seed: int = 0) -> Forcing:
    """Generate an (n, days) ensemble by Euler-integrating the learned ODE."""
    g = torch.Generator().manual_seed(seed)
    prev = torch.zeros(n, V, dtype=DTYPE)
    cols = []
    for t in range(days):
        doy = torch.full((n,), float((start_doy - 1 + t) % DAYS_PER_YEAR + 1),
                         dtype=DTYPE)
        x = torch.randn(n, V, generator=g, dtype=DTYPE)
        ctx = m._ctx(doy, prev)
        for k in range(steps):
            tau = torch.full((n,), k / steps, dtype=DTYPE)
            x = x + m.velocity(x, tau, ctx) / steps
        prev = x
        cols.append(m.denormalize(x))
    v = torch.stack(cols, dim=1)                     # (n, days, V)
    tmin, tmax = v[..., 0], torch.maximum(v[..., 1], v[..., 0])
    return Forcing(
        tmin=tmin, tmax=tmax,
        srad=v[..., 2].clamp(min=0.5), precip=v[..., 3].clamp(min=0.0),
        rh=v[..., 4].clamp(5.0, 100.0), u2=v[..., 5].clamp(min=0.1),
        doy=((torch.arange(days) + start_doy - 1) % DAYS_PER_YEAR + 1)
        .to(DTYPE).expand(n, days),
    )
