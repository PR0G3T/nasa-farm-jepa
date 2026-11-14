"""Generative weather forcing — noise-injection net trained on proper scores.

`WxGen` maps (seasonal phase, previous-day normalized weather, fresh noise)
to the next day's normalized weather. Training minimizes the *energy score*
— E‖X̂−y‖ − ½E‖X̂−X̂′‖ over K sampled members per day — a strictly proper
multivariate scoring rule, so the learned ensemble preserves the joint
precip × radiation × humidity dependence that compound heat–drought stress
needs. One forward per member per day at sampling time.

The sampler is autoregressive and seeded: each member is a full-horizon
trajectory, physically clamped back to the forcing contract (precip ≥ 0,
tmax ≥ tmin, rh ∈ [0, 100]).

Zero external data required: `fit_gen` trains on whatever history the
caller has — synthetic years out of the box, POWER series when available.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.types import Forcing

W_VARS: tuple[str, ...] = ("tmin", "tmax", "srad", "precip", "rh", "u2")
W_SCALE = torch.tensor([30.0, 40.0, 30.0, 50.0, 100.0, 10.0], dtype=DTYPE)


def _series_tensor(series: list[Forcing]) -> torch.Tensor:
    """list of Forcing → (N, T, 6) normalized day vectors."""
    cols = []
    for f in series:
        v = torch.stack([getattr(f, k)[0].to(DTYPE) for k in W_VARS], -1)
        cols.append(v)
    return torch.stack(cols) / W_SCALE


class WxGen(nn.Module):
    """Conditional one-step generator: (phase, v_{t-1}, eps) → v_t."""

    def __init__(self, hidden: int = 128, depth: int = 3,
                 noise_dim: int = 16) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        layers: list[nn.Module] = []
        d = 2 + len(W_VARS) + noise_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, len(W_VARS))]
        self.net = nn.Sequential(*layers)
        self.to(DTYPE)

    def forward(self, phase: torch.Tensor, prev: torch.Tensor,
                eps: torch.Tensor) -> torch.Tensor:
        """phase (B,2) sin/cos doy, prev (B,6), eps (B,N) → v_t (B,6)."""
        x = torch.cat([phase, prev, eps], dim=-1)
        return prev + self.net(x)


def _phase(doy: torch.Tensor) -> torch.Tensor:
    a = 2.0 * math.pi * (doy.to(DTYPE) - 1.0) / DAYS_PER_YEAR
    return torch.stack([a.sin(), a.cos()], dim=-1)


def energy_score(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fair energy score. pred (K, B, V) members, y (B, V) target."""
    k = pred.shape[0]
    term1 = (pred - y.unsqueeze(0)).norm(dim=-1).mean(0)
    d = (pred.unsqueeze(0) - pred.unsqueeze(1)).norm(dim=-1)  # (K,K,B,V)
    term2 = 0.5 * d.mean(dim=(0, 1)) * (k / max(1, k - 1))
    return (term1 - term2).mean()


def fit_gen(series: list[Forcing], epochs: int = 400, lr: float = 2e-3,
            hidden: int = 128, members: int = 8,
            seed: int = 0, model: WxGen | None = None) -> WxGen:
    """Energy-score training over every (day, series-row) transition."""
    v = _series_tensor(series)
    n, t = v.shape[0], v.shape[1]
    doy = series[0].doy[0]
    m = model or WxGen(hidden)
    if seed is not None:
        torch.manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), dtype=DTYPE)
        for s in range(1, t):
            prev = v[:, s - 1].repeat_interleave(members, 0)  # (N·K, 6)
            ph = _phase(doy[s].expand(n)).repeat_interleave(members, 0)
            eps = torch.randn(n * members, m.noise_dim, generator=gen,
                              dtype=DTYPE)
            pred = m(ph, prev, eps).view(members, n, -1)
            loss = loss + energy_score(pred, v[:, s])
        loss = loss / (t - 1)
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
               seed: int = 0, prev: torch.Tensor | None = None) -> Forcing:
    """(n, days) ensemble of forcing trajectories, reproducible by seed."""
    gen = torch.Generator().manual_seed(seed)
    ph = lambda d: _phase(torch.full((n,), float(d), dtype=DTYPE))
    v = (prev if prev is not None
         else torch.tensor([0.0, 0.1, 0.5, 0.0, 0.7, 0.2], dtype=DTYPE)
         .expand(n, -1).clone())
    cols = []
    d = start_doy
    for _ in range(days):
        eps = torch.randn(n, m.noise_dim, generator=gen, dtype=DTYPE)
        v = _clamp(m(ph(d), v, eps))
        cols.append(v)
        d = d % DAYS_PER_YEAR + 1
    w = torch.stack(cols, dim=1) * W_SCALE                    # (n, T, 6)
    tmin = torch.minimum(w[..., 0], w[..., 1])
    tmax = torch.maximum(w[..., 0], w[..., 1])
    doy = ((torch.arange(days, dtype=DTYPE) + start_doy - 1)
           % DAYS_PER_YEAR + 1).expand(n, -1)
    return Forcing(tmin=tmin, tmax=tmax, srad=w[..., 2],
                   precip=w[..., 3], rh=w[..., 4], u2=w[..., 5], doy=doy)
