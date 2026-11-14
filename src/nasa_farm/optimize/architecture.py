"""Neural architecture search over the residual network.

The residual is deliberately small — its search space is a few hundred
points, so the right method here is *not* DARTS (which is for huge spaces):
it is zero-cost proxy ranking → successive halving on short fit budgets.
Zero-cost proxies (SynFlow, Jacobian covariance, grad-norm) score
trainability/expressivity before any fitting, on a non-zero-init probe
clone — the production residual stays exactly-identity at init.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _f
from typing import Any, Callable

import torch
import torch.nn as nn

from nasa_farm.dynamics.residual import ACTS, N_FEATURES
from nasa_farm.optimize.hpo import Trial, successive_halving
from nasa_farm.optimize.space import Param, Space

ARCH_SPACE = Space([
    Param("hidden", choices=(16, 32, 64, 128, 256)),
    Param("depth", choices=(1, 2, 3, 4)),
    Param("activation", choices=tuple(ACTS)),
    Param("layernorm", choices=(False, True)),
    Param("skip", choices=(False, True)),
])


def build_residual(cfg: dict[str, Any], zero_init: bool = True) -> nn.Module:
    """Instantiate a ResidualNet from a NAS config."""
    from nasa_farm.dynamics.residual import ResidualNet
    return ResidualNet(hidden=cfg["hidden"], depth=cfg["depth"],
                       activation=cfg["activation"],
                       layernorm=cfg["layernorm"], skip=cfg["skip"],
                       zero_init=zero_init)


def _main_path(net: nn.Module) -> list[nn.Module]:
    """Forward-path layers only — skips shortcut/auxiliary branches."""
    if hasattr(net, "body") and hasattr(net, "head"):
        return [*net.body, net.head]
    return list(net.modules())


def synflow_score(net: nn.Module, input_dim: int = N_FEATURES,
                  batch: int = 32) -> float:
    """Tan et al. 2020: |W|-product signal propagation, no data needed."""
    net_p = net.to(torch.float64)
    x = torch.ones(batch, input_dim, dtype=torch.float64)
    h = x
    for m in _main_path(net_p):
        if isinstance(m, nn.Linear):
            h = h @ m.weight.detach().abs().t()
    return float(h.sum())


def jacob_cov_score(net: nn.Module, input_dim: int = N_FEATURES,
                    batch: int = 64, seed: int = 0) -> float:
    """Jacobian-covariance proxy (Mellor et al.): feature diversity of the
    mapping — higher effective rank → better-conditioned learner."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, input_dim, generator=gen, dtype=torch.float64)
    jac = torch.autograd.functional.jacobian(
        lambda inp: net.to(torch.float64)(inp).sum(0), x,
        vectorize=True)                                    # (B, out, in)
    j = jac.reshape(batch, -1)
    j = j - j.mean(0, keepdim=True)
    cov = j.t() @ j / batch
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    total = eig.sum().clamp(min=1e-12)
    p = eig / total
    entropy = -(p[p > 1e-12] * p[p > 1e-12].log()).sum()
    return float(entropy)


def grad_norm_score(net: nn.Module, input_dim: int = N_FEATURES,
                    batch: int = 64, seed: int = 0) -> float:
    """Signal magnitude at init — detects dead/exploding gradients early."""
    gen = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(batch, input_dim, generator=gen, dtype=torch.float64)
    for p in net.parameters():
        p.requires_grad_(True)
    loss = net.to(torch.float64)(x).square().mean()
    loss.backward()
    gn = torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters()
                        if p.grad is not None))
    return float(gn)


def zero_cost_rank(cfgs: list[dict], batch: int = 64,
                   seed: int = 0) -> list[tuple[dict, float]]:
    """Rank architectures without fitting: normalized blend of synflow
    (signal propagation), jacobian entropy (feature diversity), and a
    soft grad-norm sanity window. Output layer is un-zeroed for probing."""
    scored = []
    for cfg in cfgs:
        net = build_residual(cfg, zero_init=False)
        sf = synflow_score(net, batch=batch)
        jc = jacob_cov_score(net, batch=batch, seed=seed)
        gn = grad_norm_score(net, batch=batch, seed=seed)
        # synflow spans decades → log-scale; grad-norm wants a sane band
        score = (torch.log10(torch.tensor(sf).clamp(min=1e-12)).item()
                 + jc
                 - abs(torch.log10(torch.tensor(gn).clamp(min=1e-12)).item() + 2))
        scored.append((cfg, score))
    scored.sort(key=lambda t: -t[1])
    return scored


@dataclass(slots=True)
class NASResult:
    best_config: dict[str, Any]
    best_value: float
    leaderboard: list[tuple[dict, float]] = _f(default_factory=list)
    trials: list[Trial] = _f(default_factory=list)


def nas_search(
        fit_objective: Callable[[nn.Module, int], float],
        n_candidates: int = 24,
        proxy_top_k: int = 8,
        rungs: tuple[int, ...] = (10, 30, 90),
        eta: int = 3,
        seed: int = 0,
        arch_space: Space = ARCH_SPACE) -> NASResult:
    """Two-stage NAS:

    1. zero-cost proxy rank on `n_candidates` Sobol-sampled archs → top_k
    2. ASHA over `rungs` fit budgets → champion

    fit_objective(net, budget) → scalar loss (lower better). The caller
    wires the net into a WorldModel and runs trainer.fit internally — this
    keeps NAS decoupled from the model's plumbing.
    """
    cands = arch_space.sample(n_candidates, seed=seed)
    ranked = zero_cost_rank(cands, seed=seed)
    shortlist = [cfg for cfg, _ in ranked[:proxy_top_k]]

    def objective(cfg, budget):
        net = build_residual(cfg, zero_init=True)
        return fit_objective(net, budget)

    trials = successive_halving(objective, shortlist, rungs, eta)
    best = min(trials, key=lambda t: t.value)
    return NASResult(best_config=best.config, best_value=best.value,
                     leaderboard=ranked[:proxy_top_k], trials=trials)
