"""Policy distillation — amortize MPPI into a reactive daily policy.

`PolicyNet` maps the same daily features the residual sees to
(irrigation, n_fert) rates; `distill` trains it to imitate the schedule
MPPI finds, so subsequent control calls warm-start from a policy rollout
instead of zero — the TD-MPC2-style prior that cuts search iterations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nasa_farm.constants import DTYPE
from nasa_farm.control import mppi
from nasa_farm.dynamics.residual import N_FEATURES, featurize
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


class PolicyNet(nn.Module):
    """(B, N_FEATURES) → (B, 2) nonneg daily rates (softplus head)."""

    def __init__(self, hidden: int = 64, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(N_FEATURES, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 2), nn.Softplus()]
        self.net = nn.Sequential(*layers)
        self.to(torch.float64)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


def _day_feats(model, state, forcing_d: dict, action_d: dict) -> torch.Tensor:
    return featurize(
        {"sw0": state.sw[:, 0], "swr": state.sw[:, 1:].sum(-1),
         **{k: getattr(state, k) for k in
            ("n_min", "lai", "biomass", "gdd", "growing")}},
        forcing_d, action_d)


def policy_actions(policy: PolicyNet, model, field: Field,
                   forcing: Forcing, base: Actions) -> Actions:
    """Roll the policy through the model: each day's rates come from the
    policy evaluated on the *current* simulated state — closed-loop, not an
    open-loop schedule replay."""
    b, days = field.batch, forcing.days
    irr = torch.zeros(b, days, dtype=DTYPE)
    fert = torch.zeros(b, days, dtype=DTYPE)
    fld = field
    with torch.inference_mode():
        for t in range(days):
            ad = base.day(t)
            f = _day_feats(model, fld.state, forcing.day(t), ad)
            u = policy(f)
            irr[:, t], fert[:, t] = u[:, 0], u[:, 1]
            ad = {**ad, "irrigation": u[:, 0], "n_fert": u[:, 1]}
            fld, _ = model.step(fld, ad, forcing.day(t))
    return Actions(sow=base.sow, irrigation=irr, n_fert=fert)


def distill(model, field: Field, forcing: Forcing, base: Actions,
            mppi_kw: dict | None = None, epochs: int = 300,
            lr: float = 1e-2, seed: int = 0) -> tuple[PolicyNet, Actions]:
    """Fit the policy to MPPI's closed-loop action/state pairs.

    Returns (policy, teacher_actions): the teacher schedule is also the
    warm-start `base_mu` for subsequent MPPI rounds.
    """
    res = mppi(model, field, forcing, base, seed=seed, **(mppi_kw or {}))
    teacher = res.actions
    b, days = field.batch, forcing.days
    policy = PolicyNet()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    fld = field
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.zeros((), dtype=DTYPE)
        f0 = fld
        for t in range(days):
            ad = base.day(t)
            f = _day_feats(model, f0.state, forcing.day(t), ad)
            u = policy(f)
            loss = loss + ((u[:, 0] - teacher.irrigation[:, t]) ** 2
                           + (u[:, 1] - teacher.n_fert[:, t]) ** 2).mean()
            f0, _ = model.step(f0, ad, forcing.day(t))
        loss.backward()
        opt.step()
    return policy, teacher


def policy_grad(model, field: Field, forcing: Forcing, base: Actions,
                policy: PolicyNet | None = None,
                weights: dict[str, float] | None = None,
                risk: str = "mean", alpha: float = 0.2,
                epochs: int = 60, lr: float = 3e-3,
                u_scale: float = 1.0, checkpoint: int = 0,
                seed: int = 0) -> PolicyNet:
    """Analytic policy gradient through the differentiable dynamics.

    The same closed-loop rollout `distill` runs, but the policy is scored
    directly by `metric_reward` on the accumulated seasonal metrics and the
    gradient flows *through the physics* — pathwise/low-variance learning
    that PPO/SAC-style estimators cannot reach on this model class.
    `u_scale` damps the raw softplus rates into physical bounds.

    field batch B carries scenario replicas (forcing ensemble broadcast) —
    the loss is the scenario-mean reward, so the learned policy is robust
    by construction."""
    from nasa_farm.control import metric_reward
    from nasa_farm.heads import SeasonAccumulator

    policy = policy or PolicyNet()
    torch.manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    years = max(1, forcing.days // 365)
    b, days = field.batch, forcing.days
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        fld = field
        acc = SeasonAccumulator(b, years)
        for t in range(days):
            ad = base.day(t)
            u = policy(_day_feats(model, fld.state, forcing.day(t), ad))
            ad = {**ad, "irrigation": u[:, 0] * u_scale,
                  "n_fert": u[:, 1] * u_scale}
            fld, fl = model.step(fld, ad, forcing.day(t))
            acc(t, fld.state, fl)
        m = acc.result().unsqueeze(1)                # (B, 1, H, M)
        reward = metric_reward(m, weights, risk, alpha).mean()
        (-reward).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    return policy
