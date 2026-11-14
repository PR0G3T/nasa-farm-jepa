"""Multi-agent water allocation over a shared aquifer.

B agents = the batched field's parcels; each runs its own `PolicyNet` on
its own daily features. A shared daily irrigation budget couples them:
when total demand exceeds the budget, every agent is scaled pro-rata —
the commons constraint is a differentiable operation inside the rollout,
so analytic policy gradients flow through the coupling itself (agent i
feels agent j's draw in its own reward gradient).

cooperative=True maximizes the mean team reward; cooperative=False keeps
per-agent rewards separate in the report so the Nash gap is visible.
"""

from __future__ import annotations

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.control import DEFAULT_WEIGHTS, metric_reward
from nasa_farm.heads import SeasonAccumulator
from nasa_farm.policy import Policy, _day_feats
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


def cap_budget(action_d: dict, budget: float) -> dict:
    """Pro-rata projection of the day's irrigation onto the shared budget —
    differentiable, so agents learn to anticipate rationing."""
    irrigation = action_d["irrigation"]
    scale = (budget / irrigation.sum().clamp(min=1e-9)).clamp(max=1.0)
    return {**action_d, "irrigation": irrigation * scale}


def fit_allocation(model, field: Field, forcing: Forcing, base: Actions,
             policies: list[Policy] | None = None,
             budget: float = 15.0, price: float = 0.05,
             cooperative: bool = True,
             weights: dict[str, float] | None = None,
             epochs: int = 40, rate: float = 3e-3,
             u_scale: float = 1.0,
             seed: int = 0) -> tuple[list[Policy], dict]:
    """APG training of B policies through the coupled differentiable rollout.

    Returns (policies, report): report carries the team reward, per-agent
    rewards, mean water use and the min/max spread that exposes the Nash
    gap in competitive mode."""
    b, days = field.batch, forcing.days
    if forcing.batch == 1 and b > 1:
        forcing = forcing.repeat(b)          # shared weather, one row/agent
    policies = policies or [Policy() for _ in range(b)]
    if len(policies) != b:
        raise ValueError(f"{len(policies)} policies for {b} agents")
    torch.manual_seed(seed)
    parameters = [p for policy in policies for p in policy.parameters()]
    opt = torch.optim.Adam(parameters, lr=rate)
    years = max(1, days // 365)
    weights = weights or DEFAULT_WEIGHTS
    history: list[float] = []
    initial = field

    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        field, used = initial, torch.zeros(b, dtype=DTYPE)
        accumulate = SeasonAccumulator(b, years)
        for t in range(days):
            action = base.day(t)
            features = _day_feats(model, field.state, forcing.day(t), action)
            u = torch.cat([policies[i](features[i:i + 1])
                           for i in range(b)]) * u_scale
            action = cap_budget({**action, "irrigation": u[:, 0],
                             "n_fert": u[:, 1]}, budget)
            used = used + action["irrigation"]
            field, fl = model.step(field, action, forcing.day(t), t)
            accumulate(t, field.state, fl)
        m = accumulate.result().unsqueeze(1)            # (B, 1, H, M)
        r = metric_reward(m, weights) - price * used
        loss = -(r.mean() if cooperative else r.sum() / b)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        opt.step()
        history.append(float(r.mean().detach()))

    rep = {"team_reward": round(history[-1], 4) if history else float("nan"),
           "agent_rewards": [round(v, 4) for v in r.detach().tolist()],
           "water_used_mm": [round(v, 2) for v in used.detach().tolist()],
           "reward_spread": round(float(r.detach().std()), 4),
           "history": history}
    return policies, rep


def evaluate_policies(model, field: Field, forcing: Forcing, base: Actions,
                  policies: list, budget: float = 15.0, price: float = 0.05,
                  weights: dict[str, float] | None = None,
                  u_scale: float = 1.0,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll the coupled system under fixed policies → (per-agent reward,
    per-agent water used). Inference-only mirror of the fit_marl loop."""
    b, days = field.batch, forcing.days
    if forcing.batch == 1 and b > 1:
        forcing = forcing.repeat(b)
    weights = weights or DEFAULT_WEIGHTS
    used = torch.zeros(b, dtype=DTYPE)
    accumulate = SeasonAccumulator(b, max(1, days // 365))
    with torch.inference_mode():
        for t in range(days):
            action = base.day(t)
            features = _day_feats(model, field.state, forcing.day(t), action)
            u = torch.cat([policies[i](features[i:i + 1])
                           for i in range(b)]) * u_scale
            action = cap_budget({**action, "irrigation": u[:, 0],
                             "n_fert": u[:, 1]}, budget)
            used = used + action["irrigation"]
            field, fl = model.step(field, action, forcing.day(t), t)
            accumulate(t, field.state, fl)
    m = accumulate.result().unsqueeze(1)
    return metric_reward(m, weights) - price * used, used


def nash_gap(model, field: Field, forcing: Forcing, base: Actions,
             policies: list, budget: float = 15.0, price: float = 0.05,
             levels: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 30.0),
             weights: dict[str, float] | None = None,
             u_scale: float = 1.0) -> dict:
    """Per-agent best-response regret under the shared budget.

    For each agent, every constant irrigation level is substituted for its
    policy output (others held fixed); the gap between the best response
    and the achieved reward quantifies distance from Nash equilibrium.
    """
    r, _ = evaluate_policies(model, field, forcing, base, policies, budget,
                         price, weights, u_scale)
    gaps = []
    for i in range(field.batch):
        best = float(r[i])
        for u in levels:
            def policy(f, _p=policies[i], _u=u):
                output = _p(f).clone()
                output[:, 0] = _u
                return output
            alt = list(policies)
            alt[i] = policy
            ri, _ = evaluate_policies(model, field, forcing, base, alt, budget,
                                  price, weights, u_scale)
            best = max(best, float(ri[i]))
        gaps.append(round(best - float(r[i]), 4))
    return {"nash_gap": gaps, "mean_gap": round(sum(gaps) / len(gaps), 4)}
