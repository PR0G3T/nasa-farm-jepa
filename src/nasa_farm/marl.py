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
from nasa_farm.policy import PolicyNet, _day_feats
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


def cap_budget(action_d: dict, budget: float) -> dict:
    """Pro-rata projection of the day's irrigation onto the shared budget —
    differentiable, so agents learn to anticipate rationing."""
    irr = action_d["irrigation"]
    scale = (budget / irr.sum().clamp(min=1e-9)).clamp(max=1.0)
    return {**action_d, "irrigation": irr * scale}


def fit_marl(model, field: Field, forcing: Forcing, base: Actions,
             policies: list[PolicyNet] | None = None,
             budget: float = 15.0, price: float = 0.05,
             cooperative: bool = True,
             weights: dict[str, float] | None = None,
             epochs: int = 40, lr: float = 3e-3,
             u_scale: float = 1.0,
             seed: int = 0) -> tuple[list[PolicyNet], dict]:
    """APG training of B policies through the coupled differentiable rollout.

    Returns (policies, report): report carries the team reward, per-agent
    rewards, mean water use and the min/max spread that exposes the Nash
    gap in competitive mode."""
    b, days = field.batch, forcing.days
    if forcing.batch == 1 and b > 1:
        forcing = forcing.repeat(b)          # shared weather, one row/agent
    policies = policies or [PolicyNet() for _ in range(b)]
    if len(policies) != b:
        raise ValueError(f"{len(policies)} policies for {b} agents")
    torch.manual_seed(seed)
    params = [p for pol in policies for p in pol.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    years = max(1, days // 365)
    weights = weights or DEFAULT_WEIGHTS
    hist: list[float] = []

    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        fld, used = field, torch.zeros(b, dtype=DTYPE)
        acc = SeasonAccumulator(b, years)
        for t in range(days):
            ad = base.day(t)
            feats = _day_feats(model, fld.state, forcing.day(t), ad)
            u = torch.cat([policies[i](feats[i:i + 1])
                           for i in range(b)]) * u_scale
            ad = cap_budget({**ad, "irrigation": u[:, 0],
                             "n_fert": u[:, 1]}, budget)
            used = used + ad["irrigation"]
            fld, fl = model.step(fld, ad, forcing.day(t), t)
            acc(t, fld.state, fl)
        m = acc.result().unsqueeze(1)            # (B, 1, H, M)
        r = metric_reward(m, weights) - price * used
        loss = -(r.mean() if cooperative else r.sum() / b)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(float(r.mean().detach()))

    rep = {"team_reward": round(hist[-1], 4) if hist else float("nan"),
           "agent_rewards": [round(v, 4) for v in r.detach().tolist()],
           "water_used_mm": [round(v, 2) for v in used.detach().tolist()],
           "reward_spread": round(float(r.detach().std()), 4),
           "history": hist}
    return policies, rep
