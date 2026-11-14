"""Value bootstrap — amortized expected seasonal return.

`ValueNet` approximates V(s) = E[objective(rest of season) | state s],
trained on the model's own truncated rollouts: every sampled day
contributes (state features at t, realized objective of the t→end suffix).
That is the value head a TD-MPC2-style planner needs: MPPI then searches a
short horizon and lets V price the remainder instead of rolling 365 days
per candidate — O(horizon) per candidate, identical objective in
expectation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nasa_farm.constants import DTYPE
from nasa_farm.control import metric_reward
from nasa_farm.dynamics.residual import N_FEATURES
from nasa_farm.heads import SeasonAccumulator
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


class Value(nn.Module):
    """(B, N_FEATURES) daily features → (B,) expected seasonal reward."""

    def __init__(self, hidden: int = 64, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(N_FEATURES, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.to(DTYPE)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.to(DTYPE)).squeeze(-1)


def _suffix_reward(model, field: Field, forcing: Forcing, actions: Actions,
                   t0: int, weights: dict | None, risk: str,
                   alpha: float) -> torch.Tensor:
    """Objective of the t0→end suffix rollout, per batch row."""
    days = forcing.days - t0
    yrs = max(1, -(-days // 365))
    from nasa_farm.types import FORCING_VARS, ACTION_VARS
    forcing = Forcing(*(getattr(forcing, k)[:, t0:] for k in FORCING_VARS))
    act = Actions(*(getattr(actions, k)[:, t0:] for k in ACTION_VARS))
    accumulate = SeasonAccumulator(field.batch, yrs)
    with torch.inference_mode():
        model.rollout(field, forcing, act, observer=accumulate)
    return metric_reward(accumulate.result().unsqueeze(1), weights, risk, alpha)


def fit_value(model, field: Field, forcing: Forcing, actions: Actions,
              net: Value | None = None,
              weights: dict[str, float] | None = None,
              risk: str = "mean", alpha: float = 0.2,
              every: int = 15, epochs: int = 200,
              rate: float = 1e-3) -> Value:
    """Regression on suffix-rollout returns.

    One inference pass caches states on the `every` grid; each cached state
    then rolls its suffix once to score the return — T/every short
    rollouts instead of a quadratic count, and V sees on-policy states.
    """
    days = forcing.days
    grid = list(range(0, days - 30, every))
    snapshot: dict[int, FieldState] = {}
    f = field
    with torch.inference_mode():
        for t in range(days):
            if t in grid:
                snapshot[t] = f.state.clone()
            f, _ = model.step(f, actions.day(t), forcing.day(t), t)

    net = net or Value()
    xs, ys = [], []
    for t, saved in snapshot.items():
        f = Field(state=saved, soil=field.soil, latitude=field.latitude,
                  elevation=field.elevation, context=field.context)
        y = _suffix_reward(model, f, forcing, actions, t, weights, risk,
                           alpha)
        features = model._features(saved, forcing.day(min(t, days - 1)),
                                actions.day(min(t, actions.days - 1)))
        xs.append(features)
        ys.append(y)
    x = torch.cat(xs)
    y = torch.cat(ys)
    deviation = y.std().clamp(min=1.0)
    opt = torch.optim.Adam(net.parameters(), lr=rate)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = ((net(x) - y) / deviation).pow(2).mean()
        loss.backward()
        opt.step()
    net._scale = deviation
    return net


def terminal_value(model, net: Value, state, forcing_day: dict,
                   action_d: dict, gamma: float = 1.0) -> torch.Tensor:
    """γ·V(s_end) — the bootstrapped remainder of the season."""
    features = model._features(state, forcing_day, action_d)
    with torch.inference_mode():
        return gamma * net(features)
