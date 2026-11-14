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
            fld, _ = model.step(fld, ad, forcing.day(t), t)
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
            f0, _ = model.step(f0, ad, forcing.day(t), t)
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
            fld, fl = model.step(fld, ad, forcing.day(t), t)
            acc(t, fld.state, fl)
        m = acc.result().unsqueeze(1)                # (B, 1, H, M)
        reward = metric_reward(m, weights, risk, alpha).mean()
        (-reward).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    return policy


class FlowPolicy(nn.Module):
    """Conditional flow-matching action-chunk generator.

    v(a_t, t | feats) over a whole H-day action chunk (2H dims): sampling
    integrates the learned velocity field from noise to a full schedule
    in a few Euler steps. Unlike PolicyNet's unimodal rates, a flow keeps
    multimodal schedules alive (e.g. bimodal early-vs-late irrigation);
    `flow_act` turns a draw into the `base_mu` warm start of mppi."""

    def __init__(self, horizon: int, hidden: int = 128, depth: int = 3,
                 seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.horizon = horizon
        d = 2 * horizon + N_FEATURES + 1
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, 2 * horizon)]
        self.net = nn.Sequential(*layers)
        self.to(torch.float64)

    def forward(self, a: torch.Tensor, t: torch.Tensor,
                feats: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat(
            [a, t.to(a.dtype).expand(a.shape[0], 1), feats], dim=-1))


def fit_flow(policy: FlowPolicy, chunks: torch.Tensor,
             feats: torch.Tensor, epochs: int = 200,
             lr: float = 1e-3, seed: int = 0) -> dict:
    """CFM on (state-feats, action-chunk) demonstrations.

    chunks: (N, 2H) flattened schedules; feats: (N, N_FEATURES) state
    features at chunk start. Target velocity v = a − ε."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    a1, f = chunks.to(DTYPE), feats.to(DTYPE)
    n, hist = a1.shape[0], []
    for _ in range(epochs):
        i = torch.randint(0, n, (n,), generator=gen)
        t = torch.rand(n, 1, generator=gen, dtype=DTYPE)
        eps = torch.randn(a1.shape, generator=gen, dtype=DTYPE)
        pred = policy((1.0 - t) * eps + t * a1[i], t, f[i])
        loss = (pred - (a1[i] - eps)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
    return {"best_loss": round(min(hist), 8), "history": hist}


def flow_sample(policy: FlowPolicy, feats: torch.Tensor, n: int = 1,
                steps: int = 8, seed: int = 0) -> torch.Tensor:
    """feats (B, N_FEATURES) → (B·n, 2H) schedule chunks."""
    gen = torch.Generator().manual_seed(seed)
    b = feats.shape[0]
    f = feats.repeat_interleave(n, dim=0).to(DTYPE)
    a = torch.randn(b * n, 2 * policy.horizon, generator=gen, dtype=DTYPE)
    for s in range(steps):
        t = torch.full((b * n, 1), s / steps, dtype=DTYPE)
        a = a + policy(a, t, f) / steps
    return a.clamp(min=0.0)


def flow_act(policy: FlowPolicy, model, field: Field,
             forcing: Forcing, base: Actions,
             seed: int = 0) -> torch.Tensor:
    """Draw a schedule chunk and return it as mppi's (2, T) `base_mu`."""
    f = _day_feats(model, field.state, forcing.day(0), base.day(0))
    a = flow_sample(policy, f, n=1, seed=seed)[0]
    return a.reshape(2, policy.horizon)


class InverseDyn(nn.Module):
    """Goal-conditioned inverse dynamics: a_t = π(feats_t, g, τ).

    One forward pass replaces the MPPI inner search: given today's
    features, a goal vector g and the remaining-horizon fraction τ, the
    net emits the nonneg daily (irrigation, n_fert) rates that hindsight
    relabeling says moved rollouts toward g — the GC-IDM recipe."""

    def __init__(self, goal_dim: int, hidden: int = 64,
                 depth: int = 2) -> None:
        super().__init__()
        self.goal_dim = goal_dim
        d = N_FEATURES + goal_dim + 1
        layers: list[nn.Module] = [nn.Linear(d, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 2), nn.Softplus()]
        self.net = nn.Sequential(*layers)
        self.to(torch.float64)

    def forward(self, feats: torch.Tensor, goal: torch.Tensor,
                rem: torch.Tensor) -> torch.Tensor:
        g = goal.unsqueeze(0) if goal.dim() == 1 else goal
        return self.net(torch.cat(
            [feats, g.expand(feats.shape[0], -1), rem], dim=-1))


def fit_idm(model, demos: list[tuple[Field, Forcing, Actions]],
            goal_fn=None, epochs: int = 300, lr: float = 1e-2,
            hidden: int = 64, seed: int = 0) -> InverseDyn:
    """Hindsight-relabeled inverse dynamics.

    Each demo (field, forcing, actions) rolls once; the *realized*
    terminal goal g — state z when a latent is present, else the closing
    feature vector — supervises every step's action: (feats_t, g, τ_t)
    → a_t. Conditioning on the outcome rather than the trajectory index
    is what turns imitation into inverse dynamics."""
    rows_x, rows_y = [], []
    for field, forcing, actions in demos:
        fld, rows, days = field, [], forcing.days
        for t in range(days):
            fd, ad = forcing.day(t), actions.day(t)
            rows.append((_day_feats(model, fld.state, fd, ad).detach(),
                         torch.stack([ad["irrigation"], ad["n_fert"]], -1)
                         .to(DTYPE)))
            fld, _ = model.step(fld, ad, fd, t)
        g = (goal_fn(fld.state) if goal_fn
             else fld.state.z if fld.state.z is not None else rows[-1][0])
        for t, (f, u) in enumerate(rows):
            rem = torch.full((f.shape[0], 1), 1.0 - t / days, dtype=DTYPE)
            rows_x.append(torch.cat([f, g.to(DTYPE), rem], -1))
            rows_y.append(u)
    x, y = torch.cat(rows_x), torch.cat(rows_y)
    torch.manual_seed(seed)
    idm = InverseDyn(g.shape[-1], hidden=hidden)
    opt = torch.optim.Adam(idm.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = (idm.net(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    return idm


def idm_plan(idm: InverseDyn, model, field: Field, forcing: Forcing,
             base: Actions, goal: torch.Tensor) -> torch.Tensor:
    """Closed-loop goal-conditioned rollout → (B, T, 2) rates. The goal
    stays fixed while the state evolves — one pass yields either the
    executed policy or mppi's warm-start `base_mu` (transposed)."""
    b, days = field.batch, forcing.days
    mu = torch.zeros(b, days, 2, dtype=DTYPE)
    fld = field
    with torch.inference_mode():
        for t in range(days):
            ad = base.day(t)
            f = _day_feats(model, fld.state, forcing.day(t), ad)
            rem = torch.full((b, 1), 1.0 - t / days, dtype=DTYPE)
            u = idm(f, goal, rem)
            mu[:, t] = u
            ad = {**ad, "irrigation": u[:, 0], "n_fert": u[:, 1]}
            fld, _ = model.step(fld, ad, forcing.day(t), t)
    return mu


def idm_actions(idm: InverseDyn, model, field: Field, forcing: Forcing,
                base: Actions, goal: torch.Tensor) -> Actions:
    """Goal-conditioned closed-loop control — the IDM `policy_actions`."""
    mu = idm_plan(idm, model, field, forcing, base, goal)
    return Actions(sow=base.sow, irrigation=mu[..., 0], n_fert=mu[..., 1])
