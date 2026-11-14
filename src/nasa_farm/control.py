"""Model-predictive control over daily irrigation / fertilization.

MPPI (Model-Predictive Path Integral): a mean action sequence is perturbed
by colored (Ornstein–Uhlenbeck) noise, candidates are scored by rollout
reward averaged over forcing scenarios, and the mean shifts toward a
rank-weighted recombination of the elite quartile. The incumbent is
re-scored every round and the best mean ever evaluated is returned — the
result can never be worse than `base`.

Extensions on the same scaffold:

- `modes` > 1: mixture-MPPI — elites are clustered into `modes` mean
  sequences, so the sampler keeps several incompatible hypotheses alive
  instead of averaging them (bimodal futures, e.g. early vs late sow).
- `refine` > 0: gradient refinement — the incumbent plan is polished by
  Adam directly on the differentiable rollout reward (the model is
  end-to-end differentiable; sampling finds the basin, gradients climb
  it). Checkpointed rollouts bound activation memory.
- `goal`: terminal latent subgoal — a (K,) target the rollout's final z
  is held against, the planning face of hierarchical latents.
- `FbRisk`: feedback conformal risk allocation for replanning loops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _f

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.forcing import noise
from nasa_farm.heads import METRICS, SeasonAccumulator
from nasa_farm.state.field import Field
from nasa_farm.types import ACTION_VARS, FORCING_VARS, Actions, Forcing

DEFAULT_WEIGHTS: dict[str, float] = {
    "yield_kg": 1.0, "water_used": -0.3, "n_leached": -2.0,
    "stress_days": -3.0, "frost_days": -50.0, "heat_dd": -2.0,
}


@dataclass(frozen=True, slots=True)
class ControlResult:
    actions: Actions
    reward: float           # best mean-scenario reward seen (>= base's)
    history: tuple[float, ...]  # running best, one entry per iteration


@dataclass(slots=True)
class Risk:
    """Feedback conformal risk allocation: a total miscoverage budget of
    `base`·`horizon` is spent as realized errors arrive; `margin()` spreads
    what is left over the remaining steps, so an early bad stretch tightens
    later steps and a clean stretch relaxes them (Fb-CP)."""

    base: float             # per-step margin
    horizon: int
    t: int = 0
    budget: float = _f(init=False)

    def __post_init__(self) -> None:
        self.budget = self.base * self.horizon

    def margin(self) -> float:
        return self.budget / max(self.horizon - self.t, 1)

    def update(self, error: float) -> None:
        self.budget = max(0.0, self.budget - error)
        self.t += 1


def metric_reward(metrics: torch.Tensor,
                  weights: dict[str, float] | None = None,
                  risk: str = "mean", alpha: float = 0.2,
                  margin: torch.Tensor | float = 0.0) -> torch.Tensor:
    """(S, E, H, M) → (S,) reward: weighted metric sum over years, aggregated
    over forcing scenarios.

    risk: "mean" scenario average; "cvar" mean of the worst-α tail
    (drought-robust scheduling); "conf" mean minus a conformal calibration
    margin (chance-constrained reward).
    """
    weights = weights or DEFAULT_WEIGHTS
    index = {n: i for i, n in enumerate(METRICS)}
    r = torch.zeros(metrics.shape[0], metrics.shape[1], dtype=DTYPE)
    for name, w in weights.items():
        r = r + w * metrics[..., index[name]].sum(dim=-1)
    if risk == "cvar":
        k = max(1, int(round(alpha * r.shape[1])))
        return r.sort(dim=1).values[:, :k].mean(1)
    if risk == "conf":
        return r.mean(1) - torch.as_tensor(margin, dtype=DTYPE)
    return r.mean(1)


def _means(x: torch.Tensor, init: torch.Tensor,
            iterations: int = 8) -> torch.Tensor:
    """Assignment of rows x (E, D) to the nearest of `init` (M, D)
    centers — enough structure for elite clustering, no full solver."""
    c = init
    for _ in range(iterations):
        a = torch.cdist(x, c).argmin(dim=1)
        for m in range(init.shape[0]):
            sel = a == m
            if sel.any():
                c = torch.cat([c[:m], x[sel].mean(0, keepdim=True),
                               c[m + 1:]])
    return torch.cdist(x, c).argmin(dim=1)


def _quasi_newton(loss_fn, parameters: list[torch.Tensor], iterations: int,
           m: int = 8) -> None:
    """In-place L-BFGS polish: two-loop recursion + Armijo backtracking.

    Second-order information accumulated from gradient differences —
    converges on smooth differentiable objectives in far fewer rollout
    evaluations than first-order Adam.
    """
    s_h, y_h, rho = [], [], []

    def flat() -> tuple[torch.Tensor, torch.Tensor]:
        for p in parameters:
            p.grad = None
        v = loss_fn()
        v.backward()
        return v, torch.cat([p.grad.reshape(-1) for p in parameters])

    v, g = flat()
    for _ in range(iterations):
        q, al = g.clone(), []
        for s, y, r in zip(s_h[::-1], y_h[::-1], rho[::-1]):
            a = r * (s * q).sum()
            q = q - a * y
            al.append(a)
        d = -q
        for s, y, r, a in zip(s_h, y_h, rho, al[::-1]):
            d = d + s * (a - r * (y * d).sum())
        step, base = 1.0, torch.cat([p.detach().reshape(-1) for p in parameters])
        vn = None
        for _ls in range(20):
            off = 0
            with torch.no_grad():
                for p in parameters:
                    p.copy_((base[off:off + p.numel()] + step * d[off:off + p.numel()])
                            .view_as(p))
                    off += p.numel()
            vn = loss_fn()
            if float(vn) <= float(v) - 1e-4 * step * abs(float((d * g).sum())):
                break
            step *= 0.5
        else:
            off = 0
            with torch.no_grad():
                for p in parameters:
                    p.copy_(base[off:off + p.numel()].view_as(p))
                    off += p.numel()
            break
        vn, gn = flat()
        s, y = step * d, gn - g
        sy = float((s * y).sum())
        if sy > 1e-12:
            s_h.append(s), y_h.append(y), rho.append(1.0 / sy)
            s_h, y_h, rho = s_h[-m:], y_h[-m:], rho[-m:]
        v, g = vn, gn


def path_integral(model, field: Field, forcing: Forcing, base: Actions,
         iterations: int = 8, samples: int = 96, rho: float = 0.7,
         sigma: tuple[float, float] = (4.0, 10.0),
         bounds: tuple[tuple[float, float], tuple[float, float]] =
         ((0.0, 25.0), (0.0, 10.0)),
         weights: dict[str, float] | None = None,
         risk: str = "mean", alpha: float = 0.2,
         margin: torch.Tensor | float = 0.0,
         base_mu: torch.Tensor | None = None,
         sow_shifts: tuple[int, ...] = (0,),
         horizon: int = 0, value=None, gamma: float = 1.0,
         reward=None, goal: torch.Tensor | None = None,
         goal_w: float = 0.0,
         modes: int = 1, refine: int = 0, refine_lr: float = 0.1,
         refine_alg: str = "adam", gate_k: float = 0.0,
         seed: int = 0) -> ControlResult:
    """Optimize continuous irrigation (mm/day) and N (kg/ha/day) channels.

    `base` supplies sow/harvest structure; forcing may be an ensemble (E>1) —
    the reward is the scenario mean, so the schedule is robust by construction.
    `reward` overrides `metric_reward` for custom objectives.

    `horizon` < days truncates candidate rollouts and adds γ·`value`(s_H):
    the ValueNet bootstrap prices the rest of the season, so search cost
    scales with the planning horizon instead of the full year.
    """
    days = forcing.days
    E = forcing.batch
    generation = torch.Generator().manual_seed(seed)
    mu0 = torch.stack([base.irrigation[0], base.n_fert[0]])  # (2, T)
    if base_mu is not None:                      # policy-prior warm start
        mu0 = base_mu.to(mu0.dtype)
    means = mu0.unsqueeze(0).expand(modes, -1, -1).contiguous()  # (M, 2, T)
    best_r, best_mu, best_sow = -math.inf, mu0, base.sow[0]
    history: list[float] = []
    n_shifts = len(sow_shifts)
    h = horizon if 0 < horizon < days else days

    def bonus(field: Field, forcing: Forcing, acts: Actions) -> torch.Tensor:
        """Terminal extras: value bootstrap + latent subgoal penalty."""
        b = torch.zeros(field.batch, dtype=DTYPE, device=field.state.sw.device)
        if value is not None:
            from nasa_farm.value import terminal_value
            b = b + terminal_value(model, value, field.state,
                                   forcing.day(min(h - 1, days - 1)),
                                   acts.day(h - 1), gamma)
        if goal is not None and field.state.z is not None:
            g = goal.to(field.state.z.dtype)
            b = b - goal_w * (field.state.z[:, :g.shape[0]] - g).pow(2).sum(-1)
        return b

    def score(candidate: torch.Tensor, sows: torch.Tensor) -> torch.Tensor:
        """(2, S, T) candidates × (S, T) sow rows → (S,) scenario rewards."""
        s_n = candidate.shape[1]
        b = s_n * E
        acts = Actions(
            sow=sows.repeat_interleave(E, dim=0),
            irrigation=candidate[0].repeat_interleave(E, dim=0),
            n_fert=candidate[1].repeat_interleave(E, dim=0))
        repeated = forcing.repeat(s_n)
        if h < days:
            repeated = Forcing(*(getattr(repeated, k)[:, :h] for k in FORCING_VARS))
            acts = Actions(*(getattr(acts, k)[:, :h] for k in ACTION_VARS))
        accumulator = SeasonAccumulator(b, max(1, -(-h // 365)),
                                device=field.state.sw.device)
        with torch.inference_mode():
            rolled = model.rollout(field.repeat(b), repeated, acts, observer=accumulator)
        m = accumulator.result().view(s_n, E, -1, accumulator.result().shape[-1])
        r = (reward(m) if reward is not None
             else metric_reward(m, weights, risk, alpha, margin))
        return r.view(s_n, -1).mean(1) + bonus(rolled, repeated, acts) \
            .view(s_n, E).mean(1)

    r_sig = 0.0
    for _ in range(iterations):
        per = -(-samples // modes)
        parts, sows_l = [], []
        for mi in range(modes):
            eps = torch.stack([noise(per, days, rho, generation),
                               noise(per, days, rho, generation)])
            parts.append(torch.stack(
                [(means[mi, c] + eps[c] * sigma[c]).clamp(*bounds[c])
                 for c in range(2)]))                  # (2, per, T)
            sows_l.append(torch.stack([
                base.sow[0].roll(sow_shifts[i % n_shifts], dims=0)
                for i in range(per)]))
        candidate = torch.cat(parts, dim=1)                 # (2, S, T)
        sows = torch.cat(sows_l)
        candidate[:, 0] = best_mu  # incumbent: re-scored each round
        r = score(candidate, sows)
        r_sig = float(r.std())

        i_star = int(r.argmax())
        if float(r[i_star]) > best_r:
            best_r, best_mu = float(r[i_star]), candidate[:, i_star]
            best_sow = sows[i_star]
        history.append(best_r)
        # Rank-based elite recombination (CMA-style): only the top quartile
        # moves the mean — scale-free and immune to the drift a flat softmax
        # shows when the optimum sits on a bound.
        elite = max(modes, candidate.shape[1] // 4)
        order = r.argsort(descending=True)
        el = candidate[:, order[:elite]]                    # (2, E_e, T)
        if modes > 1:
            flat = torch.cat([el[0] / sigma[0], el[1] / sigma[1]], dim=-1)
            flat0 = torch.cat([means[:, 0] / sigma[0],
                               means[:, 1] / sigma[1]], dim=-1)
            assign = _means(flat, flat0)
            w = (math.log(elite + 0.5)
                 - torch.log(torch.arange(1, elite + 1, dtype=DTYPE)))
            for mi in range(modes):
                sel = assign == mi
                if not sel.any():
                    continue
                wm = (w[sel] / w[sel].sum()).unsqueeze(0).unsqueeze(-1)
                means[mi] = means[mi] + (wm * (el[:, sel] - means[mi]
                                           .unsqueeze(1))).sum(1)
        else:
            w = torch.zeros(candidate.shape[1], dtype=DTYPE)
            w[order[:elite]] = (math.log(elite + 0.5) - torch.log(
                torch.arange(1, elite + 1, dtype=DTYPE)))
            w = w / w.sum()
            means[0] = means[0] + (w.unsqueeze(0).unsqueeze(-1)
                               * (candidate - means[0].unsqueeze(1))).sum(1)

    if gate_k and best_mu is not mu0:
        # Sibling-disagreement gate: reject a plan whose margin over the
        # incumbent base is below gate_k·σ of the candidate scores — rollout
        # disagreement marks where the model cannot tell plans apart.
        r_base = float(score(mu0.unsqueeze(1), base.sow[0].unsqueeze(0))[0])
        if best_r < r_base + gate_k * r_sig:
            best_r, best_mu, best_sow = r_base, mu0, base.sow[0]

    if refine:
        # Gradient polish on the incumbent sequence through the
        # differentiable rollout — sampling found the basin, this climbs it.
        irrigation = best_mu[0].clone().requires_grad_(True)
        nf = best_mu[1].clone().requires_grad_(True)

        def polish() -> torch.Tensor:
            acts = Actions(sow=best_sow.unsqueeze(0).expand(E, -1),
                           irrigation=irrigation.clamp(*bounds[0]).unsqueeze(0)
                           .expand(E, -1),
                           n_fert=nf.clamp(*bounds[1]).unsqueeze(0)
                           .expand(E, -1))
            fd = forcing
            if h < days:
                fd = Forcing(*(getattr(fd, k)[:, :h]
                               for k in FORCING_VARS))
                acts = Actions(*(getattr(acts, k)[:, :h]
                                 for k in ACTION_VARS))
            accumulate = SeasonAccumulator(E, max(1, -(-h // 365)),
                                    device=field.state.sw.device)
            rolled = model.rollout(field.repeat(E), fd, acts, observer=accumulate,
                                checkpoint_every=24 if h > 64 else 0)
            m = accumulate.result().unsqueeze(0)              # (1, E, H, M)
            r = (reward(m) if reward is not None
                 else metric_reward(m, weights, risk, alpha, margin))
            return -(r.mean() + bonus(rolled, fd, acts).mean())

        if refine_alg == "lbfgs":
            _quasi_newton(polish, [irrigation, nf], refine)
        else:
            opt = torch.optim.Adam([irrigation, nf], lr=refine_lr)
            for _ in range(refine):
                opt.zero_grad(set_to_none=True)
                loss = polish()
                if not torch.isfinite(loss):
                    break
                loss.backward()
                opt.step()
                with torch.no_grad():
                    irrigation.clamp_(*bounds[0])
                    nf.clamp_(*bounds[1])
        refined = torch.stack([irrigation.detach(), nf.detach()])
        r_new = score(refined.unsqueeze(1),
                      best_sow.unsqueeze(0))[0]
        if float(r_new) > best_r:
            best_r, best_mu = float(r_new), refined

    acts = Actions(sow=best_sow.unsqueeze(0),
                   irrigation=best_mu[0].clamp(*bounds[0]).unsqueeze(0),
                   n_fert=best_mu[1].clamp(*bounds[1]).unsqueeze(0))
    return ControlResult(acts, best_r, tuple(history))
