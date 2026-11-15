"""Observation-schedule design — Bayesian experimental design for the EnKF.

For each candidate (day, channel), the expected information gain is the
log-ratio of prior predictive spread to instrument noise,
0.5·log det(C_yy + R) − 0.5·log det R, evaluated on one rollout of the
state ensemble. Greedy ranking then picks the measurements that shrink
posterior spread most per acquisition.
"""

from __future__ import annotations

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.observe import observables
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


def eigen(observed_prediction: torch.Tensor, r: torch.Tensor | float) -> torch.Tensor:
    """Expected information gain of one measurement channel.

    obs_pred: (B,) ensemble of predicted observables; r: error variance.
    gain = 0.5·log(1 + C_yy/R) for a scalar channel — monotone in ensemble
    spread, so days where members disagree rank highest.
    """
    r_t = torch.as_tensor(r, dtype=observed_prediction.dtype, device=observed_prediction.device)
    cyy = observed_prediction.var(unbiased=observed_prediction.numel() > 1)
    return 0.5 * torch.log1p(cyy / r_t.clamp(min=1e-12))


def design_schedule(
    model,
    field: Field,
    forcing: Forcing,
    actions: Actions,
    days: tuple[int, ...],
    channels: tuple[str, ...],
    r: torch.Tensor | float = 1.0,
    n_pick: int | None = None,
) -> list[tuple[int, str, float]]:
    """Rank (day, channel) candidates by expected information gain.

    One ensemble rollout caches states at candidate days; gains are
    evaluated channel-wise — cheap because each channel's gain is scalar.
    Returns [(day, channel, gain)] sorted descending.
    """
    candidate = set(days)
    snapshot: dict[int, object] = {}

    def cache(t: int, state, fluxes) -> None:
        if t in candidate:
            snapshot[t] = state

    with torch.inference_mode():
        model.rollout(field, forcing, actions, observer=cache)

    r_t = torch.as_tensor(r)
    ranked = []
    for t in days:
        if t not in snapshot:
            continue
        y = observables(snapshot[t], channels)
        for i, c in enumerate(channels):
            g = float(eigen(y[:, i], r_t if r_t.dim() == 0 else r_t[i]))
            ranked.append((t, c, g))
    ranked.sort(key=lambda x: -x[2])
    return ranked[:n_pick] if n_pick is not None else ranked


def plan_schedule(
    model,
    field: Field,
    forcing: Forcing,
    actions: Actions,
    days: tuple[int, ...],
    channels: tuple[str, ...],
    r: torch.Tensor | float = 1.0,
    n_pick: int = 5,
    tau: float = 7.0,
) -> list[tuple[int, str, float]]:
    """Ordered acquisition calendar — greedy EIG with redundancy discount.

    `design_schedule` ranks each (day, channel) independently; satellite
    tasking needs a *sequence*: after a day is tasked, nearby days carry
    correlated information. This applies a temporal-saturation kernel —
    surviving candidate gains are attenuated by exp(−|t−t*|/τ) per picked
    day — then re-greeds, so the calendar front-loads genuinely new
    information instead of clustering at the noisiest window."""
    ranked = design_schedule(model, field, forcing, actions, days, channels, r)
    pool = [[t, c, g] for t, c, g in ranked]
    output: list[tuple[int, str, float]] = []
    while pool and len(output) < n_pick:
        t, c, g = max(pool, key=lambda x: x[2])
        output.append((t, c, g))
        pool.remove([t, c, g])
        for row in pool:
            row[2] *= (
                1.0 - 0.9 * torch.exp(torch.tensor(-abs(row[0] - t) / tau, dtype=DTYPE)).item()
            )
    return output


def metric_jacobian(
    model,
    field: Field,
    forcing: Forcing,
    actions: Actions,
    crop_index: int,
    traits: tuple[str, ...],
    channels: tuple[str, ...],
) -> torch.Tensor:
    """J[h·c, p] = ∂(seasonal metric c of year h)/∂θ_p — exact forward-mode
    Jacobian of the seasonal-metric map w.r.t. the trait cells. One
    jacfwd costs len(traits) forward-mode passes; the singular spectrum of
    J·R⁻¹ is the formal identifiability diagnostic of the channel set."""
    from nasa_farm.heads import METRICS
    from nasa_farm.optimize.targets import season_metrics
    from nasa_farm.state.crops import INDEX

    base = model.crop_table.detach().clone()
    cols = [INDEX[t] for t in traits]
    idx = [METRICS.index(c) for c in channels]

    def fn(theta: torch.Tensor) -> torch.Tensor:
        table = base.clone()
        table[crop_index, cols] = theta
        model.crop_table = table
        return season_metrics(model, field, forcing, actions)[0][:, idx].flatten()

    j = torch.func.jacfwd(fn)(base[crop_index, cols].detach().clone())
    model.crop_table = base
    return j


def series_jacobian(
    model,
    field: Field,
    forcing: Forcing,
    actions: Actions,
    crop_index: int,
    traits: tuple[str, ...],
    channels: tuple[str, ...],
    every: int = 10,
) -> torch.Tensor:
    """J[t, c, p] = ∂obs_c(t)/∂θ_p — exact forward-mode Jacobian of the
    predicted observation series w.r.t. the trait cells, at cadence
    `every`. Rows are (day, channel) pairs; slice [:, c, :] per channel."""
    from nasa_farm.optimize.targets import series
    from nasa_farm.state.crops import INDEX

    base = model.crop_table.detach().clone()
    cols = [INDEX[t] for t in traits]

    def fn(theta: torch.Tensor) -> torch.Tensor:
        table = base.clone()
        table[crop_index, cols] = theta
        model.crop_table = table
        return series(model, field, forcing, actions, channels, every=every)[0].flatten()

    j = torch.func.jacfwd(fn)(base[crop_index, cols].detach().clone())
    model.crop_table = base
    n = j.shape[0] // len(channels)
    return j.reshape(n, len(channels), len(traits))


def select_channels(
    jacobian: torch.Tensor, scale: torch.Tensor, k: int
) -> tuple[list[int], torch.Tensor]:
    """Greedy E-optimal design over channel rows of a series Jacobian.

    Each step adds the channel maximizing the smallest eigenvalue of the
    accumulated information matrix JᵀR⁻¹J — the design criterion aimed at
    the weakest trait direction, which is exactly the direction
    equifinality hides in. `scale` whitens rows (per-channel obs scale)."""
    p = jacobian.shape[2]
    info = torch.zeros(p, p, dtype=jacobian.dtype)
    picked, rest = [], list(range(jacobian.shape[1]))

    def whiten(c: int) -> torch.Tensor:
        j = jacobian[:, c, :] / scale[c].clamp(min=1e-9)
        return j.t() @ j

    for _ in range(min(k, jacobian.shape[1])):
        c = max(rest, key=lambda c: float(torch.linalg.eigvalsh(info + whiten(c))[0]))
        picked.append(c)
        rest.remove(c)
        info = info + whiten(c)
    return picked, info


def fisher_design(
    model,
    field: Field,
    forcing: Forcing,
    actions: Actions,
    traits: tuple[str, ...],
    days: tuple[int, ...],
    channels: tuple[str, ...],
    r: float = 1.0,
    eps: float = 0.05,
    n_pick: int | None = None,
) -> list[tuple[int, str, float]]:
    """Trait-oriented design: rank (day, channel) by Fisher information.

    One rollout per perturbed trait gives the sensitivity Jacobian
    J[t, c, p] = d obs / d trait (finite difference on eps·trait); the
    marginal gain of a candidate is log1p of its Fisher contribution —
    the measurement that most reduces trait equifinality ranks first.
    Costs len(traits)+1 rollouts total.
    """
    import copy

    from nasa_farm.state.crops import INDEX

    candidate = set(days)
    snapshot: dict[int, object] = {}

    def cache(t: int, state, fluxes) -> None:
        if t in candidate:
            snapshot[t] = state

    def obs_at(m) -> dict[int, torch.Tensor]:
        snapshot.clear()
        with torch.inference_mode():
            m.rollout(field, forcing, actions, observer=cache)
        return {t: observables(snapshot[t], channels) for t in days if t in snapshot}

    base = obs_at(model)
    j = {t: torch.zeros(len(channels), len(traits)) for t in base}
    for p_i, name in enumerate(traits):
        m2 = copy.deepcopy(model)
        with torch.no_grad():
            column = INDEX[name]
            m2.crop_table[:, column] = m2.crop_table[:, column] * (1.0 + eps)
        pert = obs_at(m2)
        for t in j:
            j[t][:, p_i] = (pert[t] - base[t]).abs().mean(0) / (
                eps * float(model.crop_table[:, column].abs().mean()) + 1e-9
            )
    ranked = []
    for t, jt in j.items():
        for c_i, c in enumerate(channels):
            gain = float(torch.log1p((jt[c_i] ** 2).sum() / r))
            ranked.append((t, c, round(gain, 6)))
    ranked.sort(key=lambda x: -x[2])
    return ranked[:n_pick] if n_pick is not None else ranked
