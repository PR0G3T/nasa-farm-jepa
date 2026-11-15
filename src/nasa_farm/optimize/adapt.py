"""Test-time environment adaptation — T2-z + OpTTA on the world model.

A trained model keeps its weights frozen at deployment; the *environment*
is what changes (new site, new season, regime shift). The free variables
are therefore the ones that parameterize the environment: the initial
latent z0 and an additive site-context delta. `adapt` fits exactly those
on a short observed trajectory — a few observations of a single season
are enough to pull the environment latent into the right basin, while
the shared dynamics stay untouched.

`task_w` > 0 adds the OpTTA twist: the adaptation objective is not pure
prediction error but prediction error minus a task reward, so the
environment latent settles where the planner's objective is actually
optimized — adaptation for decision-making, not just fit.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.optimize.targets import (
    ObservedTrajectory,
    season_metrics,
    trajectory_loss,
)
from nasa_farm.optimize.trainer import fit, leaf
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


def adapt(
    model,
    field: Field,
    forcing: Forcing,
    actions: Actions,
    observed: ObservedTrajectory | None = None,
    *,
    loss_fn: Callable[[Field], torch.Tensor] | None = None,
    z: bool = True,
    context: bool = True,
    state_vars: tuple[str, ...] = (),
    task: Callable | None = None,
    task_w: float = 0.0,
    epochs: int = 40,
    rate: float = 0.05,
) -> tuple[Field, dict]:
    """Fit the environment on observed data, model frozen.

    obs: ObservedTrajectory scored by trajectory_loss (default path).
    loss_fn: custom field → scalar obs loss (e.g. a satellite-series χ²
    built on obs_series) — overrides `obs` when both are given.
    z: adapt the initial latent state. ctx: adapt an additive correction
    to field.ctx. state_vars: editable initial-state variable names
    (assim.STATE_VARS) to fit jointly — the environment is not only
    latent. task: optional (model, field, forcing, actions) → reward;
    with task_w>0 the objective adds −task_w·task (OpTTA: adapt where the
    decision objective is optimized, not only where prediction improves).

    Returns the adapted field (model untouched) plus a fit report.
    """
    from nasa_farm.assimilation.kalman import _get_variable, _put_variables

    if loss_fn is None:
        if observed is None:
            raise ValueError("adapt needs `obs` or a custom `loss_fn`")

        def loss_fn(f):
            return trajectory_loss(season_metrics(model, f, forcing, actions), observed)

    leaves: dict[str, torch.Tensor] = {}
    z0 = None
    if z:
        dim = (
            field.state.z.shape[-1]
            if field.state.z is not None
            else (model.latent.dim if model.latent is not None else 0)
        )
        if dim == 0:
            raise ValueError("adapt(z=True) needs a latent state or model latent")
        base_z = (
            field.state.z
            if field.state.z is not None
            else torch.zeros(field.batch, dim, dtype=DTYPE)
        )
        z0 = leaf(base_z)
        leaves["z0"] = z0
    dctx = None
    if context and field.context is not None:
        dctx = leaf(torch.zeros_like(field.context))
        leaves["dctx"] = dctx
    x0 = None
    if state_vars:
        x0 = leaf(torch.stack([_get_variable(field.state, n) for n in state_vars], -1))
        leaves["x0"] = x0
    if not leaves:
        raise ValueError("nothing to adapt: no z state, no ctx, no state_vars")

    def loss() -> torch.Tensor:
        updated = field.state
        if z0 is not None:
            updated = updated.replace(z=z0)
        if x0 is not None:
            updated = _put_variables(updated, state_vars, x0)
        f = Field(
            state=updated,
            soil=field.soil,
            latitude=field.latitude,
            elevation=field.elevation,
            context=field.context + dctx if dctx is not None else field.context,
        )
        value = loss_fn(f)
        if task is not None and task_w:
            value = value - task_w * task(model, f, forcing, actions).mean()
        return value

    result = fit(loss, leaves, epochs=epochs, rate=rate)
    updated = field.state
    if z0 is not None:
        updated = updated.replace(z=leaves["z0"].detach())
    if x0 is not None:
        updated = _put_variables(updated, state_vars, leaves["x0"].detach())
    output = Field(
        state=updated,
        soil=field.soil,
        latitude=field.latitude,
        elevation=field.elevation,
        context=(field.context + leaves["dctx"].detach()) if dctx is not None else field.context,
    )
    rep = {
        "best_loss": round(result.best_loss, 8),
        "epochs_run": result.epochs_run,
        "z0_shift": (
            round(float(leaves["z0"].detach().sub(base_z).norm()), 4) if z0 is not None else None
        ),
        "ctx_shift": (
            round(float(leaves["dctx"].detach().norm()), 4) if dctx is not None else None
        ),
        "handles": sorted(leaves),
    }
    if x0 is not None:
        base_x = torch.stack([_get_variable(field.state, n) for n in state_vars], -1)
        rep["state_shift"] = round(float((leaves["x0"].detach() - base_x).norm()), 4)
    return output, rep
