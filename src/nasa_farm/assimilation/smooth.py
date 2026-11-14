"""Ensemble-subspace variational smoothing (LEVDA-flavored 4DEnVar).

A 4D smoother needs gradients through the whole window — exactly what a
differentiable surrogate provides. The unknowns (initial-state and trait
perturbations) are parameterized in the low-rank subspace spanned by a
prior ensemble's anomalies:

    x = x̄ + A·c / √(E−1),   c ∈ R^E  (mean-removed)

so the posterior search lives in ≪D dimensions, the prior regularizer is
simply ½‖c‖², and no adjoint or tangent-linear model is needed — Adam on
the rollout χ² does the integration in sequence space.
"""

from __future__ import annotations

import math

import torch

from nasa_farm.assimilation.kalman import (
    STATE_VARS, Observation, _get_variable, _put_variables,
)
from nasa_farm.constants import DTYPE
from nasa_farm.observe import observables
from nasa_farm.state.crops import INDEX
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


def _prior_state_ens(state: FieldState, names: tuple[str, ...],
                     ensemble: int, sigma: float,
                     generation: torch.Generator) -> torch.Tensor:
    """(E, V) prior draws: log-normal jitter of the editable initial state."""
    x0 = torch.stack([_get_variable(state, n) for n in names], -1)[0]
    j = 1.0 + sigma * torch.randn(ensemble, x0.numel(), generator=generation,
                                  dtype=DTYPE, device=x0.device)
    return (x0.unsqueeze(0) * j).clamp(min=0.0)


def ensemble_smooth(model, field: Field, forcing: Forcing, actions: Actions,
               schedule: list[Observation],
               names: tuple[str, ...] = STATE_VARS,
               free_params: tuple[str, ...] = (), crop_idx: int = 0,
               ensemble: int = 32, sigma: float = 0.05, epochs: int = 60,
               rate: float = 0.05, seed: int = 0) -> tuple[Field, dict]:
    """Joint state+trait smoother over the whole window.

    The prior ensemble jitters the editable initial state (and the listed
    trait cells log-normally); the analysis is the prior mean plus a
    coefficient vector in the anomaly subspace, fit by minimizing the
    schedule's obs χ² + ½‖c‖² through the differentiable rollout.
    Returns the smoothed field and a report (χ², recovered coefficients).
    """
    generation = torch.Generator().manual_seed(seed)
    schedule = sorted(schedule, key=lambda observation: observation.t)

    xs = _prior_state_ens(field.state, names, ensemble, sigma, generation)
    x_bar = xs.mean(0)
    ax = (xs - x_bar) / math.sqrt(ensemble - 1)                 # (E, V)

    p_cols = torch.tensor([INDEX[p] for p in free_params],
                          dtype=torch.long) if free_params else None
    if p_cols is not None:
        base_p = model.crop_table[crop_idx, p_cols]
        ps = base_p.log().unsqueeze(0) + sigma * torch.randn(
            ensemble, p_cols.numel(), generator=generation, dtype=DTYPE)
        p_bar = ps.mean(0)
        ap = (ps - p_bar) / math.sqrt(ensemble - 1)             # (E, P)
    else:
        p_bar = torch.zeros(0, dtype=DTYPE)
        ap = torch.zeros(ensemble, 0, dtype=DTYPE)

    c_x = torch.zeros(ensemble, dtype=DTYPE, requires_grad=True)
    c_p = torch.zeros(ensemble, dtype=DTYPE, requires_grad=True)
    leaves = [c_x] + ([c_p] if p_cols is not None else [])
    opt = torch.optim.Adam(leaves, lr=rate)
    base_tbl = model.crop_table.detach().clone()
    by_day = {observation.t: observation for observation in schedule}

    def chi2() -> torch.Tensor:
        x = x_bar + (c_x - c_x.mean()) @ ax
        if p_cols is not None:
            table = base_tbl.clone()
            table[crop_idx, p_cols] = (p_bar + (c_p - c_p.mean()) @ ap).exp()
            model.crop_table = table
        updated = _put_variables(field.state, names, x.unsqueeze(0))
        f = Field(state=updated, soil=field.soil, latitude=field.latitude,
                  elevation=field.elevation, context=field.context)
        total = torch.zeros((), dtype=DTYPE)
        with torch.enable_grad():
            for t in range(forcing.days):
                f, _ = model.step(f, actions.day(t), forcing.day(t), t)
                if t in by_day:
                    observation = by_day[t]
                    predicted = observables(f.state, observation.channels)[0]
                    r = (predicted - observation.values.to(predicted.dtype)) / \
                        torch.as_tensor(observation.std, dtype=predicted.dtype)
                    total = total + r.pow(2).sum()
        return total

    best, best_c, history = float("inf"), None, []
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = chi2() + 0.5 * (c_x.pow(2).sum() + c_p.pow(2).sum())
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(leaves, 1.0)
        opt.step()
        history.append(float(loss.detach()))
        if history[-1] < best:
            best, best_c = history[-1], (c_x.detach().clone(),
                                      c_p.detach().clone())

    model.crop_table = base_tbl
    if best_c is None:
        return field, {"chi2": float("nan"), "epochs_run": 0}
    cx, crop = best_c
    x_fin = x_bar + (cx - cx.mean()) @ ax
    if p_cols is not None:
        table = base_tbl.clone()
        table[crop_idx, p_cols] = (p_bar + (crop - crop.mean()) @ ap).exp()
        model.crop_table = table
    updated = _put_variables(field.state, names, x_fin.unsqueeze(0))
    output = Field(state=updated, soil=field.soil, latitude=field.latitude,
                elevation=field.elevation, context=field.context)
    fitted = ({p: float(model.crop_table[crop_idx, INDEX[p]])
               for p in free_params} if p_cols is not None else {})
    return output, {"chi2": round(best, 6), "epochs_run": len(history),
                 "c_state_norm": round(float(cx.norm()), 4),
                 "params": fitted}
