"""E8 — test-time environment adaptation ablation.

The truth gap is an environment shift the prior does not know: the truth
field's initial soil water is scaled up (a wetter prior season, unknown
irrigation history) and its site ctx is nonzero. Three arms on held-out
realizations:

- `frozen` : nominal initial state, blank ctx — the unadapted baseline
- `t2z`    : initial state + z0 + ctx-delta fit on the calibration obs,
             model frozen
- `optta`  : same fit plus a task term — adaptation aimed at the
             planner's reward, not only at prediction error

Scored by obs RMSE against the truth field's realizations and by MPPI
reward with vs without the adapted environment, answering both "does it
predict better" and "does it decide better".
"""

from __future__ import annotations

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.control import metric_reward, path_integral
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.experiments.embedding import _trace
from nasa_farm.experiments.run import register
from nasa_farm.experiments.scene import scene
from nasa_farm.experiments.specification import Specification
from nasa_farm.heads import SeasonAccumulator
from nasa_farm.optimize.adapt import adapt
from nasa_farm.optimize.targets import ObservedSeries, series, series_loss
from nasa_farm.state.field import Field

ENV_VARS = ("sw0", "swr")
"""Editable initial-state variables the environment fit may move."""


def _task_reward(model, field: Field, forcing, actions) -> torch.Tensor:
    accumulate = SeasonAccumulator(field.batch, max(1, forcing.days // 365))
    model.rollout(field, forcing, actions, observer=accumulate)
    return metric_reward(accumulate.result().unsqueeze(0))


def _obs_rmse(model, f_true: Field, f_pred: Field, scenario, specification: Specification) -> float:
    """Mean obs RMSE across held-out realizations, truth vs predicted env."""
    errors = []
    with torch.inference_mode():
        for i in scenario.held_indices():
            f_t = scenario.realization(i)
            ot, _ = _trace(model, f_true, f_t, scenario.actions, specification)
            op, _ = _trace(model, f_pred, f_t, scenario.actions, specification)
            errors.append(float((op - ot).pow(2).mean().sqrt()))
    return round(sum(errors) / len(errors), 6)


@register("adapt")
def run(specification: Specification) -> dict:
    torch.manual_seed(specification.seed)
    scenario = scene(specification)
    c, k = len(specification.observed_channels), specification.latent_dimension
    generation = torch.Generator().manual_seed(specification.seed + 17)

    model = WorldModel(
        scenario.crops,
        residual=True,
        latent_dimension=k,
        context_dimension=c,
        seed=specification.seed,
    )
    f0, a0 = scenario.realization(0), scenario.actions

    # truth: wetter initial profile + nonzero site ctx — an environment
    # the prior knows nothing about
    st_t = scenario.field.state.replace(
        sw=scenario.field.state.sw * 1.25,
        z=torch.zeros(1, k, dtype=DTYPE),
        zh=torch.zeros(1, model.latent.hidden_dimension, dtype=DTYPE),
    )
    truth_f = Field(
        state=st_t,
        soil=scenario.field.soil,
        latitude=scenario.field.latitude,
        context=torch.ones(1, c, dtype=DTYPE),
    )

    with torch.inference_mode():
        calibration_observed, _ = _trace(model, truth_f, f0, a0, specification)
        calibration_observed = calibration_observed * (
            1.0
            + specification.noise
            * torch.randn(
                calibration_observed.shape, generator=generation, dtype=calibration_observed.dtype
            )
        )
    target = ObservedSeries(values=calibration_observed[0].detach().clone())

    st_b = scenario.field.state.replace(
        z=torch.zeros(1, k, dtype=DTYPE),
        zh=torch.zeros(1, model.latent.hidden_dimension, dtype=DTYPE),
    )
    blank = Field(
        state=st_b,
        soil=scenario.field.soil,
        latitude=scenario.field.latitude,
        context=torch.zeros(1, c, dtype=DTYPE),
    )

    def obs_loss(f: Field) -> torch.Tensor:
        predicted = series(
            model, f, f0, a0, specification.observed_channels, every=specification.cadence
        )
        return series_loss(predicted, target)

    ev_frozen = _obs_rmse(model, truth_f, blank, scenario, specification)

    keywords = {
        "z": True,
        "context": True,
        "state_vars": ENV_VARS,
        "epochs": specification.epochs,
        "rate": specification.rate,
    }
    f_t2z, rep_t2z = adapt(model, blank, f0, a0, loss_fn=obs_loss, **keywords)
    ev_t2z = _obs_rmse(model, truth_f, f_t2z, scenario, specification)

    f_op, rep_op = adapt(
        model, blank, f0, a0, loss_fn=obs_loss, task=_task_reward, task_w=0.1, **keywords
    )
    ev_op = _obs_rmse(model, truth_f, f_op, scenario, specification)

    r_base = path_integral(
        model, blank, f0, scenario.actions, iterations=3, samples=16, seed=specification.seed
    ).reward
    r_ad = path_integral(
        model, f_op, f0, scenario.actions, iterations=3, samples=16, seed=specification.seed
    ).reward

    return {
        "obs_rmse": {"frozen": ev_frozen, "t2z": ev_t2z, "optta": ev_op},
        "t2z_report": rep_t2z,
        "optta_report": rep_op,
        "mppi_reward": {"frozen": round(r_base, 4), "adapted": round(r_ad, 4)},
    }
