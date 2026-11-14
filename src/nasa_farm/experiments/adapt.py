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
from nasa_farm.control import metric_reward, mppi
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.exp.jepa import _trace
from nasa_farm.exp.run import register
from nasa_farm.exp.scene import scene
from nasa_farm.exp.spec import Spec
from nasa_farm.heads import SeasonAccumulator
from nasa_farm.optimize.adapt import adapt
from nasa_farm.optimize.targets import ObservedSeries, obs_series, series_loss
from nasa_farm.state.field import Field

ENV_VARS = ("sw0", "swr")
"""Editable initial-state variables the environment fit may move."""


def _task_reward(model, field: Field, forcing, actions) -> torch.Tensor:
    acc = SeasonAccumulator(field.batch, max(1, forcing.days // 365))
    model.rollout(field, forcing, actions, observer=acc)
    return metric_reward(acc.result().unsqueeze(0))


def _obs_rmse(model, f_true: Field, f_pred: Field, sc, spec: Spec) -> float:
    """Mean obs RMSE across held-out realizations, truth vs predicted env."""
    errs = []
    with torch.inference_mode():
        for i in range(sc.forcing_t.batch):
            f_t = sc.realization(i)
            ot, _ = _trace(model, f_true, f_t, sc.actions, spec)
            op, _ = _trace(model, f_pred, f_t, sc.actions, spec)
            errs.append(float((op - ot).pow(2).mean().sqrt()))
    return round(sum(errs) / len(errs), 6)


@register("adapt")
def run(spec: Spec) -> dict:
    torch.manual_seed(spec.seed)
    sc = scene(spec)
    c, k = len(spec.obs_channels), spec.latent_dim
    gen = torch.Generator().manual_seed(spec.seed + 17)

    model = WorldModel(sc.crops, residual=True, latent_dim=k,
                       ctx_dim=c, seed=spec.seed)
    f0, a0 = sc.realization(0), sc.actions

    # truth: wetter initial profile + nonzero site ctx — an environment
    # the prior knows nothing about
    st_t = sc.field.state.replace(
        sw=sc.field.state.sw * 1.25,
        z=torch.zeros(1, k, dtype=DTYPE),
        zh=torch.zeros(1, model.latent.h_dim, dtype=DTYPE))
    truth_f = Field(state=st_t, soil=sc.field.soil, lat=sc.field.lat,
                    ctx=torch.ones(1, c, dtype=DTYPE))

    with torch.inference_mode():
        obs_cal, _ = _trace(model, truth_f, f0, a0, spec)
        obs_cal = obs_cal * (1.0 + spec.noise * torch.randn(
            obs_cal.shape, generator=gen, dtype=obs_cal.dtype))
    tgt = ObservedSeries(values=obs_cal[0].detach().clone())

    st_b = sc.field.state.replace(
        z=torch.zeros(1, k, dtype=DTYPE),
        zh=torch.zeros(1, model.latent.h_dim, dtype=DTYPE))
    blank = Field(state=st_b, soil=sc.field.soil, lat=sc.field.lat,
                  ctx=torch.zeros(1, c, dtype=DTYPE))

    def obs_loss(f: Field) -> torch.Tensor:
        pred = obs_series(model, f, f0, a0, spec.obs_channels,
                          every=spec.obs_every)
        return series_loss(pred, tgt)

    ev_frozen = _obs_rmse(model, truth_f, blank, sc, spec)

    kw = dict(z=True, ctx=True, state_vars=ENV_VARS,
              epochs=spec.epochs, lr=spec.lr)
    f_t2z, rep_t2z = adapt(model, blank, f0, a0, loss_fn=obs_loss, **kw)
    ev_t2z = _obs_rmse(model, truth_f, f_t2z, sc, spec)

    f_op, rep_op = adapt(model, blank, f0, a0, loss_fn=obs_loss,
                         task=_task_reward, task_w=0.1, **kw)
    ev_op = _obs_rmse(model, truth_f, f_op, sc, spec)

    r_base = mppi(model, blank, f0, sc.actions, iters=3, samples=16,
                  seed=spec.seed).reward
    r_ad = mppi(model, f_op, f0, sc.actions, iters=3, samples=16,
                seed=spec.seed).reward

    return {"obs_rmse": {"frozen": ev_frozen,
                         "t2z": ev_t2z,
                         "optta": ev_op},
            "t2z_report": rep_t2z,
            "optta_report": rep_op,
            "mppi_reward": {"frozen": round(r_base, 4),
                            "adapted": round(r_ad, 4)}}
