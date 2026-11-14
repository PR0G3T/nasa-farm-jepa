"""Gradient-guided posterior over the initial state — physics-exact ULA.

Unadjusted Langevin over the editable initial-condition channels: the
target density is

    log p(x0 | y) = −χ²(obs_series(x0), y) − prior(x0)

where obs_series rolls the full differentiable model and applies the exact
observation operators — no learned surrogate, no linearization. Each step
adds sqrt(2·lr·temp) noise; post-burn-in iterates form a posterior
ensemble of initial states, an alternative to filter-style analysis when
the obs window is short and the dynamics are strongly nonlinear.
"""

from __future__ import annotations

import torch

from nasa_farm.assim.kalman import STATE_VARS, _get_var, _put_vars
from nasa_farm.constants import DTYPE
from nasa_farm.optimize.targets import ObservedSeries, obs_series, series_loss
from nasa_farm.state.field import Field


def ula_posterior(model, field: Field, forcing, actions,
                  obs: ObservedSeries,
                  channels: tuple[str, ...], every: int = 5,
                  var_names: tuple[str, ...] = STATE_VARS,
                  prior_sd: float = 0.2,
                  steps: int = 60, lr: float = 1e-3, temp: float = 1.0,
                  burn: int = 20, n_samples: int = 8,
                  seed: int = 0) -> list[Field]:
    """Langevin posterior on initial-state channels → field ensemble.

    x0 is parameterized in log-space (positive channels stay positive);
    prior is N(log x0_init, prior_sd). Returns `n_samples` fields spaced
    over the post-burn-in chain — push each through `rollout`/`assimilate`
    for posterior trajectories.
    """
    gen = torch.Generator().manual_seed(seed)
    x0 = torch.stack([_get_var(field.state, n)[0] for n in var_names])
    logx = torch.log(x0.clamp(min=1e-6)).detach().requires_grad_(True)
    logx0 = logx.detach().clone()
    chain: list[torch.Tensor] = []

    for it in range(steps):
        st = _put_vars(field.state, var_names, logx.exp().unsqueeze(0))
        f = Field(state=st, soil=field.soil, lat=field.lat,
                  elev=field.elev, ctx=field.ctx)
        pred = obs_series(model, f, forcing, actions, channels, every)
        logp = -series_loss(pred, obs) \
            - 0.5 * ((logx - logx0) / prior_sd).pow(2).sum()
        g = torch.autograd.grad(logp, logx)[0]
        with torch.no_grad():
            logx.add_(lr * g + (2.0 * lr * temp) ** 0.5
                      * torch.randn(logx.shape, generator=gen, dtype=DTYPE))
        if it >= burn:
            chain.append(logx.detach().clone())

    if not chain:
        chain = [logx.detach().clone()]
    idx = torch.linspace(0, len(chain) - 1, n_samples).long()
    out = []
    for i in idx.tolist():
        st = _put_vars(field.state, var_names, chain[i].exp().unsqueeze(0))
        out.append(Field(state=st, soil=field.soil, lat=field.lat,
                         elev=field.elev, ctx=field.ctx))
    return out
