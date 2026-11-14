"""Counterfactual rollouts — prediction-error abduction.

`obs + [pred(a') - pred(a)]`: the factual residual e_t = obs_t - pred_t is
treated as a persistent unmodeled factor and transferred to the alternative
action rollout. Action x factor interactions only enter at O(t^2), so the
one-subtraction counterfactual is exact to first order — no re-identification
of hidden state is needed for short-horizon what-ifs.
"""

from __future__ import annotations

import torch

from nasa_farm.observe.trace import ObservationTrace
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing


def abduct(observed: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """Persistent-factor offset series: forward-filled (obs - pred).

    obs/pred (B, N, C); NaN = unobserved. Before the first observed slot a
    channel carries no information, so its offset stays 0.
    """
    n = observed.shape[1]
    mask = torch.isfinite(observed)
    index = torch.arange(n, device=observed.device).view(1, n, 1).expand_as(mask)
    seen = torch.where(mask, index, -1).cummax(1).values          # (B, N, C)
    off = torch.gather(torch.nan_to_num(observed - predicted.to(observed.dtype)), 1,
                       seen.clamp(min=0))
    return off * (seen >= 0)


def counterfactual(model, field: Field, forcing: Forcing,
                   act_fact: Actions, act_alt: Actions,
                   observed: torch.Tensor, channels: tuple[str, ...],
                   every: int = 5) -> tuple[torch.Tensor, dict]:
    """Counterfactual channel series under alternative actions.

    obs: (B, N, C) factual observations (NaN allowed) at the trace cadence.
    Returns (cf, info): cf = pred_alt + abduct(obs, pred_fact).
    """
    tr_f = ObservationTrace(forcing.days, channels, every)
    model.rollout(field, forcing, act_fact, observer=tr_f)
    tr_a = ObservationTrace(forcing.days, channels, every)
    model.rollout(field, forcing, act_alt, observer=tr_a)
    pred_f, pred_a = tr_f.result(), tr_a.result()
    off = abduct(observed, pred_f)
    return pred_a + off, {"offset": off, "pred_fact": pred_f,
                          "pred_alt": pred_a}
