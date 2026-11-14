"""Ensemble score filtering — transport-based nonlinear analysis.

Members first take the deterministic EnKF mean shift, then flow along the
kernel-density score of their own ensemble: each member is attracted toward
high-density regions of the updated cloud. A deterministic analog of EnSF
valid for the unimodal posteriors encountered in column-scale agronomy,
requiring no trained score network.
"""

from __future__ import annotations

import torch

from nasa_farm.assim.kalman import STATE_VARS, _get_var, _put_vars
from nasa_farm.state.field import FieldState


def ensf_update(state: FieldState, obs_pred: torch.Tensor,
                obs: torch.Tensor, r: torch.Tensor | float,
                var_names: tuple[str, ...] = STATE_VARS,
                steps: int = 30, lr: float = 0.05,
                bw: float | None = None) -> FieldState:
    """Kernel-score transport analysis.

    state: (B,) ensemble; obs_pred: (B, C) predicted observations;
    obs: (C,) measurement; r: observation-error variance.
    """
    x = torch.stack([_get_var(state, n) for n in var_names], dim=-1)
    b = state.batch
    r_t = torch.as_tensor(r, dtype=x.dtype, device=x.device)
    xm, ym = x.mean(0), obs_pred.mean(0)
    xa, ya = x - xm, obs_pred - ym
    cxy = xa.t() @ ya / (b - 1)
    cyy = ya.t() @ ya / (b - 1) + torch.diag(r_t.expand(obs.numel()))
    k = torch.linalg.solve(cyy, cxy.t()).t()
    xn = x + (obs - obs_pred) @ k.t()                # deterministic EnKF shift

    scale = xn.std(0).clamp(min=1e-6)
    h = (bw if bw is not None else 1.0) * scale * b ** (-1.0 / (x.shape[1] + 4))
    for _ in range(steps):
        d = (xn.unsqueeze(1) - xn.unsqueeze(0)) / h
        dens = torch.exp(-0.5 * (d ** 2).sum(-1))    # (B, B) kernel affinity
        den = dens.sum(1, keepdim=True).clamp(min=1e-9)
        pull = (dens / den).unsqueeze(-1) * d * h    # KDE-score transport
        xn = xn + lr * (pull.sum(1) - pull.sum(0))
    xn = torch.nan_to_num(xn).clamp(min=0.0)
    return _put_vars(state, var_names, xn)
