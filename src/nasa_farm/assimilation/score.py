"""Ensemble Score Filter — Bao, Zhang & Zhang (2023), arXiv:2309.00983.

The analysis draws new members by integrating a backward diffusion SDE in
pseudo-time tau in [0,1] with a *training-free* Monte-Carlo score. The
forecast ensemble plays the role of the reference sample set:

  forward marginals   q(z_tau | x_j) = N(alpha_tau x_j, beta_tau^2 I)
  alpha_tau  = 1 - tau (1 - eps_a)
  beta_tau^2 = eps_b + tau (1 - eps_b)
  prior score S(z,tau) = sum_j w_j (alpha x_j - z)/beta^2 ,
      w_j softmax over -||z - alpha x_j||^2 / (2 beta^2)
  posterior score      S + h(tau) * grad_z log p(y|z),  h(tau) = 1 - tau
  backward SDE (Euler-Maruyama, tau: 1 -> 0):
      z <- z - [b z - s^2 S] dtau + s dW,
      b(t) = alpha'/alpha,  s^2(t) = beta2' - 2 beta^2 alpha'/alpha

The likelihood gradient is exact when `channels` is given: an autograd
jacobian of the observation operator at the diffused point. Without
channel names it falls back to the ensemble-linearized operator
H = cov(Y,X) cov(X)^{-1} — the standard reduced-rank surrogate.
"""

from __future__ import annotations

import math

import torch

from nasa_farm.assimilation.kalman import STATE_VARS, _get_variable, _put_variables
from nasa_farm.observe import observables
from nasa_farm.state.field import FieldState

_EPS_ALPHA = 0.05
_EPS_BETA = 0.01


def _linearize(x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Ensemble-linearized obs operator H: (B,C),(B,V) -> (C,V)."""
    xa = x - x.mean(0)
    ya = y_pred - y_pred.mean(0)
    cxx = xa.t() @ xa / (x.shape[0] - 1)
    cyx = ya.t() @ xa / (x.shape[0] - 1)
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return torch.linalg.solve(cxx + 1e-9 * eye, cyx.t()).t()


def _likelihood_grad(z: torch.Tensor, state: FieldState,
                     names: tuple[str, ...], channels: tuple[str, ...],
                     y: torch.Tensor, r_inv: torch.Tensor,
                     h_lin: torch.Tensor, y_mean: torch.Tensor,
                     x_mean: torch.Tensor) -> torch.Tensor:
    """grad_z log N(y | h(z), R) — exact autograd when channels given."""
    if channels is None:
        return (y - y_mean - (z - x_mean) @ h_lin.t()) @ r_inv @ h_lin
    with torch.inference_mode(False):
        zz = z.detach().clone()

        def h_one(xi):
            st = _put_variables(state, names, xi.unsqueeze(0))
            return observables(st, channels)[0]

        h_z, jac = torch.func.vmap(
            torch.func.value_and_grad(h_one))(zz)
    return torch.einsum("bcv,bv->bv", jac, (y - h_z) @ r_inv)


def score_update(state: FieldState, observed_prediction: torch.Tensor,
                 observed: torch.Tensor, r: torch.Tensor | float,
                 names: tuple[str, ...] = STATE_VARS,
                 channels: tuple[str, ...] | None = None,
                 steps: int = 40, eps_alpha: float = _EPS_ALPHA,
                 eps_beta: float = _EPS_BETA,
                 perturb: bool = True, generation: torch.Generator | None = None,
                 ) -> FieldState:
    """EnSF analysis: resample the posterior by backward-SDE paths.

    state/obs_pred: (B,) ensemble and (B, C) predicted observations;
    observed: (C,); r: obs-error variance (scalar, (C,) diag or (C,C)).
    `perturb` is accepted for interface symmetry with the EnKF — EnSF is
    intrinsically stochastic (its members are drawn, not shifted).
    """
    x = torch.stack([_get_variable(state, n) for n in names], dim=-1)
    b, v = x.shape
    r_t = torch.as_tensor(r, dtype=x.dtype, device=x.device)
    if r_t.ndim == 2:
        r_inv = torch.linalg.inv(r_t)
    else:
        r_inv = torch.diag(1.0 / r_t.expand(observed.numel()))
    h_lin = _linearize(x, observed_prediction)
    z = torch.randn(b, v, generator=generation, dtype=x.dtype, device=x.device)
    dtau = 1.0 / steps
    for k in range(steps, 0, -1):
        tau = k * dtau
        alpha = 1.0 - tau * (1.0 - eps_alpha)
        beta2 = eps_beta + tau * (1.0 - eps_beta)
        drift = -(1.0 - eps_alpha) / alpha
        diff2 = (1.0 - eps_beta) + 2.0 * beta2 * (1.0 - eps_alpha) / alpha

        d = z.unsqueeze(1) - alpha * x.unsqueeze(0)          # (B, B, V)
        w = torch.softmax(-0.5 * d.pow(2).sum(-1) / beta2, dim=1)
        mu = w @ (alpha * x)
        s_prior = (mu - z) / beta2
        s_like = _likelihood_grad(
            z, state, names, channels, observed, r_inv,
            h_lin, observed_prediction.mean(0), x.mean(0))
        s_post = s_prior + (1.0 - tau) * s_like
        noise = torch.randn(z.shape, generator=generation,
                            dtype=z.dtype, device=z.device)
        z = z - (drift * z - diff2 * s_post) * dtau \
            + math.sqrt(diff2 * dtau) * noise
    z = torch.nan_to_num(z).clamp(min=0.0)
    return _put_variables(state, names, z)
