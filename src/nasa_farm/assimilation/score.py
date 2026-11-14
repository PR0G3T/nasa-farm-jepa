"""Ensemble Score Filter — Bao, Zhang & Zhang (2023), arXiv:2309.00983.

Each analysis member is resampled by integrating a backward diffusion SDE
in pseudo-time tau in [0,1] with a *training-free* Monte-Carlo score
(Eq. 21-22, minibatch size one: each path carries its designated forecast
member as reference set — the choice used in the reference code):

  forward marginals   q(z_tau | x_j) = N(alpha_tau x_j, beta_tau^2 I)
  alpha_tau  = 1 - tau (1 - eps_a)      (Eq. 20)
  beta_tau^2 = eps_b + tau (1 - eps_b)  (Eq. 20)
  prior score S(z,tau) = (alpha_tau x_j - z) / beta_tau^2
  posterior score      S + h(tau) * grad_z log p(y|z),  h(tau) = 1 - tau
  backward SDE (Euler-Maruyama, tau: 1 -> 0, Eq. 23):
      z <- z - [b z - s^2 S] dtau + s dW
      b(t) = alpha'/alpha,  s^2(t) = beta2' - 2 beta^2 alpha'/alpha

The likelihood gradient is exact when `channels` is given: an autograd
jacobian of the observation operator at the diffused point (the paper's
"via automatic differentiation" option). Without channel names it falls
back to the ensemble-linearized operator H = cov(Y,X) cov(X)^{-1}.

Note the paper's own caveat: the damped-likelihood posterior score is not
the exact posterior filtering score — EnSF is an approximate sampler,
most accurate for informative observations.
"""

from __future__ import annotations

import math

import torch

from nasa_farm.assimilation.kalman import STATE_VARS, _get_variable, _put_variables
from nasa_farm.observe import observables
from nasa_farm.state.field import FieldState

_EPS_ALPHA = 0.05
_EPS_BETA = 0.01
_CLIP = 1000.0


def _linearize(x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Ensemble-linearized obs operator H: (B,V),(B,C) -> (C,V)."""
    xa = x - x.mean(0)
    ya = y_pred - y_pred.mean(0)
    cxx = xa.t() @ xa / (x.shape[0] - 1)
    cyx = ya.t() @ xa / (x.shape[0] - 1)
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return torch.linalg.solve(cxx + 1e-9 * eye, cyx.t()).t()


def _likelihood_grad(z: torch.Tensor, state: FieldState,
                     names: tuple[str, ...], channels: tuple[str, ...] | None,
                     y: torch.Tensor, r_inv: torch.Tensor,
                     h_lin: torch.Tensor, y_mean: torch.Tensor,
                     x_mean: torch.Tensor) -> torch.Tensor:
    """grad_z log N(y | h(z), R) — exact autograd when channels given."""
    if channels is None:
        return (y - y_mean - (z - x_mean) @ h_lin.t()) @ r_inv @ h_lin
    with torch.inference_mode(False):
        zz = z.detach().clone()

        def h_batch(xb):
            updated = _put_variables(state, names, xb)
            return observables(updated, channels)

        full = torch.func.jacrev(h_batch)(zz)         # (B,C,B,V)
        jac = full.diagonal(dim1=0, dim2=2).permute(2, 0, 1)  # (B,C,V)
        h_z = h_batch(zz)
    return torch.einsum("bcv,bc->bv", jac, (y - h_z) @ r_inv)


def score_update(state: FieldState, observed_prediction: torch.Tensor,
                 observed: torch.Tensor, r: torch.Tensor | float,
                 names: tuple[str, ...] = STATE_VARS,
                 channels: tuple[str, ...] | None = None,
                 steps: int = 40, eps_alpha: float = _EPS_ALPHA,
                 eps_beta: float = _EPS_BETA,
                 perturb: bool = True, generation: torch.Generator | None = None,
                 ) -> FieldState:
    """EnSF analysis: resample the posterior by backward-SDE paths.

    state/observed_prediction: (B,) ensemble and (B, C) predicted
    observables; observed: (C,); r: obs-error variance (scalar, (C,)
    diagonal or (C, C) full). `perturb` is accepted for interface
    symmetry — EnSF is intrinsically stochastic.
    """
    x = torch.stack([_get_variable(state, n) for n in names], dim=-1)
    b, v = x.shape
    r_t = torch.as_tensor(r, dtype=x.dtype, device=x.device)
    if r_t.ndim == 2:
        r_inv = torch.linalg.inv(r_t)
    else:
        r_inv = torch.diag(1.0 / r_t.expand(observed.numel()))
    h_lin = _linearize(x, observed_prediction)
    y_mean, x_mean = observed_prediction.mean(0), x.mean(0)
    z = torch.randn(b, v, generator=generation, dtype=x.dtype,
                    device=x.device)
    z = (z - z.mean(0)) / z.std(0).clamp(min=1e-9)
    dtau = 1.0 / steps
    for k in range(steps, 0, -1):
        tau = k * dtau
        alpha = 1.0 - tau * (1.0 - eps_alpha)
        beta2 = eps_beta + tau * (1.0 - eps_beta)
        drift = -(1.0 - eps_alpha) / alpha
        diff2 = (1.0 - eps_beta) + 2.0 * beta2 * (1.0 - eps_alpha) / alpha
        s_prior = (alpha * x - z) / beta2
        s_like = _likelihood_grad(
            z, state, names, channels, observed, r_inv,
            h_lin, y_mean, x_mean).clamp(-_CLIP, _CLIP)
        s_post = s_prior + (1.0 - tau) * s_like
        noise = torch.randn(z.shape, generator=generation,
                            dtype=z.dtype, device=z.device)
        z = z - (drift * z - diff2 * s_post) * dtau \
            + math.sqrt(diff2 * dtau) * noise
    z = torch.nan_to_num(z).clamp(min=0.0)
    return _put_variables(state, names, z)
