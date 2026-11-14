"""Symbolic distillation — read the learned residual back into equations.

Sparse regression over a fixed basis library (SINDy-style, orthogonal
matching pursuit) fits each residual output channel as a small closed-form
combination of interpretable features. The distilled terms can then be
reinjected into the physics core — the loop that turns missing physics into
explicit physics.
"""

from __future__ import annotations

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.residual import N_FEATURES

FEAT_NAMES: tuple[str, ...] = (
    "sw0", "swr", "n_min", "lai", "biomass", "gdd", "growing",
    "tmin", "tmax", "srad", "precip", "irrigation", "n_fert", "sow")


def basis(x: torch.Tensor) -> torch.Tensor:
    """(B, F) features → (B, L) library: x, x², tanh, pairwise products of
    the first-order terms, plus bias. L = 1 + 3F + F(F−1)/2."""
    f = x.shape[-1]
    cols = [torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device),
            x, x ** 2, torch.tanh(x)]
    pairs = [x[..., i:i + 1] * x[..., j:j + 1]
             for i in range(f) for j in range(i + 1, f)]
    return torch.cat(cols + pairs, dim=-1)


def basis_names(f: int = N_FEATURES) -> list[str]:
    names = ["1"] + list(FEAT_NAMES)[:f] + \
        [f"{n}^2" for n in FEAT_NAMES[:f]] + \
        [f"tanh({n})" for n in FEAT_NAMES[:f]]
    names += [f"{FEAT_NAMES[i]}*{FEAT_NAMES[j]}"
              for i in range(f) for j in range(i + 1, f)]
    return names


def omp(F: torch.Tensor, y: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthogonal matching pursuit → (indices, coefficients) of a ≤k-sparse
    solution to F @ w ≈ y."""
    idx, r = [], y.clone()
    F = F.clone()
    for _ in range(k):
        c = (F.t() @ r).abs()
        i = int(c.argmax())
        if i in idx:
            break
        idx.append(i)
        w, *_ = torch.linalg.lstsq(F[:, idx], y)
        r = y - F[:, idx] @ w
    w, *_ = torch.linalg.lstsq(F[:, idx], y)
    return torch.tensor(idx), w


def distill_residual(net, feats: torch.Tensor, k: int = 6,
                     ) -> dict[str, list[tuple[str, float]]]:
    """ResidualNet → {output_var: [(term, coef)]} human-readable sparse
    equations per output channel."""
    from nasa_farm.dynamics.residual import RESIDUAL_VARS
    y = net(feats)
    F = basis(feats)
    F = (F - F.mean(0)) / F.std(0).clamp(min=1e-9)
    names = basis_names()
    out: dict[str, list[tuple[str, float]]] = {}
    for j, v in enumerate(RESIDUAL_VARS):
        idx, w = omp(F, y[:, j], k)
        out[v] = [(names[i], float(c)) for i, c in zip(idx.tolist(), w)]
    return out
