"""Symbolic distillation — read the learned residual back into equations.

Sparse regression over a fixed basis library (SINDy-style, orthogonal
matching pursuit) fits each residual output channel as a small closed-form
combination of interpretable features. The distilled terms can then be
reinjected into the physics core — the loop that turns missing physics into
explicit physics.
"""

from __future__ import annotations

import torch

from nasa_farm.dynamics.residual import N_FEATURES

FEAT_NAMES: tuple[str, ...] = (
    "sw0",
    "swr",
    "n_min",
    "lai",
    "biomass",
    "gdd",
    "growing",
    "tmin",
    "tmax",
    "srad",
    "precip",
    "irrigation",
    "n_fert",
    "sow",
)


def basis(x: torch.Tensor) -> torch.Tensor:
    """(B, F) features → (B, L) library: x, x², tanh, pairwise products of
    the first-order terms, plus bias. L = 1 + 3F + F(F−1)/2."""
    f = x.shape[-1]
    cols = [torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device), x, x**2, torch.tanh(x)]
    pairs = [x[..., i : i + 1] * x[..., j : j + 1] for i in range(f) for j in range(i + 1, f)]
    return torch.cat(cols + pairs, dim=-1)


def basis_names(f: int = N_FEATURES) -> list[str]:
    names = (
        ["1"]
        + list(FEAT_NAMES)[:f]
        + [f"{n}^2" for n in FEAT_NAMES[:f]]
        + [f"tanh({n})" for n in FEAT_NAMES[:f]]
    )
    names += [f"{FEAT_NAMES[i]}*{FEAT_NAMES[j]}" for i in range(f) for j in range(i + 1, f)]
    return names


def pursuit(F: torch.Tensor, y: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthogonal matching pursuit → (indices, coefficients) of a ≤k-sparse
    solution to F @ w ≈ y."""
    index, r = [], y.clone()
    F = F.clone()
    for _ in range(k):
        c = (F.t() @ r).abs()
        i = int(c.argmax())
        if i in index:
            break
        index.append(i)
        w, *_ = torch.linalg.lstsq(F[:, index], y)
        r = y - F[:, index] @ w
    w, *_ = torch.linalg.lstsq(F[:, index], y)
    return torch.tensor(index), w


def grow(
    x: torch.Tensor, y: torch.Tensor, k: int = 6, rounds: int = 2
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """AutoSINDy-style library growth.

    Start from the fixed basis; after each OMP fit, augment the library
    with products of the *selected* terms and the first-order features —
    the dictionary chases the structure the data actually uses, then OMP
    refits on the grown library. Returns (indices, coefficients, names).
    """
    F = basis(x)
    names = basis_names()
    mean, deviation = F.mean(0), F.std(0).clamp(min=1e-9)
    normalized = (F - mean) / deviation
    f = x.shape[-1]
    for _ in range(rounds + 1):
        index, w = pursuit(normalized, y, k)
        sel = set(index.tolist())
        new_cols, new_names = [], []
        for i in sel:
            if i == 0:
                continue
            for j in range(1, f + 1):
                column = normalized[:, i] * normalized[:, j]
                if column.std() < 1e-6:
                    continue
                new_cols.append((column - column.mean()) / column.std())
                new_names.append(f"{names[i]}·{names[j]}")
        if not new_cols:
            break
        normalized = torch.cat([normalized] + [c.unsqueeze(-1) for c in new_cols], dim=-1)
        names += new_names
    index, w = pursuit(normalized, y, k)
    return index, w, names


def distill_residual(
    net,
    features: torch.Tensor,
    k: int = 6,
    autogrow: bool = False,
) -> dict[str, list[tuple[str, float]]]:
    """ResidualNet → {output_var: [(term, coef)]} human-readable sparse
    equations per output channel. autogrow=True grows the library from
    selected terms (AutoSINDy-style) before the final OMP fit."""
    from nasa_farm.dynamics.residual import RESIDUAL_VARS

    y = net(features)
    if autogrow:
        output: dict[str, list[tuple[str, float]]] = {}
        for j, v in enumerate(RESIDUAL_VARS):
            index, w, names = grow(features, y[:, j], k)
            output[v] = [(names[i], float(c)) for i, c in zip(index.tolist(), w, strict=True)]
        return output
    F = basis(features)
    F = (F - F.mean(0)) / F.std(0).clamp(min=1e-9)
    names = basis_names()
    output: dict[str, list[tuple[str, float]]] = {}
    for j, v in enumerate(RESIDUAL_VARS):
        index, w = pursuit(F, y[:, j], k)
        output[v] = [(names[i], float(c)) for i, c in zip(index.tolist(), w, strict=True)]
    return output
