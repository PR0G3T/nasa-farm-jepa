"""Sparse identification of the learned residual.

The ResidualNet is a black-box correction law; SINDy turns it into a
readable one. A candidate library over featurize() inputs (linear terms,
hinge terms at empirical quartiles, pairwise products of the coupled
subset) is fit by sequential thresholded least squares per RESIDUAL_VARS
channel — sparse equations the physics did not write down.
"""

from __future__ import annotations

import torch

from nasa_farm.dynamics.residual import RESIDUAL_VARS
from nasa_farm.state.field import Field
from nasa_farm.types import Actions, Forcing

FEATURES: tuple[str, ...] = (
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

_SUB = (0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13)
"""Feature indices admitted into pairwise products (srad/tmin excluded:
they enter through their hinge terms)."""


def library(features: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    """(N, F) → design matrix (N, n_terms) plus term names."""
    q = features.quantile(0.75, dim=0)
    hinge = torch.relu(features - q)
    cols = list(_SUB)
    pairs = [(i, j) for i in cols for j in cols if i < j]
    prod = torch.stack([features[:, i] * features[:, j] for i, j in pairs], -1)
    names = (
        list(FEATURES)
        + [f"relu({n})" for n in FEATURES]
        + [f"{FEATURES[i]}*{FEATURES[j]}" for i, j in pairs]
    )
    return torch.cat([features, hinge, prod], -1), names


def thresholded(
    X: torch.Tensor,
    Y: torch.Tensor,
    lam: float = 1e-4,
    threshold: float = 1e-3,
    iterations: int = 10,
) -> torch.Tensor:
    """Sequential thresholded least squares → (n_terms, n_out) coefs."""
    d = X.shape[1]
    eye = lam * torch.eye(d, dtype=X.dtype, device=X.device)
    coef = torch.linalg.solve(X.t() @ X + eye, X.t() @ Y)
    for _ in range(iterations):
        small = coef.abs() < threshold
        coef = coef.masked_fill(small, 0.0)
        for j in range(Y.shape[1]):
            big = ~small[:, j]
            if big.any():
                coef[big, j] = torch.linalg.solve(
                    X[:, big].t() @ X[:, big] + eye[big][:, big], X[:, big].t() @ Y[:, j]
                )
    return coef


def collect(
    model,
    field: Field,
    forcing: Forcing,
    actions: Actions,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, F) residual inputs and (N, V) outputs along the model rollout."""
    X, Y = [], []
    for t in range(forcing.days):
        features = model._features(field.state, forcing.day(t), actions.day(t))
        output = model._nn(model.residual, features, field.state.z)
        Y.append(output[0] if isinstance(output, tuple) else output)
        X.append(features)
        field, _ = model.step(field, actions.day(t), forcing.day(t), t)
    return torch.cat(X), torch.cat(Y)


def identify(
    model,
    field: Field,
    forcing: Forcing,
    actions: Actions,
    threshold: float = 1e-3,
    lam: float = 1e-4,
) -> dict:
    """SINDy on the residual → readable per-channel equations."""
    X, Y = collect(model, field, forcing, actions)
    scale = Y.std(0).clamp(min=1e-9)
    phi, names = library(X)
    coef = thresholded(phi, Y / scale, lam=lam, threshold=threshold).detach()
    eqs = {}
    for j, v in enumerate(RESIDUAL_VARS):
        nz = coef[:, j].abs() > 0
        eqs[v] = {
            names[i]: round(float(coef[i, j]), 6) for i in torch.nonzero(nz).flatten().tolist()
        }
    result = (Y / scale - phi @ coef).detach().pow(2).sum()
    total = (Y / scale).detach().pow(2).sum()
    r2 = 1.0 - float(result) / float(total) if float(total) > 1e-12 else 0.0
    return {"equations": eqs, "coef": coef, "names": names, "scale": scale, "r2": round(r2, 4)}
