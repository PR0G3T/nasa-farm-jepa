"""Climate-analogue weighting — bias the scenario ensemble toward historical
years resembling a reference (seasonal outlook, ENSO state, forecast mean).

Kernel weights over annual feature vectors, then resample members with
replacement so downstream code keeps working on plain Forcing batches.
"""

from __future__ import annotations

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.types import Forcing


def year_features(f: Forcing) -> torch.Tensor:
    """(F,) annual signature: total precip, mean temperatures, mean
    radiation, dry-day count, mean humidity — the analogue distance space."""
    dry = (f.precip[0] < 1.0).sum().to(DTYPE)
    return torch.stack(
        [
            f.precip[0].sum() / 500.0,
            f.tmin[0].mean() / 20.0,
            f.tmax[0].mean() / 30.0,
            f.srad[0].mean() / 20.0,
            dry / 200.0,
            f.rh[0].mean() / 80.0,
        ]
    )


def analog_weights(
    history: list[Forcing], target: Forcing | torch.Tensor, bandwidth: float = 1.0
) -> torch.Tensor:
    """(N,) normalized importance weights — Gaussian kernel on feature
    distance to the reference year (or raw feature vector)."""
    t = year_features(target) if isinstance(target, Forcing) else target
    d = torch.stack([((year_features(h) - t) ** 2).sum() for h in history])
    w = torch.exp(-d / (2.0 * bandwidth**2))
    return w / w.sum().clamp(min=1e-12)


def weighted_ensemble(
    history: list[Forcing], weights: torch.Tensor, horizon: int, seed: int = 0
) -> Forcing:
    """Resample historical years ∝ weights, tile across the horizon."""
    from nasa_farm.forcing.base import climatology_ensemble

    g = torch.Generator().manual_seed(seed)
    pick = torch.multinomial(weights, len(history), replacement=True, generator=g)
    return climatology_ensemble([history[i] for i in pick.tolist()], horizon)
