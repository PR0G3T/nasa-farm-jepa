"""Champion-challenger gate — only a validated refit replaces the live model.

`gate` snapshots the weights, runs the caller's refit, scores both
versions on the caller's metric and keeps the challenger only when it is
strictly better. A failed refit rolls back byte-for-byte, so the physics
prior is never lost to a bad online update.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def gate(model: nn.Module, fit: Callable[[], object],
         score: Callable[[], float]) -> dict:
    """Snapshot → fit() → promote iff score() improved (lower is better).

    `fit` mutates `model` in place; `score` evaluates the live model.
    Returns {promoted, champion, challenger} metrics for audit."""
    champ = {k: v.detach().clone() for k, v in model.state_dict().items()}
    c0 = float(score())
    fit()
    c1 = float(score())
    if not c1 < c0:
        model.load_state_dict(champ)
    return {"promoted": c1 < c0, "champion": c0, "challenger": c1}
