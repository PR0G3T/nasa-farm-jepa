"""First-order trainer — the engine behind gradient HPO and NAS scoring.

`fit` optimizes any set of leaf tensors through a caller-supplied loss
closure. The closure typically rebuilds a field/model context and calls
`targets.season_metrics` — which keeps autograd intact — so parameters like
the crop trait table or soil storages are updated by direct rollout
gradients: variational data assimilation, strictly more informative than
any derivative-free evaluation of the same budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _f
from typing import Callable

import torch


def leaf(t: torch.Tensor | float) -> torch.Tensor:
    """Detachable trainable leaf — optimizer updates flow through the model
    when the leaf is also the tensor the model reads (e.g. crop_table)."""
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    return t.detach().clone().requires_grad_(True)


def trait_leaf(model) -> torch.Tensor:
    """crop_table → trainable leaf, live-bound to the model buffer."""
    t = model.crop_table.detach().clone().requires_grad_(True)
    model.crop_table = t
    return t


@dataclass(slots=True)
class FitResult:
    best_loss: float
    history: list[float]
    epochs_run: int
    parameters: dict[str, torch.Tensor] = _f(default_factory=dict)


def fit(loss_fn: Callable[[], torch.Tensor],
        parameters: dict[str, torch.Tensor],
        epochs: int = 200, rate: float = 1e-2, clip: float = 1.0,
        patience: int = 30, verbose: bool = False) -> FitResult:
    """Adam + grad clip + early stopping. Deterministic given the closure."""
    def evaluate() -> tuple[float, dict | None]:
        loss = loss_fn()
        if not torch.isfinite(loss):
            return float("nan"), None
        loss.backward()
        return float(loss.detach()), None
    return _train(evaluate, parameters, epochs, rate, clip, patience, verbose)


def fit_batch(evaluate: Callable[[], tuple[float, dict[str, torch.Tensor] | None]],
              parameters: dict[str, torch.Tensor],
              epochs: int = 200, rate: float = 1e-2, clip: float = 1.0,
              patience: int = 30, verbose: bool = False) -> FitResult:
    """`fit` whose epoch evaluation may run anywhere — a process pool, a
    remote, whatever — as long as it returns (loss, grads) with grads
    aligned on `parameters` keys. Returning None for grads means the
    evaluation already backpropagated onto the leaves (in-process path)."""
    return _train(evaluate, parameters, epochs, rate, clip, patience, verbose)


def _train(evaluate: Callable[[], tuple[float, dict | None]],
           parameters: dict[str, torch.Tensor],
           epochs: int, rate: float, clip: float,
           patience: int, verbose: bool) -> FitResult:
    leaves = list(parameters.values())
    opt = torch.optim.Adam(leaves, lr=rate)
    best, history = float("inf"), []
    best_params = {k: v.detach().clone() for k, v in parameters.items()}
    stall = 0
    for evaporation in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        v, grads = evaluate()
        if not math.isfinite(v):
            break
        if grads:
            for k, p in parameters.items():
                p.grad = grads[k]
        torch.nn.utils.clip_grad_norm_(leaves, clip)
        opt.step()
        history.append(v)
        if verbose and evaporation % 25 == 0:
            print(f"  epoch {evaporation}: loss {v:.5f}")
        if v < best - 1e-9:
            best, stall = v, 0
            best_params = {k: p.detach().clone() for k, p in parameters.items()}
        else:
            stall += 1
            if stall >= patience:
                break
    # leave the *best* parameters in place, not the last ones
    with torch.no_grad():
        for k, p in parameters.items():
            p.copy_(best_params[k])
    return FitResult(best_loss=best, history=history,
                     epochs_run=len(history), parameters=best_params)
