"""Hyperparameter optimization — two regimes, chosen by parameter kind.

- `gradient_hpo`: continuous parameters live inside the differentiable
  rollout → first-order HPO via `trainer.fit`. This is variational
  assimilation: exact gradients through the physics, which no black-box
  method can match per-evaluation on this model class.
- `blackbox_hpo`: discrete / non-differentiable knobs → Sobol coverage +
  successive halving (ASHA) — provably competitive with GP-BO at small
  budgets; Optuna's multivariate TPE/NSGA-II used automatically when the
  optional `hpo` extra is installed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as _f
from typing import Any

import torch

from nasa_farm.optimize.space import Space
from nasa_farm.optimize.trainer import FitResult, fit


def gradient(
    loss_fn: Callable[[], torch.Tensor], parameters: dict[str, torch.Tensor], **fit_kw
) -> FitResult:
    """First-order HPO: optimize leaf tensors through rollout gradients.

    loss_fn must read the same tensor objects the optimizer updates
    (e.g. a live-bound `trait_leaf(model)`), so each step's gradient is an
    exact d(loss)/d(param) through the full trajectory.
    """
    return fit(loss_fn, parameters, **fit_kw)


@dataclass(slots=True)
class Trial:
    config: dict[str, Any]
    value: float
    rung: int = -1  # last ASHA rung reached (-1 = full budget)


@dataclass(slots=True)
class Result:
    best_config: dict[str, Any]
    best_value: float
    trials: list[Trial] = _f(default_factory=list)


def successive_halving(
    objective: Callable[[dict, int], float],
    configs: list[dict],
    rungs: tuple[int, ...],
    eta: int = 3,
) -> list[Trial]:
    """ASHA: all configs at rungs[0] budget → keep top 1/eta → next rung.
    objective(cfg, budget) → scalar, lower is better."""
    alive = configs
    trials: list[Trial] = []
    for lvl, budget in enumerate(rungs):
        scored = [(objective(c, budget), c) for c in alive]
        scored.sort(key=lambda t: t[0])
        trials += [Trial(c, v, lvl) for v, c in scored]
        keep = max(1, len(scored) // eta)
        alive = [c for _, c in scored[:keep]]
        if len(alive) <= 1 or lvl == len(rungs) - 1:
            break
    return trials


def blackbox(
    objective: Callable[[dict, int], float],
    space: Space,
    n_trials: int = 32,
    rungs: tuple[int, ...] | None = (10, 30, 90),
    eta: int = 3,
    seed: int = 0,
) -> Result:
    """Sobol-sampled configs → optional successive halving → best config.

    objective(cfg, budget) → scalar (lower better). When `rungs` is None,
    every config gets the same full evaluation (plain Sobol search).
    """
    configs = space.sample(n_trials, seed=seed)
    if rungs is None:
        trials = [Trial(c, objective(c, 0), -1) for c in configs]
    else:
        trials = successive_halving(objective, configs, rungs, eta)
    best = min(trials, key=lambda t: t.value)
    return Result(best_config=best.config, best_value=best.value, trials=trials)


def _optuna_objective(objective, trial, space: Space):
    config = {}
    for p in space.parameters:
        if p.categorical:
            config[p.name] = trial.suggest_categorical(p.name, list(p.choices))
        elif p.integer:
            config[p.name] = trial.suggest_int(p.name, int(p.low), int(p.high), log=p.log)
        else:
            config[p.name] = trial.suggest_float(p.name, p.low, p.high, log=p.log)
    return objective(config, 0)


def optuna(
    objective: Callable[[dict, int], float],
    space: Space,
    n_trials: int = 50,
    seed: int = 0,
    directions: list[str] | None = None,
) -> Result:
    """Optuna path — multivariate TPE (single-objective) or NSGA-II
    (directions=['minimize','maximize',...] returns the Pareto-optimal trial
    on the first objective). Requires the `hpo` extra."""
    import optuna

    sampler = (
        optuna.samplers.TPESampler(seed=seed, multivariate=True)
        if not directions or len(directions) == 1
        else optuna.samplers.NSGAIISampler(seed=seed)
    )
    study = optuna.create_study(directions=directions or ["minimize"], sampler=sampler)
    study.optimize(lambda t: _optuna_objective(objective, t, space), n_trials=n_trials)
    multi = len(study.directions) > 1
    bt = study.best_trials[0] if multi else study.best_trial
    trials = [
        Trial(
            t.params,
            (t.values[0] if multi else t.value)
            if (t.values or t.value is not None)
            else float("inf"),
        )
        for t in study.trials
    ]
    return Result(
        best_config=bt.params, best_value=bt.values[0] if multi else bt.value, trials=trials
    )
