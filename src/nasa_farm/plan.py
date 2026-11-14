"""Planning over the world model — enumeration, batched evaluation, Pareto.

Rotation space is discrete and small (C^H sequences), so exhaustive
evaluation beats sampling-based search: exact Pareto front, zero
approximation error, fully vectorized.
"""

from __future__ import annotations

import itertools
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.heads import OBJECTIVE_DIRECTION, SeasonAccumulator, objectives
from nasa_farm.state.field import Field
from nasa_farm.state.crops import IDX
from nasa_farm.types import Actions, Forcing


def enumerate_rotations(crop_indices: list[int], horizon_years: int) -> torch.Tensor:
    """(R, H) tensor of every crop sequence; R = C^H."""
    seqs = list(itertools.product(crop_indices, repeat=horizon_years))
    return torch.tensor(seqs, dtype=torch.long)


def expand_actions(plans: torch.Tensor, crop_table: torch.Tensor,
                   irr_mm_day: float = 0.0, irr_window: int = 150,
                   n_fert_kg: float = 0.0, fert_delay: int = 30) -> Actions:
    """(R, H) plans → daily Actions (R, H·365).

    Irrigation: `irr_mm_day` every day for `irr_window` days after sowing.
    Fertilizer: `n_fert_kg` applied `fert_delay` days after sowing.
    """
    r, h = plans.shape
    days = h * DAYS_PER_YEAR
    sow = torch.full((r, days), -1, dtype=torch.long)
    irr = torch.zeros(r, days, dtype=DTYPE)
    fert = torch.zeros(r, days, dtype=DTYPE)
    sow_doy = crop_table[plans, IDX["sow_doy"]].long()          # (R, H)
    for y in range(h):
        day0 = y * DAYS_PER_YEAR
        sow_day = day0 + (sow_doy[:, y] - 1).clamp(min=0, max=DAYS_PER_YEAR - 1)
        rows = torch.arange(r)
        sow[rows, sow_day] = plans[:, y]
        if irr_mm_day > 0:
            for d in range(irr_window):
                idx = (sow_day + d).clamp(max=days - 1)
                after = sow_day + d
                mask = after < day0 + DAYS_PER_YEAR
                irr[rows[mask], idx[mask]] += irr_mm_day
        if n_fert_kg > 0:
            fday = (sow_day + fert_delay).clamp(max=days - 1)
            mask = sow_day + fert_delay < day0 + DAYS_PER_YEAR
            fert[rows[mask], fday[mask]] += n_fert_kg
    return Actions(sow=sow, irrigation=irr, n_fert=fert)


@dataclass(frozen=True, slots=True)
class Evaluation:
    metrics: torch.Tensor            # (R, S, H, M)
    objectives: dict[str, torch.Tensor]  # name → (R,)


def _rollout_metrics(model: WorldModel, field0: Field, plans: torch.Tensor,
                     forcing_ens: Forcing, irr_mm_day: float,
                     n_fert_kg: float) -> torch.Tensor:
    """One contiguous plan block → raw seasonal metrics (r_c·S, H, M)."""
    r, h = plans.shape
    s = forcing_ens.batch
    field = field0.repeat(r * s) if field0.batch == 1 else field0
    actions = expand_actions(plans, model.crop_table,
                             irr_mm_day=irr_mm_day, n_fert_kg=n_fert_kg)
    actions = Actions(actions.sow.repeat_interleave(s, dim=0),
                      actions.irrigation.repeat_interleave(s, dim=0),
                      actions.n_fert.repeat_interleave(s, dim=0))
    forcing = forcing_ens.repeat(r)
    acc = SeasonAccumulator(r * s, h)
    with torch.inference_mode():
        model.rollout(field, forcing, actions, observer=acc)
    return acc.result()


def auto_workers() -> int:
    """No-GIL builds scale threads for real; GIL builds stay serial."""
    gil = getattr(sys, "_is_gil_enabled", lambda: True)
    return min(os.cpu_count() or 1, 8) if not gil() else 1


def evaluate(model: WorldModel, field0: Field, plans: torch.Tensor,
             forcing_ens: Forcing, irr_mm_day: float = 0.0,
             n_fert_kg: float = 0.0, workers: int = 1) -> Evaluation:
    """Roll out every plan under every scenario, batched; `workers` shards
    the plan axis across threads — true parallelism on free-threaded Python
    since `model.step` is a pure function with zero shared mutation."""
    r, h = plans.shape
    s = forcing_ens.batch
    w = auto_workers() if workers == 0 else min(workers, r)
    if w > 1 and field0.batch != 1:
        raise ValueError(
            "workers>1 requires a single template field (batch == 1); "
            "pre-batched fields are only supported with workers=1")
    if w > 1:
        chunks = plans.chunk(w)
        # Rollouts are dispatch-bound, not FLOP-bound: intra-op OMP only
        # contends between worker threads, so run each op single-threaded.
        omp_prev = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with ThreadPoolExecutor(max_workers=w) as pool:
                parts = list(pool.map(
                    lambda p: _rollout_metrics(model, field0, p, forcing_ens,
                                               irr_mm_day, n_fert_kg),
                    chunks))
        finally:
            torch.set_num_threads(omp_prev)
        season = torch.cat(parts, dim=0)
    else:
        season = _rollout_metrics(model, field0, plans, forcing_ens,
                                  irr_mm_day, n_fert_kg)
    metrics = season.reshape(r, s, h, -1)
    return Evaluation(metrics=metrics, objectives=objectives(season, r, s))


def objective_matrix(obj: dict[str, torch.Tensor],
                     keys: tuple[str, ...] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack objectives → (R, M) and direction signs (M,) (+1 maximize)."""
    keys = keys or tuple(obj.keys())
    mat = torch.stack([obj[k] for k in keys], dim=1)
    signs = torch.tensor([OBJECTIVE_DIRECTION[k] for k in keys], dtype=DTYPE)
    return mat, signs


def pareto_front(points: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Boolean (R,) mask of non-dominated points. points (R, M)."""
    z = points * signs
    ge = (z.unsqueeze(0) >= z.unsqueeze(1)).all(-1)  # ge[a,b]: b weakly ≥ a
    gt = (z.unsqueeze(0) > z.unsqueeze(1)).any(-1)
    dominated = (ge & gt).any(1)  # a dominated iff ∃ b strictly dominating it
    return ~dominated


def scalarize(points: torch.Tensor, signs: torch.Tensor,
              weights: torch.Tensor) -> torch.Tensor:
    """Range-normalized weighted score (R,). Higher is better."""
    z = points * signs
    span = (z.max(0).values - z.min(0).values).clamp(min=1e-9)
    zn = (z - z.min(0).values) / span
    return (zn * weights / weights.sum()).sum(-1)


def recommend(evaluation: Evaluation, weights: dict[str, float] | None = None,
              top_k: int = 5) -> dict:
    """Pareto set + priority-weighted ranking."""
    mat, signs = objective_matrix(evaluation.objectives)
    mask = pareto_front(mat, signs)
    w = torch.ones(mat.shape[1], dtype=DTYPE)
    if weights:
        keys = tuple(evaluation.objectives.keys())
        for k, v in weights.items():
            if k in keys:
                w[keys.index(k)] = float(v)
    score = scalarize(mat, signs, w)
    order = torch.argsort(score, descending=True)
    return {
        "pareto_idx": mask.nonzero().flatten(),
        "ranking": order[:top_k],
        "scores": score,
        "objectives": evaluation.objectives,
        "metric_names": tuple(evaluation.objectives.keys()),
    }
