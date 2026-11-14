"""Planning over the world model — enumeration, batched evaluation, Pareto.

Rotation space is discrete and small (C^H sequences), so exhaustive
evaluation beats sampling-based search: exact Pareto front, zero
approximation error, fully vectorized.
"""

from __future__ import annotations

import itertools
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


@dataclass(frozen=True)
class Evaluation:
    metrics: torch.Tensor            # (R, S, H, M)
    objectives: dict[str, torch.Tensor]  # name → (R,)


def evaluate(model: WorldModel, field0: Field, plans: torch.Tensor,
             forcing_ens: Forcing, irr_mm_day: float = 0.0,
             n_fert_kg: float = 0.0) -> Evaluation:
    """Roll out every plan under every scenario in one batched pass."""
    r, h = plans.shape
    s = forcing_ens.batch
    b = r * s

    field = field0.repeat(b) if field0.batch == 1 else field0
    actions = expand_actions(plans, model.crop_table,
                             irr_mm_day=irr_mm_day, n_fert_kg=n_fert_kg)
    actions = Actions(actions.sow.repeat_interleave(s, dim=0),
                      actions.irrigation.repeat_interleave(s, dim=0),
                      actions.n_fert.repeat_interleave(s, dim=0))
    forcing = forcing_ens.repeat(r)

    acc = SeasonAccumulator(b, h)
    with torch.inference_mode():
        model.rollout(field, forcing, actions, observer=acc)
    metrics = acc.result().reshape(r, s, h, -1)
    return Evaluation(metrics=metrics, objectives=objectives(acc.result(), r, s))


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
