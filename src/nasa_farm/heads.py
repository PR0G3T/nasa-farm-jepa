"""Interpretable heads — online seasonal metrics.

`SeasonAccumulator` folds each daily step into per-year metrics without
storing trajectories: memory stays O(B × H) regardless of horizon length.
"""

from __future__ import annotations

import torch

from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.state.field import FieldState

METRICS: tuple[str, ...] = (
    "yield_kg",      # kg/ha harvested grain/hay in year h
    "water_used",    # mm evapotranspired in year h
    "water_lost",    # mm runoff + deep drainage in year h
    "n_leached",     # kg N/ha leached in year h
    "n2o_kg",        # kg N-N2O/ha emitted in year h
    "stress_days",   # d with water_fac < 0.5 while growing
    "cover_days",    # d with a living canopy
    "biomass_end",   # kg/ha standing biomass at year end
    "n_min_end",     # kg/ha mineral N at year end
    "soc_end",       # kg/ha total SOC (active+stable) at year end
    "soc_f_end",     # kg/ha active-pool SOC at year end
    "tws_end",       # mm aquifer storage anomaly at year end
    "heat_dd",       # °C·d reproductive heat load in year h
    "frost_days",    # d with a lethal-frost event in year h
)

_END: tuple[str, ...] = ("biomass_end", "n_min_end", "soc_end", "soc_f_end",
                         "tws_end")
"""Absolute-write metrics (last-day value), all others accumulate."""

SUM_I = [i for i, n in enumerate(METRICS) if n not in _END]
END_I = [i for i, n in enumerate(METRICS) if n in _END]


def day_update(state: FieldState, fluxes: dict) -> torch.Tensor:
    """(B, M) daily metric update — SUM channels carry increments, _END
    channels carry the absolute value to write. Pure and differentiable, so
    the checkpointed rollout can fold it inside segments."""
    g = state.growing.to(DTYPE)
    cols = {
        "yield_kg": fluxes["yield_event"],
        "water_used": fluxes["es"] + fluxes["t"].sum(-1),
        "water_lost": fluxes["runoff"] + fluxes["deep"],
        "n_leached": fluxes["leached"],
        "n2o_kg": fluxes["n2o"],
        "stress_days": ((fluxes["water_fac"] < 0.5) & state.growing).to(DTYPE),
        "cover_days": g,
        "biomass_end": state.biomass,
        "n_min_end": state.n_min,
        "soc_end": state.soc_f + state.soc_s,
        "soc_f_end": state.soc_f,
        "tws_end": state.tws,
        "heat_dd": fluxes["heat_dd"],
        "frost_days": fluxes["frost"].to(DTYPE),
    }
    return torch.stack([cols[n] for n in METRICS], dim=-1)


def fold(accumulate: torch.Tensor, t: int, update: torch.Tensor) -> torch.Tensor:
    """acc' = acc with one daily update folded in — out-of-place, autograd-safe
    equivalent of `acc[:, h, SUM] += u` + `acc[:, h, END] = u`."""
    h = min(t // DAYS_PER_YEAR, accumulate.shape[1] - 1)
    m = accumulate.shape[-1]
    sel = (torch.arange(accumulate.shape[1], device=accumulate.device) == h).to(accumulate.dtype)
    end_mask = torch.zeros(m, dtype=torch.bool, device=accumulate.device)
    end_mask[END_I] = True
    sum_mask = ~end_mask
    u = update.unsqueeze(1)
    accumulate = accumulate + sel.view(1, -1, 1) * u * sum_mask.to(u.dtype)
    accumulate = torch.where(
        sel.view(1, -1, 1).bool() & end_mask.view(1, 1, -1),
        u.expand(-1, accumulate.shape[1], -1), accumulate)
    return accumulate


class SeasonAccumulator:
    """observer(t, state, fluxes) → (B, H, M) metric tensor at the end."""

    def __init__(self, batch: int, horizon_years: int,
                 device: torch.device | str | None = None) -> None:
        self.horizon = horizon_years
        self.accumulate = torch.zeros(batch, horizon_years, len(METRICS),
                               dtype=DTYPE, device=device)

    def __call__(self, t: int, state: FieldState, fluxes: dict) -> None:
        h = min(t // DAYS_PER_YEAR, self.horizon - 1)
        u = day_update(state, fluxes)
        accumulate = self.accumulate[:, h]
        accumulate[:, SUM_I] += u[:, SUM_I]
        accumulate[:, END_I] = u[:, END_I]

    def result(self) -> torch.Tensor:
        """(B, H, M)."""
        return self.accumulate


def objectives(season_metrics: torch.Tensor, n_plans: int, n_scen: int,
               risk_q: float = 0.1) -> dict[str, torch.Tensor]:
    """Fold (B=R·S, H, M) into per-plan objective vectors (R,).

    Robustness uses the `risk_q` quantile across scenarios, not the mean —
    that is what 'risk for each rotation' means in the challenge brief.
    """
    i = {name: k for k, name in enumerate(METRICS)}
    m = season_metrics.reshape(n_plans, n_scen, *season_metrics.shape[1:])

    yield_tot = m[..., i["yield_kg"]].sum(-1)                 # (R, S)
    water = m[..., i["water_used"]].sum(-1)
    lost = m[..., i["water_lost"]].sum(-1)
    leach = m[..., i["n_leached"]].sum(-1)
    n2o = m[..., i["n2o_kg"]].sum(-1)
    stress = m[..., i["stress_days"]].mean(-1)
    heat = m[..., i["heat_dd"]].sum(-1)
    frost = m[..., i["frost_days"]].sum(-1)
    soc_gain = m[..., -1, i["soc_end"]] - m[..., 0, i["soc_end"]]
    nmin_end = m[..., -1, i["n_min_end"]]

    def q(x): return torch.quantile(x, risk_q, dim=1)
    def mean(x): return x.mean(1)

    return {
        "yield_mean": mean(yield_tot),
        "yield_risk": q(yield_tot),          # pessimistic-but-likely yield
        "water_used": mean(water + lost),      # total water consumed/lost
        "n_leached": mean(leach),
        "n2o_kg": mean(n2o),
        "stress_days": mean(stress),
        "heat_dd": mean(heat),                 # reproductive heat load
        "frost_days": mean(frost),             # lethal-frost frequency
        "soc_gain": mean(soc_gain),
        "n_min_end": mean(nmin_end),
    }


OBJECTIVE_DIRECTION: dict[str, int] = {
    "yield_mean": +1, "yield_risk": +1, "water_used": -1,
    "n_leached": -1, "n2o_kg": -1, "stress_days": -1, "heat_dd": -1,
    "frost_days": -1, "soc_gain": +1, "n_min_end": +1,
}
