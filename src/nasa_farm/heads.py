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
    "stress_days",   # d with water_fac < 0.5 while growing
    "cover_days",    # d with a living canopy
    "biomass_end",   # kg/ha standing biomass at year end
    "n_min_end",     # kg/ha mineral N at year end
    "soc_end",       # kg/ha SOC at year end
)


class SeasonAccumulator:
    """observer(t, state, fluxes) → (B, H) metric tensor at the end."""

    def __init__(self, batch: int, horizon_years: int,
                 device: torch.device | str | None = None) -> None:
        self.horizon = horizon_years
        self.acc = torch.zeros(batch, horizon_years, len(METRICS),
                               dtype=DTYPE, device=device)
        self._i = {name: i for i, name in enumerate(METRICS)}

    def __call__(self, t: int, state: FieldState, fluxes: dict) -> None:
        h = min(t // DAYS_PER_YEAR, self.horizon - 1)
        acc, i = self.acc[:, h], self._i
        acc[:, i["yield_kg"]] += fluxes["yield_event"]
        acc[:, i["water_used"]] += fluxes["es"] + fluxes["t1"] + fluxes["t2"]
        acc[:, i["water_lost"]] += fluxes["runoff"] + fluxes["deep"]
        acc[:, i["n_leached"]] += fluxes["leached"]
        stress = (fluxes["water_fac"] < 0.5) & state.growing
        acc[:, i["stress_days"]] += stress.to(DTYPE)
        acc[:, i["cover_days"]] += state.growing.to(DTYPE)
        acc[:, i["biomass_end"]] = state.biomass
        acc[:, i["n_min_end"]] = state.n_min
        acc[:, i["soc_end"]] = state.soc

    def result(self) -> torch.Tensor:
        """(B, H, M)."""
        return self.acc


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
    stress = m[..., i["stress_days"]].mean(-1)
    soc_gain = m[..., -1, i["soc_end"]] - m[..., 0, i["soc_end"]]
    nmin_end = m[..., -1, i["n_min_end"]]

    def q(x): return torch.quantile(x, risk_q, dim=1)
    def mu(x): return x.mean(1)

    return {
        "yield_mean": mu(yield_tot),
        "yield_risk": q(yield_tot),          # pessimistic-but-likely yield
        "water_used": mu(water + lost),      # total water consumed/lost
        "n_leached": mu(leach),
        "stress_days": mu(stress),
        "soc_gain": mu(soc_gain),
        "n_min_end": mu(nmin_end),
    }


OBJECTIVE_DIRECTION: dict[str, int] = {
    "yield_mean": +1, "yield_risk": +1, "water_used": -1,
    "n_leached": -1, "stress_days": -1, "soc_gain": +1, "n_min_end": +1,
}
