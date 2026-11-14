"""N-layer soil water balance — exactly conservative by construction.

Storage-change identity enforced per step:
    ΔΣsw ≡ infiltration − runoff − evaporation − transpiration − deep drainage
Inter-layer percolation is an internal transfer and cancels in the budget.

Transpiration follows the Ritchie root-zone scheme: plants transpire at
potential rate while plant-available water exceeds 50 % of capacity, then
decline linearly; uptake is split between layers proportionally to local
availability × root weight. Reduces bit-for-bit to the classic two-layer
column when L=2.
"""

from __future__ import annotations

import torch

from nasa_farm.state.soil import SoilProfile

K_PERC = 0.4
"""Fraction of above-field-capacity water draining per layer per day."""

WUF_THRESHOLD = 0.5
"""PAW fraction below which transpiration falls below potential."""


def plant_available(sw: torch.Tensor, wp: torch.Tensor) -> torch.Tensor:
    """Water accessible to roots (mm), (B, L)."""
    return torch.relu(sw - wp)


def paw_fraction(sw: torch.Tensor, soil: SoilProfile) -> torch.Tensor:
    """Plant-available water as fraction of root-zone capacity ∈ [0,1]."""
    paw = plant_available(sw, soil.wp).sum(-1)
    paw_max = (soil.fc - soil.wp).sum(-1)
    return torch.clamp(paw / paw_max, 0.0, 1.0)


def water_uptake_factor(sw: torch.Tensor, soil: SoilProfile) -> torch.Tensor:
    """Transpiration/growth reduction ∈ [0,1]; 1 while PAW ≥ 50 % capacity."""
    return torch.clamp(2.0 * paw_fraction(sw, soil), 0.0, 1.0)


def water_stress(sw: torch.Tensor, soil: SoilProfile) -> torch.Tensor:
    """Per-layer availability factor ∈ [0,1] (diagnostic), (B, L)."""
    return torch.clamp((sw - soil.wp) / (soil.fc - soil.wp), 0.0, 1.0)


def evaporation_stress(sw0: torch.Tensor, air0: torch.Tensor, fc0: torch.Tensor) -> torch.Tensor:
    """Soil-evaporation supply factor ∈ [0,1] relative to air-dry bound."""
    return torch.clamp((sw0 - air0) / (fc0 - air0), 0.0, 1.0)


def water_balance_step(
    sw: torch.Tensor,
    soil: SoilProfile,
    infiltration: torch.Tensor,
    evap_potential: torch.Tensor,
    trans_potential: torch.Tensor,
    root_w: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Advance one day. sw/root_w (B, L). Returns (sw', fluxes)."""
    wuf = water_uptake_factor(sw, soil)
    trans_actual = trans_potential * wuf

    # Evaporation has first claim on layer-0 water down to air-dry; roots can
    # only extract what remains above wilting point.
    ew = evaporation_stress(sw[:, 0], soil.air[:, 0], soil.fc[:, 0])
    es = torch.minimum(evap_potential * ew, torch.relu(sw[:, 0] - soil.air[:, 0]))

    available = torch.relu(sw - soil.wp)
    available = torch.cat(
        [torch.relu(sw[:, 0] - es - soil.wp[:, 0]).unsqueeze(1),
         available[:, 1:]], dim=1)
    den = (root_w * available).sum(-1, keepdim=True)
    share = torch.where(den > 0, root_w * available / den.clamp(min=1e-9),
                        torch.zeros_like(available))
    t = torch.minimum(trans_actual.unsqueeze(-1) * share, available)

    runoff = torch.relu(sw[:, 0] + infiltration - es - t[:, 0] - soil.sat[:, 0])
    xs = [sw[:, 0] + infiltration - runoff - es - t[:, 0]] + \
        [sw[:, i] - t[:, i] for i in range(1, sw.shape[1])]

    perc = torch.zeros_like(infiltration)
    deep = torch.zeros_like(infiltration)
    for i in range(sw.shape[1]):
        p = torch.relu(xs[i] - soil.fc[:, i]) * K_PERC
        xs[i] = xs[i] - p
        if i == 0:
            perc = p
        if i + 1 < sw.shape[1]:
            xs[i + 1] = xs[i + 1] + p
        else:
            deep = p

    fluxes = {"es": es, "t": t, "runoff": runoff, "perc": perc, "deep": deep,
              "wuf": wuf, "pawf": paw_fraction(sw, soil),
              "w": water_stress(sw, soil)}
    return torch.stack(xs, dim=1), fluxes
