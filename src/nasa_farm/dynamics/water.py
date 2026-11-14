"""Two-layer soil water balance — exactly conservative by construction.

Storage-change identity enforced per step:
    Δ(sw1 + sw2) ≡ infiltration − runoff − evaporation − transpiration − deep drainage
Layer-1 drainage is an internal transfer and cancels in the global budget.

Transpiration follows the Ritchie root-zone scheme: plants transpire at
potential rate while plant-available water exceeds 50 % of capacity, then
decline linearly; uptake is split between layers proportionally to local
availability × root fraction.
"""

from __future__ import annotations

import torch

from nasa_farm.state.soil import SoilProfile

K_PERC = 0.4
"""Fraction of above-field-capacity water draining per day."""

WUF_THRESHOLD = 0.5
"""PAW fraction below which transpiration falls below potential."""


def plant_available(sw: torch.Tensor, wp: torch.Tensor) -> torch.Tensor:
    """Water accessible to roots (mm)."""
    return torch.relu(sw - wp)


def paw_fraction(sw1: torch.Tensor, sw2: torch.Tensor,
                 soil: SoilProfile) -> torch.Tensor:
    """Plant-available water as fraction of root-zone capacity ∈ [0,1]."""
    paw = plant_available(sw1, soil.layer1.theta_wp) + \
        plant_available(sw2, soil.layer2.theta_wp)
    paw_max = (soil.layer1.theta_fc - soil.layer1.theta_wp) + \
        (soil.layer2.theta_fc - soil.layer2.theta_wp)
    return torch.clamp(paw / paw_max, 0.0, 1.0)


def water_uptake_factor(sw1: torch.Tensor, sw2: torch.Tensor,
                        soil: SoilProfile) -> torch.Tensor:
    """Transpiration/growth reduction ∈ [0,1]; 1 while PAW ≥ 50 % capacity."""
    return torch.clamp(2.0 * paw_fraction(sw1, sw2, soil), 0.0, 1.0)


def water_stress(sw: torch.Tensor, wp: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
    """Per-layer availability factor ∈ [0,1] (diagnostic only)."""
    return torch.clamp((sw - wp) / (fc - wp), 0.0, 1.0)


def evaporation_stress(sw: torch.Tensor, air: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
    """Soil-evaporation supply factor ∈ [0,1] relative to air-dry bound."""
    return torch.clamp((sw - air) / (fc - air), 0.0, 1.0)


def water_balance_step(
    sw1: torch.Tensor,
    sw2: torch.Tensor,
    soil: SoilProfile,
    infiltration: torch.Tensor,
    evap_potential: torch.Tensor,
    trans_potential: torch.Tensor,
    root_frac2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Advance one day. Returns (sw1', sw2', fluxes). All tensors (B,)."""
    l1, l2 = soil.layer1, soil.layer2

    wuf = water_uptake_factor(sw1, sw2, soil)
    trans_actual = trans_potential * wuf

    # Evaporation has first claim on layer-1 water down to air-dry; roots can
    # only extract what remains above wilting point.
    ew = evaporation_stress(sw1, l1.theta_air, l1.theta_fc)
    es = torch.minimum(evap_potential * ew, torch.relu(sw1 - l1.theta_air))

    avail1 = torch.relu(sw1 - es - l1.theta_wp)
    avail2 = plant_available(sw2, l2.theta_wp)
    frac1 = 1.0 - root_frac2
    den = frac1 * avail1 + root_frac2 * avail2
    share1 = torch.where(den > 0, frac1 * avail1 / torch.clamp(den, min=1e-9),
                         torch.zeros_like(den))
    t1 = torch.minimum(trans_actual * share1, avail1)
    t2 = torch.minimum(trans_actual * (1.0 - share1), avail2)

    runoff = torch.relu(sw1 + infiltration - es - t1 - l1.theta_sat)
    sw1a = sw1 + infiltration - runoff - es - t1

    perc = torch.relu(sw1a - l1.theta_fc) * K_PERC
    sw1b = sw1a - perc

    sw2a = sw2 + perc - t2
    deep = torch.relu(sw2a - l2.theta_fc) * K_PERC
    sw2b = sw2a - deep

    fluxes = {"es": es, "t1": t1, "t2": t2, "runoff": runoff, "perc": perc,
              "deep": deep, "wuf": wuf, "pawf": paw_fraction(sw1, sw2, soil),
              "w1": water_stress(sw1, l1.theta_wp, l1.theta_fc),
              "w2": water_stress(sw2, l2.theta_wp, l2.theta_fc)}
    return sw1b, sw2b, fluxes
