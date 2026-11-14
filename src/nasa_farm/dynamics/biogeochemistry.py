"""Nitrogen and carbon pools — conservative first-order kinetics.

Pools: n_min (mineral N), n_org (slow organic N), litter (surface residue),
soc. Every transfer moves mass between pools or exports it explicitly
(leaching, harvested yield) so budgets stay auditable.
"""

from __future__ import annotations

import torch

K_MIN = 2.5e-4      # /d max mineralization of n_org
K_LEACH = 0.6       # fraction of drained-water N fraction lost
K_LITTER = 3.0e-3   # /d max litter decay
K_SOC = 5.0e-5      # /d max SOC decay
HUMIFICATION = 0.35  # fraction of decayed litter stabilized into SOC
N_SUFFICIENT = 80.0  # kg/ha mineral N at which fixation down-regulates


def bio_factor(tmin: torch.Tensor, tmax: torch.Tensor,
               water_factor: torch.Tensor) -> torch.Tensor:
    """Microbial activity scalar ∈ [0,1]: linear to 30 °C × water supply."""
    ft = torch.clamp(0.5 * (tmin + tmax) / 30.0, 0.0, 1.0)
    return ft * water_factor


def nitrogen_step(
    n_min: torch.Tensor,
    n_org: torch.Tensor,
    bio: torch.Tensor,
    water_drain: torch.Tensor,
    water_store: torch.Tensor,
    n_demand: torch.Tensor,
    legume: torch.Tensor,
    fix_frac: torch.Tensor,
    n_fert: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Mineral/organic N update. Returns (n_min', n_org', fluxes)."""
    mineralized = n_org * K_MIN * bio
    n_org_a = n_org - mineralized

    scarcity = torch.clamp(1.0 - n_min / N_SUFFICIENT, 0.0, 1.0)
    fixed = n_demand * legume * fix_frac * scarcity
    uptake = torch.minimum(n_demand - fixed, torch.relu(n_min))
    leached = torch.minimum(
        n_min * K_LEACH * water_drain / torch.clamp(water_store, min=1.0),
        torch.relu(n_min - uptake),
    )

    n_min_a = n_min + mineralized + n_fert + fixed - uptake - leached
    fluxes = {"mineralized": mineralized, "fixed": fixed,
              "uptake": uptake, "leached": leached}
    return n_min_a, n_org_a, fluxes


def nitrogen_stress(n_min: torch.Tensor, n_half: float = 25.0) -> torch.Tensor:
    """Michaelis–Menten N sufficiency ∈ [0,1)."""
    return n_min / (n_min + n_half)


def carbon_step(
    litter: torch.Tensor,
    soc: torch.Tensor,
    bio: torch.Tensor,
    residue_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Litter/SOC update. Returns (litter', soc', fluxes)."""
    litter_a = litter + residue_in
    decayed = litter_a * K_LITTER * bio
    soc_decay = soc * K_SOC * bio
    soc_a = soc + HUMIFICATION * decayed - soc_decay
    litter_b = litter_a - decayed
    fluxes = {"litter_decay": decayed, "humified": HUMIFICATION * decayed,
              "soc_decay": soc_decay, "respired": (1.0 - HUMIFICATION) * decayed + soc_decay}
    return litter_b, soc_a, fluxes
