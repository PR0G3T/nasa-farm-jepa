"""Nitrogen and carbon pools — conservative first-order kinetics.

Pools: n_min (mineral N), n_org (slow organic N), litter (surface residue),
soc_f (active SOC, ~yr turnover), soc_s (stable SOC, ~decadal). Every
transfer moves mass between pools or exports it explicitly (leaching,
denitrification, residue removal, harvest) so budgets stay auditable.

Denitrification follows the standard hot-moment scheme: a rate scaled by
WFPS² (filled pore space → anaerobicity) × bio factor on mineral N; a fixed
fraction is emitted as N2O-N, the rest leaves as N2.
"""

from __future__ import annotations

import torch

K_MIN = 2.5e-4       # /d max mineralization of n_org
K_LEACH = 0.6        # fraction of drained-water N fraction lost
K_LITTER = 3.0e-3    # /d max litter decay
K_SOC_F = 1.5e-4     # /d max active-pool decay
K_SOC_S = 1.0e-5     # /d max stable-pool decay
STAB = 0.08          # fraction of active-pool decay stabilized into soc_s
HUMIFICATION = 0.35  # fraction of decayed litter stabilized into soc_f
N_SUFFICIENT = 80.0  # kg/ha mineral N at which fixation down-regulates
K_DENIT = 2.0e-3     # /d max denitrification on n_min at full saturation
N2O_FRAC = 0.01      # fraction of denitrified N emitted as N2O-N
TILL_MIX = 0.6       # fraction of surface litter incorporated by full tillage
TILL_BIO = 0.5       # microbial-activity boost on a full-tillage day


def bio_factor(tmin: torch.Tensor, tmax: torch.Tensor,
               water_factor: torch.Tensor) -> torch.Tensor:
    """Microbial activity scalar ∈ [0,1]: linear to 30 °C × water supply."""
    ft = torch.clamp(0.5 * (tmin + tmax) / 30.0, 0.0, 1.0)
    return ft * water_factor


def wfps(sw_total: torch.Tensor, sat_total: torch.Tensor) -> torch.Tensor:
    """Water-filled pore space ∈ [0,1] — the denitrification driver."""
    return torch.clamp(sw_total / sat_total.clamp(min=1.0), 0.0, 1.0)


def nitrogen_step(
    n_min: torch.Tensor,
    n_org: torch.Tensor,
    biology: torch.Tensor,
    water_drain: torch.Tensor,
    water_store: torch.Tensor,
    n_demand: torch.Tensor,
    legume: torch.Tensor,
    fix_frac: torch.Tensor,
    n_fert: torch.Tensor,
    w: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Mineral/organic N update. `w` = WFPS driving denitrification.
    Returns (n_min', n_org', fluxes)."""
    mineralized = n_org * K_MIN * biology
    n_org_a = n_org - mineralized

    scarcity = torch.clamp(1.0 - n_min / N_SUFFICIENT, 0.0, 1.0)
    fixed = n_demand * legume * fix_frac * scarcity
    n_pre = n_min + mineralized + n_fert + fixed
    uptake = torch.minimum(n_demand - fixed, torch.relu(n_pre))
    leached = torch.minimum(
        n_pre * K_LEACH * water_drain / torch.clamp(water_store, min=1.0),
        torch.relu(n_pre - uptake),
    )
    denit = (n_pre - uptake - leached).clamp(min=0.0) * K_DENIT * (w ** 2) * biology \
        if not isinstance(w, float) else n_pre * 0.0
    n2o = denit * N2O_FRAC

    n_min_a = n_pre - uptake - leached - denit
    fluxes = {"mineralized": mineralized, "fixed": fixed,
              "uptake": uptake, "leached": leached,
              "denit": denit, "n2o": n2o}
    return n_min_a, n_org_a, fluxes


def nitrogen_stress(n_min: torch.Tensor, n_half: float = 25.0) -> torch.Tensor:
    """Michaelis–Menten N sufficiency ∈ [0,1)."""
    return n_min / (n_min + n_half)


def carbon_step(
    litter: torch.Tensor,
    soc_f: torch.Tensor,
    soc_s: torch.Tensor,
    biology: torch.Tensor,
    residue_in: torch.Tensor,
    till: torch.Tensor | None = None,
    res_remove: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Litter + two-pool SOC update. Returns (litter', soc_f', soc_s', fluxes).

    Budget: Δ(litter+soc_f+soc_s) ≡ residue_in − exported − respired.
    Tillage incorporates surface litter into the active pool and transiently
    boosts microbial activity — the classic tillage SOC-decline mechanism.
    """
    zero = torch.zeros_like(litter)
    till = zero if till is None else till.clamp(0.0, 1.0)
    res_remove = zero if res_remove is None else res_remove.clamp(0.0, 1.0)

    litter_a = litter + residue_in
    exported = litter_a * res_remove
    litter_a = litter_a - exported

    incorporated = litter_a * TILL_MIX * till
    bio_eff = torch.minimum(biology * (1.0 + TILL_BIO * till), torch.ones_like(biology))
    litter_a = litter_a - incorporated
    soc_f_a = soc_f + incorporated

    decayed = litter_a * K_LITTER * bio_eff
    litter_b = litter_a - decayed
    decay_f = soc_f_a * K_SOC_F * bio_eff
    decay_s = soc_s * K_SOC_S * bio_eff

    soc_f_b = soc_f_a + HUMIFICATION * decayed - decay_f
    soc_s_b = soc_s + STAB * decay_f - decay_s
    respired = ((1.0 - HUMIFICATION) * decayed
                + (1.0 - STAB) * decay_f + decay_s)
    fluxes = {"litter_decay": decayed, "humified": HUMIFICATION * decayed,
              "soc_f_decay": decay_f, "soc_s_decay": decay_s,
              "stabilized": STAB * decay_f, "exported": exported,
              "incorporated": incorporated, "respired": respired}
    return litter_b, soc_f_b, soc_s_b, fluxes
