"""Canopy dynamics: thermal time, LAI, radiation-use-efficiency biomass.

Functional and fully batched — a "step" maps tensors to tensors; lifecycle
events (sowing, maturity, termination) are applied as masks by the caller
(`WorldModel.step`), never hidden inside the process equations.
"""

from __future__ import annotations

import torch

from nasa_farm.dynamics.atmosphere import growing_degree_days, temperature_factor
from nasa_farm.state.crops import IDX

PAR_FRACTION = 0.47
"""Photosynthetically active fraction of shortwave radiation."""


def traits(table: torch.Tensor, crop_idx: torch.Tensor) -> torch.Tensor:
    """(B, P) trait rows for active crops; bare fields get row 0 (masked out)."""
    return table[crop_idx.clamp(min=0)]


def leaf_area_index(gdd: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """LAI from accumulated GDD and trait row t (B, P)."""
    em, lm, sen_on, mat = (t[:, IDX[k]] for k in ("gdd_em", "gdd_lai_max", "gdd_sen", "gdd_mat"))
    grow = torch.clamp((gdd - em) / (lm - em).clamp(min=1.0), 0.0, 1.0)
    sen = torch.clamp((gdd - sen_on) / (mat - sen_on).clamp(min=1.0), 0.0, 1.0)
    lai_g = 1.0 - torch.exp(-3.0 * grow)
    return t[:, IDX["lai_max"]] * lai_g * (1.0 - sen)


def intercepted_fraction(lai: torch.Tensor, k_ext: torch.Tensor) -> torch.Tensor:
    """Beer–Lambert canopy light interception ∈ [0,1)."""
    return 1.0 - torch.exp(-k_ext * lai)


def canopy_step(
    gdd: torch.Tensor,
    biomass: torch.Tensor,
    tr: torch.Tensor,
    tmin: torch.Tensor,
    tmax: torch.Tensor,
    srad: torch.Tensor,
    water_fac: torch.Tensor,
    nitrogen_fac: torch.Tensor,
    growing: torch.Tensor,
    lai: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Advance phenology + biomass one day. Returns (gdd', lai', biomass', fluxes)."""
    active = growing.to(gdd.dtype)
    gdd_new = gdd + growing_degree_days(tmin, tmax, tr[:, IDX["tbase"]]) * active
    lai_new = leaf_area_index(gdd_new, tr) * active

    fpar = intercepted_fraction(lai_new, tr[:, IDX["k_ext"]])
    tf = temperature_factor(tmin, tmax, tr[:, IDX["tbase"]], tr[:, IDX["topt"]])
    # WOFOST-style: N limits canopy expansion, only ~40 % of it limits daily
    # assimilation directly.
    n_growth = 0.6 + 0.4 * nitrogen_fac
    limiter = torch.minimum(tf, torch.minimum(water_fac, n_growth))
    delta_b = tr[:, IDX["rue"]] * (PAR_FRACTION * srad) * fpar * limiter * active

    biomass_new = biomass + delta_b
    mature = (gdd_new >= tr[:, IDX["gdd_mat"]]) & growing
    fluxes = {"delta_b": delta_b, "fpar": fpar, "limiter": limiter, "mature": mature}
    return gdd_new, lai_new, biomass_new, fluxes


def harvest(biomass: torch.Tensor, tr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split standing biomass into yield and retained residue."""
    yield_kg = biomass * tr[:, IDX["harvest_index"]]
    residue = (biomass - yield_kg) * tr[:, IDX["residue_frac"]]
    return yield_kg, residue


def water_uptake_split(tr: torch.Tensor) -> torch.Tensor:
    return tr[:, IDX["root_frac2"]]


def potential_transpiration(et0: torch.Tensor, lai: torch.Tensor, tr: torch.Tensor,
                            growing: torch.Tensor) -> torch.Tensor:
    cover = intercepted_fraction(lai, tr[:, IDX["k_ext"]])
    return et0 * tr[:, IDX["kc_max"]] * cover * growing.to(et0.dtype)


def potential_evaporation(et0: torch.Tensor, lai: torch.Tensor, tr: torch.Tensor,
                          growing: torch.Tensor) -> torch.Tensor:
    cover = intercepted_fraction(lai, tr[:, IDX["k_ext"]])
    bare = 1.0 - cover * growing.to(et0.dtype)
    # 0.6 ≈ FAO-56 mean soil-evaporation coefficient over the stage-1+2 cycle
    return et0 * 0.6 * bare
