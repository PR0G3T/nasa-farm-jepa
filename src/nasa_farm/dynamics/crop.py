"""Canopy dynamics: thermal time, LAI, radiation-use-efficiency biomass.

Functional and fully batched — a "step" maps tensors to tensors; lifecycle
events (sowing, maturity, termination) are applied as masks by the caller
(`WorldModel.step`), never hidden inside the process equations.
"""

from __future__ import annotations

import torch

from nasa_farm.dynamics.atmosphere import growing_degree_days, temperature_factor
from nasa_farm.state.crops import INDEX

PAR_FRACTION = 0.47
"""Photosynthetically active fraction of shortwave radiation."""

GM2_TO_KG_HA = 10.0
"""RUE yields g DM/m²; state biomass is kg/ha (1 g/m² = 10 kg/ha)."""

FROST_SALVAGE = 0.3
"""Fraction of harvest index recovered when a lethal frost ends the cycle."""

VERN_T_LO, VERN_T_HI = -4.0, 8.0
"""Effective chilling window (°C tmean) for vernalization."""


def traits(table: torch.Tensor, crop_idx: torch.Tensor) -> torch.Tensor:
    """(B, P) trait rows for active crops; bare fields get row 0 (masked out)."""
    return table[crop_idx.clamp(min=0)]


def leaf_area_index(gdd: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """LAI from accumulated GDD and trait row t (B, P)."""
    em, lm, sen_on, mat = (t[:, INDEX[k]] for k in ("gdd_em", "gdd_lai_max", "gdd_sen", "gdd_mat"))
    grow = torch.clamp((gdd - em) / (lm - em).clamp(min=1.0), 0.0, 1.0)
    sen = torch.clamp((gdd - sen_on) / (mat - sen_on).clamp(min=1.0), 0.0, 1.0)
    lai_g = 1.0 - torch.exp(-3.0 * grow)
    return t[:, INDEX["lai_max"]] * lai_g * (1.0 - sen)


def intercepted_fraction(lai: torch.Tensor, k_ext: torch.Tensor) -> torch.Tensor:
    """Beer–Lambert canopy light interception ∈ [0,1)."""
    return 1.0 - torch.exp(-k_ext * lai)


def photoperiod_factor(daylen: torch.Tensor, traits: torch.Tensor) -> torch.Tensor:
    """Development rate scalar ∈ (0,1] — DSSAT-style: day length past the
    critical photoperiod delays progress proportionally to `pp_sens`."""
    delay = traits[:, INDEX["pp_sens"]] * torch.relu(daylen - traits[:, INDEX["crit_daylen"]])
    return 1.0 / (1.0 + delay)


def co2_factor(co2_ppm: torch.Tensor, traits: torch.Tensor, ref_ppm: float = 420.0) -> torch.Tensor:
    """RUE multiplier from atmospheric CO2 — power response `(co2/ref)^resp`.

    C4 crops saturate early (resp ≈ 0.05), C3 keep responding (≈ 0.2), the
    FACE-experiment elasticity range.
    """
    return torch.pow(torch.clamp(co2_ppm / ref_ppm, min=1e-3), traits[:, INDEX["co2_resp"]])


def _vern_update(vernalization: torch.Tensor, tmean: torch.Tensor,
                 growing: torch.Tensor) -> torch.Tensor:
    """Chill-day accumulation while growing; state preserved when fallow."""
    eff = ((tmean > VERN_T_LO) & (tmean < VERN_T_HI)).to(vernalization.dtype)
    return torch.where(growing, vernalization + eff, vernalization)


def _heat_update(heat: torch.Tensor, tmax: torch.Tensor, gdd_new: torch.Tensor,
                 traits: torch.Tensor, growing: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproductive-window heat degree-days → (heat_acc', daily increment)."""
    win_lo = traits[:, INDEX["gdd_fl"]] * traits[:, INDEX["gdd_mat"]]
    in_win = (gdd_new >= win_lo) & (gdd_new < traits[:, INDEX["gdd_mat"]]) & growing
    inc = torch.relu(tmax - traits[:, INDEX["heat_t"]]) * in_win.to(heat.dtype)
    return heat + inc, inc


def canopy_step(
    gdd: torch.Tensor,
    vernalization: torch.Tensor,
    heat_acc: torch.Tensor,
    biomass: torch.Tensor,
    traits: torch.Tensor,
    tmin: torch.Tensor,
    tmax: torch.Tensor,
    srad: torch.Tensor,
    daylen: torch.Tensor | None,
    co2: torch.Tensor | None,
    water_fac: torch.Tensor,
    nitrogen_fac: torch.Tensor,
    growing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Advance phenology + biomass one day. Returns (gdd', lai', biomass', fluxes).

    fluxes carries the updated cycle trackers (`vern_new`, `heat_new`) plus
    the event masks `mature` (natural) and `frost` (lethal cold kill).
    `daylen`/`co2` = None neutralize photoperiod / CO2 response respectively.
    """
    active = growing.to(gdd.dtype)
    tmean = 0.5 * (tmin + tmax)

    vern_new = _vern_update(vernalization, tmean, growing)
    pp = 1.0 if daylen is None else photoperiod_factor(daylen, traits)
    gdd_new = gdd + growing_degree_days(tmin, tmax, traits[:, INDEX["tbase"]]) * pp * active

    # Unvernalized crops stall at the reproductive threshold (winter cereals).
    request = traits[:, INDEX["vern_req"]]
    gate = (request > 0) & (vern_new < request)
    gdd_new = torch.where(
        gate, torch.minimum(gdd_new, traits[:, INDEX["gdd_fl"]] * traits[:, INDEX["gdd_mat"]]),
        gdd_new)

    lai_new = leaf_area_index(gdd_new, traits) * active
    heat_new, heat_inc = _heat_update(heat_acc, tmax, gdd_new, traits, growing)

    fpar = intercepted_fraction(lai_new, traits[:, INDEX["k_ext"]])
    tf = temperature_factor(tmin, tmax, traits[:, INDEX["tbase"]], traits[:, INDEX["topt"]])
    # WOFOST-style: N limits canopy expansion, only ~40 % of it limits daily
    # assimilation directly.
    n_growth = 0.6 + 0.4 * nitrogen_fac
    co2f = 1.0 if co2 is None else co2_factor(co2, traits)
    limiter = torch.minimum(tf, torch.minimum(water_fac, n_growth))
    delta_b = (traits[:, INDEX["rue"]] * (PAR_FRACTION * srad) * fpar
               * limiter * co2f * active * GM2_TO_KG_HA)

    biomass_new = biomass + delta_b
    frost = growing & (tmin < traits[:, INDEX["t_kill"]])
    mature = ((gdd_new >= traits[:, INDEX["gdd_mat"]]) | frost) & growing
    fluxes = {"delta_b": delta_b, "fpar": fpar, "limiter": limiter, "mature": mature,
              "frost": frost, "heat_dd": heat_inc,
              "vern_new": vern_new, "heat_new": heat_new}
    return gdd_new, lai_new, biomass_new, fluxes


def harvest(biomass: torch.Tensor, traits: torch.Tensor,
            heat_acc: torch.Tensor | None = None,
            frost: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Split standing biomass into yield and retained residue.

    Reproductive heat degrades the harvest index exponentially
    (`heat_sus` per °C·d); lethal frost salvages only FROST_SALVAGE of it.
    """
    high = traits[:, INDEX["harvest_index"]]
    if heat_acc is not None:
        high = high * torch.exp(-traits[:, INDEX["heat_sus"]] * heat_acc)
    if frost is not None:
        high = torch.where(frost, high * FROST_SALVAGE, high)
    yield_kg = biomass * high
    residue = (biomass - yield_kg) * traits[:, INDEX["residue_frac"]]
    return yield_kg, residue


def water_uptake_split(traits: torch.Tensor) -> torch.Tensor:
    return traits[:, INDEX["root_frac2"]]


def potential_transpiration(et0: torch.Tensor, lai: torch.Tensor, traits: torch.Tensor,
                            growing: torch.Tensor) -> torch.Tensor:
    cover = intercepted_fraction(lai, traits[:, INDEX["k_ext"]])
    return et0 * traits[:, INDEX["kc_max"]] * cover * growing.to(et0.dtype)


def potential_evaporation(et0: torch.Tensor, lai: torch.Tensor, traits: torch.Tensor,
                          growing: torch.Tensor) -> torch.Tensor:
    cover = intercepted_fraction(lai, traits[:, INDEX["k_ext"]])
    bare = 1.0 - cover * growing.to(et0.dtype)
    # 0.6 ≈ FAO-56 mean soil-evaporation coefficient over the stage-1+2 cycle
    return et0 * 0.6 * bare
