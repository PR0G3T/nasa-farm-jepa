"""Extra processes — injectable truth-side dynamics.

Each entry is a small conservative operator applied inside `WorldModel.step`
only when the model was built with `extra=(...)`. The twin scene turns them
on for the nature model alone, producing a genuine structural gap (missing
processes) that the residual, latent and SINDy stacks must absorb — the
OSSE protocol's "different model class as truth".

All transfers conserve mass: water moves between stores, senescent biomass
moves to litter, immobilized N moves to the organic pool, denitrified N is
counted in n2o. Nothing is created or destroyed.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from nasa_farm.state.field import FieldState
from nasa_farm.state.soil import SoilProfile

Extra = Callable[[FieldState, SoilProfile, dict, dict, dict], FieldState]


def waterlog(s: FieldState, soil: SoilProfile, f: dict, a: dict,
             fl: dict) -> FieldState:
    """Surface hypoxia: saturated topsoil suffocates the crop and denitrifies."""
    sat0 = soil.sat[:, 0]
    excess = torch.relu(s.sw[:, 0] - (soil.fc[:, 0] + 0.15 * (sat0 - soil.fc[:, 0]))) \
        / (sat0 - soil.fc[:, 0]).clamp(min=1.0)
    d_b = (0.02 * excess * s.biomass * s.growing).clamp(min=0.0)
    d_n = (0.05 * excess * s.n_min).clamp(min=0.0)
    return s.replace(biomass=s.biomass - d_b, litter=s.litter + d_b,
                     n_min=s.n_min - d_n, n2o=s.n2o + d_n)


def heat_shock(s: FieldState, soil: SoilProfile, f: dict, a: dict,
               fl: dict) -> FieldState:
    """Heat-wave senescence: extreme tmax sheds leaf and biomass to litter."""
    excess = torch.relu(f["tmax"] - 34.0) / 6.0
    d_b = (0.05 * excess * s.biomass * s.growing).clamp(min=0.0)
    fraction = (d_b / s.biomass.clamp(min=1.0)).clamp(max=1.0)
    return s.replace(biomass=s.biomass - d_b, litter=s.litter + d_b,
                     lai=s.lai * (1.0 - fraction))


def n_immob(s: FieldState, soil: SoilProfile, f: dict, a: dict,
            fl: dict) -> FieldState:
    """Residue immobilization: heavy rain on litter locks mineral N organic."""
    rain = torch.relu(f["precip"] - 25.0) / 25.0
    d = (0.02 * rain * (s.litter / 5000.0).clamp(max=1.0) * s.n_min).clamp(min=0.0)
    return s.replace(n_min=s.n_min - d, n_org=s.n_org + d)


def crust(s: FieldState, soil: SoilProfile, f: dict, a: dict,
          fl: dict) -> FieldState:
    """Sealed surface: surplus water above field capacity bypasses to storage."""
    d = (0.25 * torch.relu(s.sw[:, 0] - soil.fc[:, 0])).clamp(min=0.0)
    sw = s.sw.clone()
    sw[:, 0] = sw[:, 0] - d
    return s.replace(sw=sw, tws=s.tws + d)


EXTRAS: dict[str, Extra] = {
    "waterlog": waterlog, "heat_shock": heat_shock,
    "n_immob": n_immob, "crust": crust,
}


def step_extra(names: tuple[str, ...], state: FieldState, soil: SoilProfile,
               forcing_day: dict, action_d: dict, fluxes: dict) -> FieldState:
    """Apply the named extra processes, in order."""
    for n in names:
        state = EXTRAS[n](state, soil, forcing_day, action_d, fluxes)
    return state
