"""WorldModel — differentiable latent dynamics of a field.

step(field, action, forcing) composes: atmosphere → canopy → water balance →
N/C pools → lifecycle events → optional learned residual. Operator splitting
uses pre-step stress factors for growth; every subprocess is conservative
so the composite step inherits auditable budgets.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics import biogeochemistry as bg
from nasa_farm.dynamics import crop as cp
from nasa_farm.dynamics import water
from nasa_farm.dynamics.atmosphere import (
    et0_penman_monteith, extraterrestrial_radiation,
)
from nasa_farm.dynamics.residual import RESIDUAL_VARS, ResidualNet, featurize
from nasa_farm.state.crops import IDX, CropLibrary
from nasa_farm.state.field import Field, FieldState
from nasa_farm.types import Actions, Forcing


class WorldModel(nn.Module):
    """Latent dynamics. `step`/`rollout` are pure w.r.t. the passed field."""

    def __init__(self, crops: CropLibrary | None = None, residual: bool = True) -> None:
        super().__init__()
        self.crops = crops or CropLibrary.default()
        self.register_buffer("crop_table", self.crops.table)
        self.residual = ResidualNet() if residual else None

    def enable_compile(self, **compile_kw) -> "WorldModel":
        """JIT-compile `step` via torch.compile — ~2× on repeated rollouts.

        Opt-in because the first call pays a ~10 s graph build; worth it for
        planning sweeps and training loops, not for a single short rollout.
        """
        self.step = torch.compile(self.step, **compile_kw)  # type: ignore[method-assign]
        return self

    def _apply_residual(self, state: FieldState, soil, forcing_d: dict, action_d: dict) -> FieldState:
        feats = featurize(
            {**{k: getattr(state, k) for k in
                ("sw1", "sw2", "n_min", "lai", "biomass", "gdd", "growing")}},
            forcing_d, action_d,
        )
        delta = self.residual(feats)
        upd = dict(zip(RESIDUAL_VARS, delta.unbind(-1)))
        sw1 = (state.sw1 + upd["sw1"]).clamp(soil.layer1.theta_air, soil.layer1.theta_sat)
        sw2 = (state.sw2 + upd["sw2"]).clamp(min=0.0)
        sw2 = torch.minimum(sw2, soil.layer2.theta_sat)
        return state.replace(
            sw1=sw1, sw2=sw2,
            n_min=(state.n_min + upd["n_min"]).clamp(min=0.0),
            lai=(state.lai + upd["lai"]).clamp(min=0.0),
            biomass=(state.biomass + upd["biomass"]).clamp(min=0.0),
        )

    def step(self, field: Field, action: dict, forcing_d: dict) -> tuple[Field, dict]:
        s = field.state
        tr = cp.traits(self.crop_table, s.crop)

        ra = extraterrestrial_radiation(field.lat, forcing_d["doy"])
        et0 = et0_penman_monteith(
            forcing_d["tmin"], forcing_d["tmax"], forcing_d["rh"],
            forcing_d["u2"], forcing_d["srad"], ra, field.elev_or_default())

        water_fac = water.water_uptake_factor(s.sw1, s.sw2, field.soil)
        pawf = water.paw_fraction(s.sw1, s.sw2, field.soil)
        n_fac = bg.nitrogen_stress(s.n_min)

        gdd, lai, biomass, cflux = cp.canopy_step(
            s.gdd, s.biomass, tr, forcing_d["tmin"], forcing_d["tmax"], forcing_d["srad"],
            water_fac, n_fac, s.growing, s.lai,
        )

        ep = cp.potential_evaporation(et0, lai, tr, s.growing)
        tp = cp.potential_transpiration(et0, lai, tr, s.growing)
        infiltration = forcing_d["precip"] + action["irrigation"]
        sw1, sw2, wflux = water.water_balance_step(
            s.sw1, s.sw2, field.soil, infiltration, ep, tp,
            cp.water_uptake_split(tr))

        bio = bg.bio_factor(forcing_d["tmin"], forcing_d["tmax"], pawf)
        n_demand = tr[:, IDX["n_demand"]] * cflux["delta_b"]
        n_min, n_org, nflux = bg.nitrogen_step(
            s.n_min, s.n_org, bio, wflux["perc"] + wflux["deep"], sw1 + sw2,
            n_demand, tr[:, IDX["legume"]], tr[:, IDX["fix_frac"]], action["n_fert"])
        litter, soc, c2flux = bg.carbon_step(s.litter, s.soc, bio, torch.zeros_like(s.litter))

        # Lifecycle events ------------------------------------------------
        yield_event = torch.zeros_like(s.biomass)
        residue_in = torch.zeros_like(s.biomass)

        mature = cflux["mature"]
        if mature.any():
            y, r = cp.harvest(biomass, tr)
            yield_event = torch.where(mature, y, yield_event)
            residue_in = torch.where(mature, r, residue_in)
            biomass = torch.where(mature, torch.zeros_like(biomass), biomass)
            lai = torch.where(mature, torch.zeros_like(lai), lai)

        growing = s.growing & ~mature
        harvested = s.harvested | mature

        sow = action["sow"]
        sowing = sow >= 0
        if sowing.any():
            # Terminating existing standing biomass (cover crop / perennial /
            # immature crop) counts as harvest; retained fraction → litter.
            y_t, r_t = cp.harvest(biomass, tr)
            yield_event = torch.where(sowing, yield_event + y_t, yield_event)
            residue_in = torch.where(sowing, residue_in + r_t, residue_in)
            zero = torch.zeros_like(biomass)
            biomass = torch.where(sowing, zero, biomass)
            lai = torch.where(sowing, zero, lai)
            gdd = torch.where(sowing, zero, gdd)
            crop = torch.where(sowing, sow, s.crop)
            growing = growing | sowing
            harvested = torch.where(sowing, torch.zeros_like(harvested), harvested)
        else:
            crop = s.crop

        # season_yield counts grain of the current crop cycle; the termination
        # yield emitted on a sowing day still lands in cum_yield and fluxes.
        season_yield = torch.where(sowing, torch.zeros_like(s.season_yield),
                                   s.season_yield + yield_event)
        cum_yield = s.cum_yield + yield_event
        litter = litter + residue_in

        new_state = FieldState(
            sw1=sw1, sw2=sw2, n_min=n_min, n_org=n_org, soc=soc, litter=litter,
            gdd=gdd, lai=lai, biomass=biomass,
            season_yield=season_yield, cum_yield=cum_yield,
            crop=crop, growing=growing, harvested=harvested,
        )

        if self.residual is not None:
            new_state = self._apply_residual(new_state, field.soil, forcing_d, action)

        fluxes = {**wflux, **nflux, **c2flux,
                  "et0": et0, "delta_b": cflux["delta_b"], "limiter": cflux["limiter"],
                  "water_fac": water_fac, "yield_event": yield_event,
                  "residue_in": residue_in}
        return field.__class__(state=new_state, soil=field.soil, lat=field.lat), fluxes

    def rollout(
        self,
        field: Field,
        forcing: Forcing,
        actions: Actions,
        observer=None,
    ) -> Field:
        """Advance `forcing.days` steps. `observer(t, state, fluxes)` is called
        each day — pass an accumulator to avoid materializing trajectories."""
        for t in range(forcing.days):
            field, fluxes = self.step(field, actions.day(t), forcing.day(t))
            if observer is not None:
                observer(t, field.state, fluxes)
        return field
