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
    day_length, et0_penman_monteith, extraterrestrial_radiation,
)
from nasa_farm.dynamics.latent import LatentDyn
from nasa_farm.dynamics.residual import RESIDUAL_VARS, ResidualNet, featurize
from nasa_farm.state.crops import IDX, CropLibrary
from nasa_farm.state.field import Field, FieldState
from nasa_farm.types import Actions, Forcing


class WorldModel(nn.Module):
    """Latent dynamics. `step`/`rollout` are pure w.r.t. the passed field."""

    def __init__(self, crops: CropLibrary | None = None, residual: bool = True,
                 latent_dim: int = 0) -> None:
        super().__init__()
        self.crops = crops or CropLibrary.default()
        self.register_buffer("crop_table", self.crops.table)
        self.residual = ResidualNet() if residual else None
        self.latent = LatentDyn(latent_dim) if latent_dim else None

    def enable_compile(self, **compile_kw) -> "WorldModel":
        """JIT-compile `step` via torch.compile — ~2× on repeated rollouts.

        Opt-in because the first call pays a ~10 s graph build; worth it for
        planning sweeps and training loops, not for a single short rollout.
        """
        self.step = torch.compile(self.step, **compile_kw)  # type: ignore[method-assign]
        return self

    def _features(self, state: FieldState, forcing_d: dict, action_d: dict) -> torch.Tensor:
        return featurize(
            {**{k: getattr(state, k) for k in
                ("sw1", "sw2", "n_min", "lai", "biomass", "gdd", "growing")}},
            forcing_d, action_d,
        )

    def _apply_residual(self, state: FieldState, soil, feats: torch.Tensor) -> FieldState:
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
        daylen = day_length(field.lat, forcing_d["doy"])

        water_fac = water.water_uptake_factor(s.sw1, s.sw2, field.soil)
        pawf = water.paw_fraction(s.sw1, s.sw2, field.soil)
        n_fac = bg.nitrogen_stress(s.n_min)

        gdd, lai, biomass, cflux = cp.canopy_step(
            s.gdd, s.vern, s.heat_acc, s.biomass, tr,
            forcing_d["tmin"], forcing_d["tmax"], forcing_d["srad"],
            daylen, forcing_d.get("co2"),
            water_fac, n_fac, s.growing, s.lai,
        )
        vern, heat_acc = cflux["vern_new"], cflux["heat_new"]

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
            y, r = cp.harvest(biomass, tr, heat_acc, cflux["frost"])
            yield_event = torch.where(mature, y, yield_event)
            residue_in = torch.where(mature, r, residue_in)
            biomass = torch.where(mature, torch.zeros_like(biomass), biomass)
            lai = torch.where(mature, torch.zeros_like(lai), lai)

        growing = s.growing & ~mature
        harvested = s.harvested | mature

        sow = action["sow"]
        sowing = sow >= 0
        z = s.z
        if sowing.any():
            # Terminating existing standing biomass (cover crop / perennial /
            # immature crop) counts as harvest; retained fraction → litter.
            y_t, r_t = cp.harvest(biomass, tr, heat_acc)
            yield_event = torch.where(sowing, yield_event + y_t, yield_event)
            residue_in = torch.where(sowing, residue_in + r_t, residue_in)
            zero = torch.zeros_like(biomass)
            biomass = torch.where(sowing, zero, biomass)
            lai = torch.where(sowing, zero, lai)
            gdd = torch.where(sowing, zero, gdd)
            vern = torch.where(sowing, zero, vern)
            heat_acc = torch.where(sowing, zero, heat_acc)
            if z is not None:
                z = torch.where(sowing.unsqueeze(-1), torch.zeros_like(z), z)
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
            vern=vern, heat_acc=heat_acc, z=z,
        )

        if self.residual is not None or (self.latent is not None and z is not None):
            feats = self._features(new_state, forcing_d, action)
            if self.residual is not None:
                new_state = self._apply_residual(new_state, field.soil, feats)
            if self.latent is not None and z is not None:
                new_state = new_state.replace(
                    z=self.latent(feats, new_state.z))

        fluxes = {**wflux, **nflux, **c2flux,
                  "et0": et0, "delta_b": cflux["delta_b"], "limiter": cflux["limiter"],
                  "water_fac": water_fac, "yield_event": yield_event,
                  "residue_in": residue_in,
                  "heat_dd": cflux["heat_dd"], "frost": cflux["frost"].to(DTYPE)}
        return field.__class__(state=new_state, soil=field.soil, lat=field.lat,
                               elev=field.elev), fluxes

    def rollout(
        self,
        field: Field,
        forcing: Forcing,
        actions: Actions,
        observer=None,
        checkpoint_every: int = 0,
    ) -> Field:
        """Advance `forcing.days` steps. `observer(t, state, fluxes)` is called
        each day — pass an accumulator to avoid materializing trajectories.

        checkpoint_every > 0 (grad mode): each segment recomputes its days on
        the backward pass instead of storing them — rollout activation memory
        drops to O(days/k + k) for identical gradients. Only SeasonAccumulator
        observers are allowed (their fold is pure-functional).
        """
        ckpt = (checkpoint_every if torch.is_grad_enabled() else 0)
        if ckpt and not any(t.requires_grad for t in
                            (field.state.sw1, forcing.tmin, *self.parameters())):
            ckpt = 0  # no backward possible → no activation-memory win
        if not ckpt:
            for t in range(forcing.days):
                field, fluxes = self.step(field, actions.day(t), forcing.day(t))
                if observer is not None:
                    observer(t, field.state, fluxes)
            return field
        return self._ckpt_rollout(field, forcing, actions, observer, ckpt)

    def _ckpt_rollout(self, field: Field, forcing: Forcing, actions: Actions,
                      observer, every: int) -> Field:
        from torch.utils.checkpoint import checkpoint

        from nasa_farm.heads import SeasonAccumulator, day_update, fold
        from nasa_farm.types import FORCING_VARS

        if observer is not None and not isinstance(observer, SeasonAccumulator):
            raise ValueError(
                "checkpointed rollout supports only SeasonAccumulator observers")
        names = [n for n in field.state.__dataclass_fields__
                 if getattr(field.state, n) is not None]
        acc = observer.acc if observer is not None else None
        static = {"soil": field.soil, "lat": field.lat, "elev": field.elev}
        cls = field.__class__
        avars = ("sow", "irrigation", "n_fert")
        nf, na = len(FORCING_VARS), len(avars)

        for t0 in range(0, forcing.days, every):
            t1 = min(t0 + every, forcing.days)
            # forcing/actions are explicit checkpoint inputs so gradients
            # cross segment boundaries through the recompute.
            ext = [getattr(forcing, k)[:, t0:t1] for k in FORCING_VARS] \
                + [getattr(actions, k)[:, t0:t1] for k in avars]

            # t0/t1 bound as keyword defaults: recompute happens after the
            # loop ends, when the loop vars would all read their last values.
            def seg(*args, _t0=t0, _t1=t1):
                off = len(names)
                a = args[off] if acc is not None else None
                e = args[off + (acc is not None):]
                f = cls(state=FieldState(**dict(zip(names, args[:off]))),
                        **static)
                for t in range(_t1 - _t0):
                    fd = {k: e[j][:, t] for j, k in enumerate(FORCING_VARS)}
                    ad = {k: e[nf + j][:, t] for j, k in enumerate(avars)}
                    f, fl = self.step(f, ad, fd)
                    if a is not None:
                        a = fold(a, _t0 + t, day_update(f.state, fl))
                tail = () if a is None else (a,)
                return tuple(getattr(f.state, n) for n in names) + tail

            out = checkpoint(
                seg, *[getattr(field.state, n) for n in names],
                *(() if acc is None else (acc,)), *ext,
                use_reentrant=True)
            field = cls(state=FieldState(**dict(zip(names, out[:len(names)]))),
                        **static)
            if acc is not None:
                acc = out[-1]
        if observer is not None:
            observer.acc = acc
        return field
