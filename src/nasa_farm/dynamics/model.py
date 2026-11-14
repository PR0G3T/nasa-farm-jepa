"""WorldModel — differentiable latent dynamics of a field.

step(field, action, forcing) composes: atmosphere → canopy → N-layer water
balance → N/C pools (incl. denitrification/N2O) → lifecycle events →
optional learned residual and recurrent latent predictor. Operator
splitting uses pre-step stress factors for growth; every subprocess is
conservative so the composite step inherits auditable budgets.

The aquifer store `tws` absorbs deep drainage (recharge) and drains with a
slow first-order discharge — it is the GRACE-observable total-water-storage
anomaly and the coupling variable for landscape routing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics import biogeochemistry as bg
from nasa_farm.dynamics import crop as physiology
from nasa_farm.dynamics import water
from nasa_farm.dynamics.atmosphere import (
    day_length, et0_penman_monteith, extraterrestrial_radiation,
)
from nasa_farm.dynamics.extra import EXTRAS, step_extra
from nasa_farm.dynamics.hierarchical import Hierarchy
from nasa_farm.dynamics.latent import OPS, Factor, Dynamics
from nasa_farm.dynamics.residual import RESIDUAL_VARS, Residual, featurize
from nasa_farm.state.crops import INDEX, CropLibrary
from nasa_farm.state.field import Field, FieldState
from nasa_farm.state.soil import root_weights
from nasa_farm.types import ACTION_VARS, Actions, Forcing

K_TWS = 2.0e-3
"""/d aquifer discharge rate (≈500 d e-folding, baseflow recession)."""


class WorldModel(nn.Module):
    """Latent dynamics. `step`/`rollout` are pure w.r.t. the passed field."""

    def __init__(self, crops: CropLibrary | None = None, residual: bool = True,
                 latent_dimension: int = 0, latent_hidden: int = 64,
                 stochastic: bool = False, context_dimension: int = 0,
                 latent_strides: tuple[int, ...] = (),
                 latent_factors: dict[str, tuple[int, ...]] | str | None = None,
                 link: str = "identity",
                 extra: tuple[str, ...] = (),
                 diffusion: bool = False,
                 nn_dtype: torch.dtype = DTYPE,
                 seed: int | None = None) -> None:
        super().__init__()
        self.crops = crops or CropLibrary.default()
        self.register_buffer("crop_table", self.crops.table)
        self.nn_dtype = nn_dtype
        # Variety embeddings (G×E): a learnable per-crop vector appended to
        # the latent ctx each step, so cultivar identity conditions residual
        # dynamics. Kept out of crop_table — it is a model parameter.
        self.var_emb = (nn.Parameter(self.crops.var.clone())
                        if self.crops.var is not None else None)
        parts = (OPS if latent_factors == "ops" else latent_factors) or None
        z_dim = (latent_dimension * len(parts) if parts
                 else latent_dimension * max(len(latent_strides), 1))
        lctx = context_dimension + (self.var_emb.shape[1] if self.var_emb is not None
                          else 0)
        for n in extra:
            if n not in EXTRAS:
                raise KeyError(f"unknown extra process: {n}")
        self.extra = tuple(extra)
        # latent z conditions the residual: what the physics does not model
        # gets to correct the physics — the physics-anchored JEPA contract.
        # diffusion=True: the residual also emits process-noise variances,
        # sampled with self._gen — learned aleatoric model uncertainty on
        # top of forcing scenarios (neural-SDE layer).
        self.residual = (Residual(context_dimension=z_dim, diffusion=diffusion)
                         if residual else None)
        self.latent = None
        if latent_dimension:
            if parts:
                self.latent = Factor(latent_dimension, parts, latent_hidden, lctx)
            elif len(latent_strides) > 1:
                self.latent = Hierarchy(latent_dimension, latent_hidden,
                                      latent_strides, stochastic, lctx)
            else:
                self.latent = Dynamics(latent_dimension, latent_hidden,
                                        stochastic, lctx, link=link)
        self._gen = (torch.Generator().manual_seed(seed)
                     if seed is not None else None)
        if nn_dtype != DTYPE:
            self.quantize(nn_dtype)

    def quantize(self, dtype: torch.dtype) -> "WorldModel":
        """Run the learned nets in a narrower dtype; physics stays float64 —
        residual/latent outputs are cast back before touching state."""
        self.nn_dtype = dtype
        for net in (self.residual, self.latent):
            if net is not None:
                net.to(dtype)
        return self

    def enable_compile(self, **compile_kw) -> "WorldModel":
        """JIT-compile `step` via torch.compile — ~2× on repeated rollouts.

        Opt-in because the first call pays a ~10 s graph build; worth it for
        planning sweeps and training loops, not for a single short rollout.
        """
        self.step = torch.compile(self.step, **compile_kw)  # type: ignore[method-assign]
        return self

    def _nn(self, net: nn.Module, feats: torch.Tensor, *arguments):
        arguments = tuple(a.to(self.nn_dtype) if isinstance(a, torch.Tensor) else a
                     for a in arguments)
        output = net(feats.to(self.nn_dtype), *arguments)
        if isinstance(output, tuple):
            return tuple(o.to(DTYPE) for o in output)
        return output.to(DTYPE)

    def _features(self, state: FieldState, forcing_day: dict, action_d: dict) -> torch.Tensor:
        return featurize(
            {"sw0": state.sw[:, 0], "swr": state.sw[:, 1:].sum(-1),
             **{k: getattr(state, k) for k in
                ("n_min", "lai", "biomass", "gdd", "growing")}},
            forcing_day, action_d,
        )

    def _apply_residual(self, state: FieldState, soil, feats: torch.Tensor) -> FieldState:
        output = self._nn(self.residual, feats, state.z)
        if self.residual.diffusion:
            delta, logvar = output
            if self._gen is not None:
                signal = (0.5 * logvar).exp() * self.residual.scale.to(DTYPE)
                delta = delta + signal * torch.randn(
                    delta.shape, generator=self._gen, dtype=DTYPE,
                    device=delta.device)
        else:
            delta = output
        update = dict(zip(RESIDUAL_VARS, delta.unbind(-1)))
        sw0 = (state.sw[:, 0] + update["sw0"]).clamp(
            soil.air[:, 0], soil.sat[:, 0])
        if state.n_layers > 1:
            root = state.sw[:, 1:]
            w = root / root.sum(-1, keepdim=True).clamp(min=1e-9)
            root = torch.minimum(
                (root + update["swr"].unsqueeze(-1) * w).clamp(min=0.0),
                soil.sat[:, 1:])
            sw = torch.cat([sw0.unsqueeze(1), root], dim=1)
        else:
            sw = sw0.unsqueeze(1)
        return state.replace(
            sw=sw,
            n_min=(state.n_min + update["n_min"]).clamp(min=0.0),
            lai=(state.lai + update["lai"]).clamp(min=0.0),
            biomass=(state.biomass + update["biomass"]).clamp(min=0.0),
        )

    def _dctx(self, field: Field) -> torch.Tensor | None:
        """Dynamic conditioning ctx = static site ctx + variety embedding
        of the currently sown crop — cultivar/G×E identity reaches the
        latent the day sowing changes crop."""
        if self.var_emb is None:
            return field.context
        v = self.var_emb[field.state.crop.clamp(min=0)]
        return v if field.context is None else torch.cat([field.context, v], -1)

    def step(self, field: Field, action: dict, forcing_day: dict,
             t: int = 0) -> tuple[Field, dict]:
        s = field.state
        edited = forcing_day["tmin"].shape[0]
        if s.batch == 1 and edited > 1:
            # broadcast semantics: a single field under an ensemble of forcing
            from dataclasses import replace
            field = replace(
                field, state=s.repeat(edited),
                soil=field.soil.repeat(edited) if field.soil.batch == 1
                else field.soil)
            s = field.state
        traits = physiology.traits(self.crop_table, s.crop)

        ra = extraterrestrial_radiation(field.latitude, forcing_day["doy"])
        et0 = et0_penman_monteith(
            forcing_day["tmin"], forcing_day["tmax"], forcing_day["rh"],
            forcing_day["u2"], forcing_day["srad"], ra, field.elev_or_default())
        daylen = day_length(field.latitude, forcing_day["doy"])

        water_fac = water.water_uptake_factor(s.sw, field.soil)
        pawf = water.paw_fraction(s.sw, field.soil)
        n_fac = bg.nitrogen_stress(s.n_min)

        gdd, lai, biomass, cflux = physiology.canopy_step(
            s.gdd, s.vernalization, s.heat_acc, s.biomass, traits,
            forcing_day["tmin"], forcing_day["tmax"], forcing_day["srad"],
            daylen, forcing_day.get("co2"),
            water_fac, n_fac, s.growing,
        )
        vernalization, heat_acc = cflux["vern_new"], cflux["heat_new"]

        evaporation = physiology.potential_evaporation(et0, lai, traits, s.growing)
        transpiration = physiology.potential_transpiration(et0, lai, traits, s.growing)
        infiltration = forcing_day["precip"] + action["irrigation"]
        rw = root_weights(traits[:, INDEX["root_frac2"]], s.n_layers)
        sw, wflux = water.water_balance_step(
            s.sw, field.soil, infiltration, evaporation, transpiration, rw)
        et = wflux["es"] + wflux["t"].sum(-1)
        tws = s.tws * (1.0 - K_TWS) + wflux["deep"]

        bio = bg.bio_factor(forcing_day["tmin"], forcing_day["tmax"], pawf)
        n_demand = traits[:, INDEX["n_demand"]] * cflux["delta_b"]
        n_min, n_org, nflux = bg.nitrogen_step(
            s.n_min, s.n_org, bio, wflux["perc"] + wflux["deep"], sw.sum(-1),
            n_demand, traits[:, INDEX["legume"]], traits[:, INDEX["fix_frac"]],
            action["n_fert"], bg.wfps(sw.sum(-1), field.soil.sat.sum(-1)))
        n2o = s.n2o + nflux["n2o"]
        litter, soc_f, soc_s, c2flux = bg.carbon_step(
            s.litter, s.soc_f, s.soc_s, bio, torch.zeros_like(s.litter),
            action.get("till"), action.get("res_remove"))

        # Lifecycle events ------------------------------------------------
        yield_event = torch.zeros_like(s.biomass)
        residue_in = torch.zeros_like(s.biomass)

        mature = cflux["mature"]
        # Unconditional compute + mask select: no .any() host syncs, so the
        # step stays stream-capturable (CUDA graphs) — identical values.
        y, r = physiology.harvest(biomass, traits, heat_acc, cflux["frost"])
        yield_event = torch.where(mature, y, yield_event)
        residue_in = torch.where(mature, r, residue_in)
        biomass = torch.where(mature, torch.zeros_like(biomass), biomass)
        lai = torch.where(mature, torch.zeros_like(lai), lai)

        growing = s.growing & ~mature
        harvested = s.harvested | mature

        sow = action["sow"]
        sowing = sow >= 0
        z = s.z
        # Terminating existing standing biomass (cover crop / perennial /
        # immature crop) counts as harvest; retained fraction → litter.
        y_t, r_t = physiology.harvest(biomass, traits, heat_acc)
        yield_event = torch.where(sowing, yield_event + y_t, yield_event)
        residue_in = torch.where(sowing, residue_in + r_t, residue_in)
        zero = torch.zeros_like(biomass)
        biomass = torch.where(sowing, zero, biomass)
        lai = torch.where(sowing, zero, lai)
        gdd = torch.where(sowing, zero, gdd)
        vernalization = torch.where(sowing, zero, vernalization)
        heat_acc = torch.where(sowing, zero, heat_acc)
        if z is not None:
            z = torch.where(sowing.unsqueeze(-1), torch.zeros_like(z), z)
        crop = torch.where(sowing, sow, s.crop)
        growing = growing | sowing
        harvested = torch.where(sowing, torch.zeros_like(harvested), harvested)

        # season_yield counts grain of the current crop cycle; the termination
        # yield emitted on a sowing day still lands in cum_yield and fluxes.
        season_yield = torch.where(sowing, torch.zeros_like(s.season_yield),
                                   s.season_yield + yield_event)
        cum_yield = s.cum_yield + yield_event
        litter = litter + residue_in

        zh = s.zh
        new_state = FieldState(
            sw=sw, n_min=n_min, n_org=n_org, litter=litter,
            soc_f=soc_f, soc_s=soc_s, et=et, n2o=n2o, tws=tws,
            gdd=gdd, lai=lai, biomass=biomass,
            season_yield=season_yield, cum_yield=cum_yield,
            crop=crop, growing=growing, harvested=harvested,
            vernalization=vernalization, heat_acc=heat_acc, z=z, zh=zh,
            depth=s.depth,
            tair=0.5 * (forcing_day["tmin"] + forcing_day["tmax"]),
        )

        if self.extra:
            new_state = step_extra(self.extra, new_state, field.soil,
                                   forcing_day, action, wflux)

        if self.residual is not None or (self.latent is not None and z is not None):
            feats = self._features(new_state, forcing_day, action)
            if self.residual is not None:
                new_state = self._apply_residual(new_state, field.soil, feats)
            if self.latent is not None and z is not None:
                hierarchy = isinstance(self.latent, Hierarchy)
                z2, zh2 = self._nn(self.latent, feats, new_state.z, zh,
                                   self._dctx(field), self._gen,
                                   *((t,) if hierarchy else ()))
                new_state = new_state.replace(z=z2, zh=zh2)

        fluxes = {**wflux, **nflux, **c2flux,
                  "et0": et0, "et": et, "delta_b": cflux["delta_b"],
                  "limiter": cflux["limiter"],
                  "water_fac": water_fac, "yield_event": yield_event,
                  "residue_in": residue_in,
                  "heat_dd": cflux["heat_dd"], "frost": cflux["frost"].to(DTYPE)}
        return field.__class__(state=new_state, soil=field.soil, latitude=field.latitude,
                               elevation=field.elevation, context=field.context), fluxes

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
                            (field.state.sw, forcing.tmin, *self.parameters())):
            ckpt = 0  # no backward possible → no activation-memory win
        if not ckpt:
            for t in range(forcing.days):
                field, fluxes = self.step(field, actions.day(t),
                                          forcing.day(t), t)
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
        accumulate = observer.accumulate if observer is not None else None
        static = {"soil": field.soil, "lat": field.latitude, "elev": field.elevation,
                  "ctx": field.context}
        cls = field.__class__
        avars = ACTION_VARS
        nf, na = len(FORCING_VARS), len(avars)

        for t0 in range(0, forcing.days, every):
            t1 = min(t0 + every, forcing.days)
            # forcing/actions are explicit checkpoint inputs so gradients
            # cross segment boundaries through the recompute.
            ext = [getattr(forcing, k)[:, t0:t1] for k in FORCING_VARS] \
                + [getattr(actions, k)[:, t0:t1] for k in avars]

            # t0/t1 bound as keyword defaults: recompute happens after the
            # loop ends, when the loop vars would all read their last values.
            def segment(*arguments, _t0=t0, _t1=t1):
                off = len(names)
                a = arguments[off] if accumulate is not None else None
                e = arguments[off + (accumulate is not None):]
                f = cls(state=FieldState(**dict(zip(names, arguments[:off]))),
                        **static)
                for t in range(_t1 - _t0):
                    fd = {k: e[j][:, t] for j, k in enumerate(FORCING_VARS)}
                    action = {k: e[nf + j][:, t] for j, k in enumerate(avars)}
                    f, fl = self.step(f, action, fd, _t0 + t)
                    if a is not None:
                        a = fold(a, _t0 + t, day_update(f.state, fl))
                tail = () if a is None else (a,)
                return tuple(getattr(f.state, n) for n in names) + tail

            output = checkpoint(
                segment, *[getattr(field.state, n) for n in names],
                *(() if accumulate is None else (accumulate,)), *ext,
                use_reentrant=True)
            field = cls(state=FieldState(**dict(zip(names, output[:len(names)]))),
                        **static)
            if accumulate is not None:
                accumulate = output[-1]
        if observer is not None:
            observer.accumulate = accumulate
        return field
