"""Experiment scene — the world every runner evaluates against.

`scene(spec)` builds it once: the field, the held-out "nature drew this"
forcing realization (`forcing_t`), the members ensemble the forecaster can
only distribute over (`forcing_e`), the management schedule, the perturbed
truth model and the literature-trait prior.

Keeping scene construction out of the experiments makes twin, bench and
assim bit-comparable — identical field, weather draws and truth gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.experiments.specification import Specification
from nasa_farm.forcing import (
    augment, climatology_ensemble, synthetic_year, tile_to_horizon,
)
from nasa_farm.optimize.targets import (
    ObservedTrajectory, season_metrics, synthetic_observation,
    trajectory_loss,
)
from nasa_farm.optimize.trainer import fit, leaf
from nasa_farm.plan import expand_actions
from nasa_farm.state.crops import INDEX, CropLibrary
from nasa_farm.state.field import Field, initial_state
from nasa_farm.state.soil import SoilProfile
from nasa_farm.types import Actions, Forcing

_TRUTH_SEED_OFFSET = 7919  # decorrelates the realization from the ensemble


@dataclass(frozen=True, slots=True)
class Scene:
    crops: CropLibrary
    field: Field
    forcing_time: Forcing    # (R, T) — held-out realizations truth/obs live on
    forcing_e: Forcing    # (S, T) — the predictive weather ensemble
    actions: Actions      # (1, T) — spec.crop sown every year at its DOY
    truth: WorldModel     # prior traits × spec.perturb on spec.crop
    prior: WorldModel     # literature traits — the uncalibrated baseline

    def realization(self, i: int = 0) -> Forcing:
        """Row i of the held-out realizations — the calibration season."""
        from nasa_farm.types import FORCING_VARS
        return Forcing(*(getattr(self.forcing_time, k)[i:i + 1]
                         for k in FORCING_VARS))


def _forcings(specification: Specification) -> tuple[Forcing, Forcing]:
    """(realizations, ensemble). POWER: the last `realizations` requested
    years are held out, earlier years form the climatology ensemble.
    Synthetic: seeds decorrelated from the ensemble's — a forecast
    ensemble is never centered on the truth."""
    if specification.years:
        from nasa_farm.acquire.power import Power
        history = Power().forcing_years(specification.latitude, specification.longitude, list(specification.years))
        r = min(specification.realizations, len(history))
        return (Forcing.concat([tile_to_horizon(y, specification.horizon)
                                for y in history[-r:]]),
                climatology_ensemble(history[:-r] or history[-r:], specification.horizon))
    base = tile_to_horizon(synthetic_year(specification.latitude, seed=specification.seed),
                           specification.horizon)
    truth = Forcing.concat([
        tile_to_horizon(
            synthetic_year(specification.latitude,
                           seed=specification.seed + _TRUTH_SEED_OFFSET + k),
            specification.horizon)
        for k in range(specification.realizations)])
    return truth, augment(base, specification.members - 1, seed=specification.seed)


def scene(specification: Specification) -> Scene:
    crops = CropLibrary.default()
    soil = SoilProfile.preset(specification.soil, depths=specification.depths)
    field = Field(initial_state(1, soil), soil,
                  torch.tensor([specification.latitude], dtype=DTYPE))
    forcing_time, forcing_e = _forcings(specification)
    i_c = crops.index(specification.crop)
    actions = expand_actions(
        torch.tensor([[i_c] * specification.horizon]), crops.table)

    table = crops.table.clone()
    for p, s in specification.perturb.items():
        table[i_c, INDEX[p]] *= s
    truth = WorldModel(CropLibrary(table=table, names=crops.names),
                       residual=False, extra=specification.extra)
    return Scene(crops=crops, field=field, forcing_time=forcing_time,
                 forcing_e=forcing_e, actions=actions, truth=truth,
                 prior=WorldModel(crops, residual=False))


def observe(specification: Specification, sc: Scene) -> ObservedTrajectory:
    """Sparse noisy seasonal obs of the truth rollout on the calibration
    realization — the same `ObservedTrajectory` contract real
    yield/remote-sensing series fill."""
    return synthetic_observation(sc.truth, sc.field, sc.realization(0),
                                 sc.actions, specification.observed_channels,
                                 noise=specification.noise, seed=specification.seed)


_BOUND = 2.0
"""Fitted traits stay within [base/BOUND, base·BOUND] — a factor-2
plausibility band around literature values keeps calibration physical
(no harvest index > 1, no negative GDD)."""


def calibrate(specification: Specification, sc: Scene,
              observed: ObservedTrajectory) -> tuple[WorldModel, dict]:
    """Variational calibration: gradient-fit the perturbed trait cells of
    spec.crop through the rollout on the calibration realization. Returns
    the fitted model plus a per-parameter start/truth/fitted/recovery
    report.

    Parameters are optimized in an unconstrained logit space and mapped
    through a sigmoid into the factor-BOUND plausibility band, so the
    estimator can never leave physically meaningful ranges."""
    candidate = WorldModel(sc.crops, residual=False)
    i_c = sc.crops.index(specification.crop)
    names = tuple(specification.perturb)
    if not names:
        return candidate, {"params": {}, "best_loss": 0.0, "epochs_run": 0}
    cols = torch.tensor([INDEX[p] for p in names])
    base = candidate.crop_table.detach().clone()
    low, high = base[i_c, cols] / _BOUND, base[i_c, cols] * _BOUND
    # sigmoid(−ln2) = 1/3 → fitted == prior exactly at init
    w = leaf(torch.full((len(names),), -math.log(2.0), dtype=DTYPE))
    forcing_t0 = sc.realization(0)

    def loss() -> torch.Tensor:
        table = base.clone()
        table[i_c, cols] = low + (high - low) * torch.sigmoid(w)
        candidate.crop_table = table
        return trajectory_loss(
            season_metrics(candidate, sc.field, forcing_t0, sc.actions), observed)

    result = fit(loss, {"w": w}, epochs=specification.epochs, rate=specification.rate)
    with torch.no_grad():
        fitted = low + (high - low) * torch.sigmoid(w)
        table = base.clone()
        table[i_c, cols] = fitted
        candidate.crop_table = table

    truth_tbl = sc.truth.crop_table
    parameters = {}
    for j, p in enumerate(names):
        start, truth_v, fit_v = (float(base[i_c, cols[j]]),
                                 float(truth_tbl[i_c, cols[j]]),
                                 float(fitted[j].detach()))
        span = truth_v - start
        parameters[p] = {
            "start": start, "truth": truth_v, "fitted": fit_v,
            "recovery_pct": round(100 * (fit_v - start) / span, 2)
            if abs(span) > 1e-12 else 100.0}
    return candidate, {"params": parameters, "best_loss": round(result.best_loss, 8),
                  "epochs_run": result.epochs_run}
