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
from nasa_farm.exp.spec import Spec
from nasa_farm.forcing import (
    augment, climatology_ensemble, synthetic_year, tile_to_horizon,
)
from nasa_farm.optimize.targets import (
    ObservedTrajectory, season_metrics, synthetic_observation,
    trajectory_loss,
)
from nasa_farm.optimize.trainer import fit, leaf
from nasa_farm.plan import expand_actions
from nasa_farm.state.crops import IDX, CropLibrary
from nasa_farm.state.field import Field, initial_state
from nasa_farm.state.soil import SoilProfile
from nasa_farm.types import Actions, Forcing

_TRUTH_SEED_OFFSET = 7919  # decorrelates the realization from the ensemble


@dataclass(frozen=True, slots=True)
class Scene:
    crops: CropLibrary
    field: Field
    forcing_t: Forcing    # (R, T) — held-out realizations truth/obs live on
    forcing_e: Forcing    # (S, T) — the predictive weather ensemble
    actions: Actions      # (1, T) — spec.crop sown every year at its DOY
    truth: WorldModel     # prior traits × spec.perturb on spec.crop
    prior: WorldModel     # literature traits — the uncalibrated baseline

    def realization(self, i: int = 0) -> Forcing:
        """Row i of the held-out realizations — the calibration season."""
        from nasa_farm.types import FORCING_VARS
        return Forcing(*(getattr(self.forcing_t, k)[i:i + 1]
                         for k in FORCING_VARS))


def _forcings(spec: Spec) -> tuple[Forcing, Forcing]:
    """(realizations, ensemble). POWER: the last `realizations` requested
    years are held out, earlier years form the climatology ensemble.
    Synthetic: seeds decorrelated from the ensemble's — a forecast
    ensemble is never centered on the truth."""
    if spec.years:
        from nasa_farm.acquire.power import PowerClient
        hist = PowerClient().forcing_years(spec.lat, spec.lon, list(spec.years))
        r = min(spec.realizations, len(hist))
        return (Forcing.concat([tile_to_horizon(y, spec.horizon)
                                for y in hist[-r:]]),
                climatology_ensemble(hist[:-r] or hist[-r:], spec.horizon))
    base = tile_to_horizon(synthetic_year(spec.lat, seed=spec.seed),
                           spec.horizon)
    truth = Forcing.concat([
        tile_to_horizon(
            synthetic_year(spec.lat,
                           seed=spec.seed + _TRUTH_SEED_OFFSET + k),
            spec.horizon)
        for k in range(spec.realizations)])
    return truth, augment(base, spec.members - 1, seed=spec.seed)


def scene(spec: Spec) -> Scene:
    crops = CropLibrary.default()
    soil = SoilProfile.preset(spec.soil, depths=spec.depths)
    field = Field(initial_state(1, soil), soil,
                  torch.tensor([spec.lat], dtype=DTYPE))
    forcing_t, forcing_e = _forcings(spec)
    i_c = crops.index(spec.crop)
    actions = expand_actions(
        torch.tensor([[i_c] * spec.horizon]), crops.table)

    tbl = crops.table.clone()
    for p, s in spec.perturb.items():
        tbl[i_c, IDX[p]] *= s
    truth = WorldModel(CropLibrary(table=tbl, names=crops.names),
                       residual=False)
    return Scene(crops=crops, field=field, forcing_t=forcing_t,
                 forcing_e=forcing_e, actions=actions, truth=truth,
                 prior=WorldModel(crops, residual=False))


def observe(spec: Spec, sc: Scene) -> ObservedTrajectory:
    """Sparse noisy seasonal obs of the truth rollout on the calibration
    realization — the same `ObservedTrajectory` contract real
    yield/remote-sensing series fill."""
    return synthetic_observation(sc.truth, sc.field, sc.realization(0),
                                 sc.actions, spec.channels,
                                 noise=spec.noise, seed=spec.seed)


_BOUND = 2.0
"""Fitted traits stay within [base/BOUND, base·BOUND] — a factor-2
plausibility band around literature values keeps calibration physical
(no harvest index > 1, no negative GDD)."""


def calibrate(spec: Spec, sc: Scene,
              obs: ObservedTrajectory) -> tuple[WorldModel, dict]:
    """Variational calibration: gradient-fit the perturbed trait cells of
    spec.crop through the rollout on the calibration realization. Returns
    the fitted model plus a per-parameter start/truth/fitted/recovery
    report.

    Parameters are optimized in an unconstrained logit space and mapped
    through a sigmoid into the factor-BOUND plausibility band, so the
    estimator can never leave physically meaningful ranges."""
    cand = WorldModel(sc.crops, residual=False)
    i_c = sc.crops.index(spec.crop)
    names = tuple(spec.perturb)
    if not names:
        return cand, {"params": {}, "best_loss": 0.0, "epochs_run": 0}
    cols = torch.tensor([IDX[p] for p in names])
    base = cand.crop_table.detach().clone()
    lo, hi = base[i_c, cols] / _BOUND, base[i_c, cols] * _BOUND
    # sigmoid(−ln2) = 1/3 → fitted == prior exactly at init
    w = leaf(torch.full((len(names),), -math.log(2.0), dtype=DTYPE))
    forcing_t0 = sc.realization(0)

    def loss() -> torch.Tensor:
        tbl = base.clone()
        tbl[i_c, cols] = lo + (hi - lo) * torch.sigmoid(w)
        cand.crop_table = tbl
        return trajectory_loss(
            season_metrics(cand, sc.field, forcing_t0, sc.actions), obs)

    res = fit(loss, {"w": w}, epochs=spec.epochs, lr=spec.lr)
    with torch.no_grad():
        fitted = lo + (hi - lo) * torch.sigmoid(w)
        tbl = base.clone()
        tbl[i_c, cols] = fitted
        cand.crop_table = tbl

    truth_tbl = sc.truth.crop_table
    params = {}
    for j, p in enumerate(names):
        start, truth_v, fit_v = (float(base[i_c, cols[j]]),
                                 float(truth_tbl[i_c, cols[j]]),
                                 float(fitted[j].detach()))
        span = truth_v - start
        params[p] = {
            "start": start, "truth": truth_v, "fitted": fit_v,
            "recovery_pct": round(100 * (fit_v - start) / span, 2)
            if abs(span) > 1e-12 else 100.0}
    return cand, {"params": params, "best_loss": round(res.best_loss, 8),
                  "epochs_run": res.epochs_run}
