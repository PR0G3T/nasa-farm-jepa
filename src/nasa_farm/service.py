"""Plan service — the single pipeline contract shared by CLI, HTTP API and
batch export.

`plan()` owns the whole fetch → build → evaluate → recommend flow so no
caller ever re-implements it. Input/output are pydantic models: the HTTP
layer validates for free, the batch exporter reuses the same request, and
the CLI is a thin argv → request adapter.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field as PField

from nasa_farm import CropLibrary, Field, WorldModel, initial_state
from nasa_farm.acquire.power import PowerClient
from nasa_farm.acquire.soilgrids import SoilGridsClient
from nasa_farm.constants import DTYPE
from nasa_farm.forcing import climatology_ensemble, synthetic_year
from nasa_farm.plan import enumerate_rotations, evaluate, expand_actions, recommend
from nasa_farm.state.soil import DEPTHS, SoilProfile
from nasa_farm.types import Actions, Forcing

DEFAULT_CROPS: tuple[str, ...] = ("maize", "soybean", "wheat", "clover", "fallow")
TRACE_CH: tuple[str, ...] = ("canopy", "water", "stress", "frost")
TRACE_BIN = 7
SCENARIO_KEYS: frozenset[str] = frozenset(
    {"delta_t", "precip_scale", "co2", "irrig", "fert"})
"""Request fields a `--scenario name:k=v` override may touch."""


class Point(BaseModel):
    """One site to plan for; `id` echoes back so batches stay self-aligned."""

    id: str | None = None
    lat: float = PField(ge=-90, le=90)
    lon: float = PField(ge=-180, le=180)


class PlanRequest(BaseModel):
    """Everything needed to rank rotations at one or more sites."""

    points: list[Point] = PField(min_length=1, max_length=64)
    horizon: int = PField(default=3, ge=1, le=10)
    crops: list[str] = PField(default_factory=lambda: list(DEFAULT_CROPS),
                              min_length=1)
    years: int = PField(default=5, ge=1, le=20)
    delta_t: float = PField(default=0.0, ge=-10, le=10)
    precip_scale: float = PField(default=1.0, ge=0, le=3)
    irrig: float = PField(default=0.0, ge=0, le=50)
    fert: float = PField(default=0.0, ge=0, le=500)
    co2: float = PField(default=420.0, ge=200, le=1500)
    soil: str | None = None
    depths: list[float] | None = None
    residual: bool = True
    offline: bool = False
    weights: dict[str, float] | None = None
    top_k: int = PField(default=5, ge=1, le=50)


class SiteResult(BaseModel):
    """Plan outcome for one point; `status='error'` keeps a batch alive."""

    point: Point
    status: Literal["ok", "error"]
    detail: str | None = None
    plans_evaluated: int = 0
    scenarios: int = 0
    pareto: list[dict[str, Any]] = []
    top: list[dict[str, Any]] = []


class PlanResponse(BaseModel):
    results: list[SiteResult]


class _Trace:
    """observer(t, state, fluxes) → ensemble-mean daily signals, binned weekly."""

    def __init__(self, days: int, device) -> None:
        self.acc = torch.zeros(len(TRACE_CH), days, dtype=DTYPE, device=device)

    def __call__(self, t: int, state, fluxes: dict) -> None:
        self.acc[0, t] = state.growing.to(DTYPE).mean()
        self.acc[1, t] = fluxes["water_fac"].mean()
        self.acc[2, t] = ((fluxes["water_fac"] < 0.5)
                          & state.growing).to(DTYPE).mean()
        self.acc[3, t] = fluxes["frost"].to(DTYPE).mean()

    def result(self) -> dict[str, list[float]]:
        """Weekly bins — mean for states, max for the frost event channel."""
        w = math.ceil(self.acc.shape[1] / TRACE_BIN)
        pad = w * TRACE_BIN - self.acc.shape[1]
        b = torch.nn.functional.pad(self.acc, (0, pad)).unflatten(1, (w, TRACE_BIN))
        out = b.mean(-1)
        out[3] = b[3].amax(-1)
        return {k: [round(float(v), 3) for v in out[i]]
                for i, k in enumerate(TRACE_CH)}


def _trace(model: WorldModel, field0: Field, plan_row: torch.Tensor,
           ens: Forcing, irr: float, fert: float) -> dict[str, list[float]]:
    """One extra rollout of the recommended plan → weekly ensemble signals."""
    s = ens.batch
    field = field0.repeat(s) if field0.batch == 1 else field0
    a = expand_actions(plan_row, model.crop_table,
                       irr_mm_day=irr, n_fert_kg=fert)
    actions = Actions(a.sow.repeat_interleave(s, dim=0),
                      a.irrigation.repeat_interleave(s, dim=0),
                      a.n_fert.repeat_interleave(s, dim=0))
    tr = _Trace(ens.days, field.state.sw.device)
    with torch.inference_mode():
        model.rollout(field, ens, actions, observer=tr)
    return tr.result()


def _forcings(req: PlanRequest, p: Point) -> list:
    if req.offline:
        return [synthetic_year(p.lat, seed=y) for y in range(req.years)]
    last = dt.date.today().year - 1
    return PowerClient().forcing_years(
        p.lat, p.lon, list(range(last - req.years + 1, last + 1)))


def _site(point: Point, req: PlanRequest, crops: CropLibrary,
          plans: torch.Tensor, depths: tuple[float, ...],
          workers: int, device: str, compile: str) -> SiteResult:
    try:
        ens = climatology_ensemble(
            _forcings(req, point), req.horizon,
            delta_t=req.delta_t, precip_scale=req.precip_scale)
        ens = dataclasses.replace(
            ens, co2=torch.full_like(ens.tmin, req.co2))

        if req.soil:
            soil = SoilProfile.preset(req.soil, depths=depths)
        elif req.offline:
            soil = SoilProfile.preset("loam", depths=depths)
        else:
            soil = SoilGridsClient().profile(point.lat, point.lon)

        field = Field(state=initial_state(1, soil), soil=soil,
                      lat=torch.tensor([point.lat], dtype=DTYPE))
        model = WorldModel(crops, residual=req.residual)
        ev = evaluate(model, field, plans, ens,
                      irr_mm_day=req.irrig, n_fert_kg=req.fert,
                      workers=workers, device=device, compile=compile)
        rec = recommend(ev, weights=req.weights,
                        top_k=min(req.top_k, plans.shape[0]))
        top = [
            {"rotation": [crops.names[c] for c in plans[i].tolist()],
             "score": round(float(rec["scores"][i]), 4),
             **{k: round(float(v[i]), 2)
                for k, v in rec["objectives"].items()}}
            for i in rec["ranking"].tolist()
        ]
        if top:
            top[0]["trace"] = _trace(model, field, plans[rec["ranking"][:1]],
                                     ens, req.irrig, req.fert)
        return SiteResult(
            point=point, status="ok",
            plans_evaluated=int(plans.shape[0]), scenarios=int(ens.batch),
            pareto=[
                {"rotation": [crops.names[c] for c in plans[i].tolist()],
                 "rank": int((rec["ranking"] == i).nonzero().flatten()[0])
                 if (rec["ranking"] == i).any() else None,
                 **{k: round(float(v[i]), 2)
                    for k, v in rec["objectives"].items()}}
                for i in rec["pareto_idx"].tolist()
            ],
            top=top)
    except Exception as exc:
        return SiteResult(point=point, status="error", detail=str(exc))


def plan(req: PlanRequest, *, workers: int = 0, device: str = "auto",
         compile: str = "auto") -> PlanResponse:
    """Evaluate every plan × scenario at every point — request-level errors
    (unknown crop) raise, per-point failures degrade to status='error'."""
    crops = CropLibrary.default()
    unknown = [c for c in req.crops if c not in crops.names]
    if unknown:
        raise ValueError(
            f"unknown crops: {', '.join(unknown)} "
            f"(have: {', '.join(crops.names)})")
    plans = enumerate_rotations([crops.index(c) for c in req.crops],
                                req.horizon)
    depths = (tuple(float(d) for d in req.depths)
              if req.depths else DEPTHS)
    return PlanResponse(results=[
        _site(p, req, crops, plans, depths, workers, device, compile)
        for p in req.points])
