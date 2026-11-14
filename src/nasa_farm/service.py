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
import os
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field as PField

from nasa_farm import CropLibrary, Field, WorldModel, initial_state
from nasa_farm.acquire.power import Power, series_to_forcing, series_to_wet
from nasa_farm.acquire.soilgrid import SoilGrids
from nasa_farm.constants import DTYPE
from nasa_farm.forcing import ensemble, synthetic_year
from nasa_farm.plan import enumerate_rotations, evaluate, expand_actions, recommend
from nasa_farm.state.soil import DEPTHS, SoilProfile
from nasa_farm.types import Actions, Forcing

DEFAULT_CROPS: tuple[str, ...] = ("maize", "soybean", "wheat", "clover", "fallow")
TRACE_CH: tuple[str, ...] = ("canopy", "water", "stress", "heat", "frost")
TRACE_BIN = 7
SCENARIO_KEYS: frozenset[str] = frozenset(
    {"delta_t", "precip_scale", "co2", "irrig", "fert"})
"""Request fields a `--scenario name:k=v` override may touch."""


class Point(BaseModel):
    """One site to plan for; `id` echoes back so batches stay self-aligned."""

    id: str | None = None
    latitude: float = PField(ge=-90, le=90)
    longitude: float = PField(ge=-180, le=180)


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
    fertilizer: float = PField(default=0.0, ge=0, le=500)
    co2: float = PField(default=420.0, ge=200, le=1500)
    source: str = "power"
    members: int = PField(default=16, ge=1, le=51)
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
    history: dict[str, Any] = {}


class PlanResponse(BaseModel):
    results: list[SiteResult]


class _Trace:
    """observer(t, state, fluxes) → ensemble-mean daily signals, binned weekly."""

    def __init__(self, days: int, device) -> None:
        self.accumulate = torch.zeros(len(TRACE_CH), days, dtype=DTYPE, device=device)

    def __call__(self, t: int, state, fluxes: dict) -> None:
        self.accumulate[0, t] = state.growing.to(DTYPE).mean()
        self.accumulate[1, t] = fluxes["water_fac"].mean()
        self.accumulate[2, t] = ((fluxes["water_fac"] < 0.5)
                          & state.growing).to(DTYPE).mean()
        self.accumulate[3, t] = (fluxes["heat_dd"] > 0).to(DTYPE).mean()
        self.accumulate[4, t] = fluxes["frost"].to(DTYPE).mean()

    def result(self) -> dict[str, list[float]]:
        """Weekly bins — mean for states, max for the frost event channel."""
        w = math.ceil(self.accumulate.shape[1] / TRACE_BIN)
        pad = w * TRACE_BIN - self.accumulate.shape[1]
        b = torch.nn.functional.pad(self.accumulate, (0, pad)).unflatten(1, (w, TRACE_BIN))
        output = b.mean(-1)
        output[-1] = b[-1].amax(-1)
        return {k: [round(float(v), 3) for v in output[i]]
                for i, k in enumerate(TRACE_CH)}


def _trace(model: WorldModel, field0: Field, plan_row: torch.Tensor,
           ensemble: Forcing, irrigation: float, fertilizer: float) -> dict[str, list[float]]:
    """One extra rollout of the recommended plan → weekly ensemble signals."""
    s = ensemble.batch
    field = field0.repeat(s) if field0.batch == 1 else field0
    a = expand_actions(plan_row, model.crop_table,
                       irr_mm_day=irrigation, n_fert_kg=fertilizer)
    actions = Actions(a.sow.repeat_interleave(s, dim=0),
                      a.irrigation.repeat_interleave(s, dim=0),
                      a.n_fert.repeat_interleave(s, dim=0))
    traits = _Trace(ensemble.days, field.state.sw.device)
    with torch.inference_mode():
        model.rollout(field, ensemble, actions, observer=traits)
    return traits.result()


def _runs(mask: torch.Tensor, min_len: int) -> list[list[int]]:
    """Boolean 1-D mask → [[start, end), ...] runs of length ≥ min_len."""
    m = torch.nn.functional.pad(mask.to(torch.int8), (1, 1))
    edges = (m[1:] - m[:-1]).nonzero().flatten().reshape(-1, 2)
    return [[int(a), int(b)] for a, b in edges if int(b - a) >= min_len]


def _history(series: list[Forcing], y0: int,
             sm: torch.Tensor | None = None) -> dict[str, Any]:
    """Observed anomaly windows from the daily forcing years → day ranges.

    Heat is site-relative (P90 of the series itself); dry spells follow the
    WMO convention (runs of < 1 mm days); frost is every tmin < 0 day;
    `sm` is the assimilated root-zone soil wetness (GWETROOT) curve."""
    tmax = torch.cat([f.tmax[0] for f in series])
    tmin = torch.cat([f.tmin[0] for f in series])
    pr = torch.cat([f.precip[0] for f in series])
    output = {
        "y0": y0,
        "years": len(series),
        "dry": _runs(pr < 1.0, 7),
        "heat": _runs(tmax > tmax.quantile(0.9), 2),
        "frost": _runs(tmin < 0.0, 1),
    }
    if sm is not None:
        output["sm"] = [round(float(v), 3) for v in sm]
    return output


def _in_nldas(p: Point) -> bool:
    from nasa_farm.acquire.reanalysis import covers
    return covers(p.latitude, p.longitude)


def _forcings(request: PlanRequest, p: Point) -> tuple[list, torch.Tensor | None]:
    """Daily forcing years + observed soil wetness (lake first, else POWER)."""
    if request.offline:
        return [synthetic_year(p.latitude, seed=y) for y in range(request.years)], None
    last = dt.date.today().year - 1
    years = list(range(last - request.years + 1, last + 1))
    if (root := os.environ.get("NASA_FARM_LAKE")) and p.id:
        from nasa_farm.optimize.data import forcings, wetness
        try:
            series = forcings(Path(root), p.id, years)
            if series and (wet := wetness(Path(root), p.id)) is not None:
                return series, torch.from_numpy(wet[:len(years) * 365])
        except (KeyError, FileNotFoundError, ValueError):
            pass
    if not request.offline and _in_nldas(p):
        from nasa_farm.acquire.reanalysis import Reanalysis
        try:
            return Reanalysis().forcing_years(p.latitude, p.longitude, years), None
        except (OSError, KeyError, ValueError):
            pass
    client = Power()
    raw = client.data_years(p.latitude, p.longitude, years)
    series = [series_to_forcing(d, y) for d, y in zip(raw, years)]
    sm = torch.cat([series_to_wet(d, y) for d, y in zip(raw, years)])
    return series, sm


def _site(point: Point, request: PlanRequest, crops: CropLibrary,
          plans: torch.Tensor, depths: tuple[float, ...],
          workers: int, device: str, compile: str) -> SiteResult:
    try:
        series, sm = _forcings(request, point)
        source = request.source if not request.offline else (
            "power" if request.source == "power" else "synth")
        ens = ensemble(source, point.latitude, point.longitude, (),
                       request.horizon, request.members, 0, history=series)
        if request.delta_t or request.precip_scale != 1.0:
            ens = ens.perturb(request.delta_t, request.precip_scale)
        ens = dataclasses.replace(
            ens, co2=torch.full_like(ens.tmin, request.co2))

        if request.soil:
            soil = SoilProfile.preset(request.soil, depths=depths)
        elif request.offline:
            soil = SoilProfile.preset("loam", depths=depths)
        else:
            soil = SoilGrids().profile(point.latitude, point.longitude)

        field = Field(state=initial_state(1, soil), soil=soil,
                      latitude=torch.tensor([point.latitude], dtype=DTYPE))
        model = WorldModel(crops, residual=request.residual)
        if request.residual and (root := os.environ.get("NASA_FARM_LAKE")):
            pt = Path(root) / "residual.pt"
            if pt.is_file():
                model.residual.load_state_dict(
                    torch.load(pt, weights_only=True))
        event = evaluate(model, field, plans, ens,
                      irr_mm_day=request.irrig, n_fert_kg=request.fertilizer,
                      workers=workers, device=device, compile=compile)
        record = recommend(event, weights=request.weights,
                        top_k=min(request.top_k, plans.shape[0]))
        top = [
            {"rotation": [crops.names[c] for c in plans[i].tolist()],
             "score": round(float(record["scores"][i]), 4),
             **{k: round(float(v[i]), 2)
                for k, v in record["objectives"].items()}}
            for i in record["ranking"].tolist()
        ]
        if top:
            top[0]["trace"] = _trace(model, field, plans[record["ranking"][:1]],
                                     ens, request.irrig, request.fertilizer)
        return SiteResult(
            point=point, status="ok",
            plans_evaluated=int(plans.shape[0]), scenarios=int(ens.batch),
            history=_history(series, dt.date.today().year - request.years, sm),
            pareto=[
                {"rotation": [crops.names[c] for c in plans[i].tolist()],
                 "rank": int((record["ranking"] == i).nonzero().flatten()[0])
                 if (record["ranking"] == i).any() else None,
                 **{k: round(float(v[i]), 2)
                    for k, v in record["objectives"].items()}}
                for i in record["pareto_idx"].tolist()
            ],
            top=top)
    except Exception as exception:
        return SiteResult(point=point, status="error", detail=str(exception))


def plan(request: PlanRequest, *, workers: int = 0, device: str = "auto",
         compile: str = "auto") -> PlanResponse:
    """Evaluate every plan × scenario at every point — request-level errors
    (unknown crop) raise, per-point failures degrade to status='error'."""
    crops = CropLibrary.default()
    unknown = [c for c in request.crops if c not in crops.names]
    if unknown:
        raise ValueError(
            f"unknown crops: {', '.join(unknown)} "
            f"(have: {', '.join(crops.names)})")
    plans = enumerate_rotations([crops.index(c) for c in request.crops],
                                request.horizon)
    depths = (tuple(float(d) for d in request.depths)
              if request.depths else DEPTHS)
    return PlanResponse(results=[
        _site(p, request, crops, plans, depths, workers, device, compile)
        for p in request.points])
