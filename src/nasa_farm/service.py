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
import json
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
from nasa_farm.heads import OBJECTIVE_DIRECTION
from nasa_farm.plan import (enumerate_rotations, evaluate, expand_actions,
                            objective_matrix, recommend)
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
    """Plan outcome for one point; `status='error'` keeps a batch alive.

    The full evaluated objective matrix travels with the result — ranking
    is a view concern, so any client re-scalarizes under its own weights
    without re-rolling the model. `pareto`, `ranking` and `scores` are
    indices/values into `rotations` × `objectives` rows."""

    point: Point
    status: Literal["ok", "error"]
    detail: str | None = None
    plans_evaluated: int = 0
    scenarios: int = 0
    objectives: list[list[float]] = []
    pareto: list[int] = []
    ranking: list[int] = []
    scores: list[float] = []
    traces: list[dict[str, list[float]]] = []
    history: dict[str, Any] = {}


class PlanResponse(BaseModel):
    results: list[SiteResult]
    columns: list[str] = []          # objective names — matrix column order
    directions: list[int] = []       # +1 maximize / -1 minimize, aligned
    rotations: list[list[str]] = []  # (R, H) crop names, aligned with rows


class _Traces:
    """observer(t, state, fluxes) → weekly ensemble signals for every plan.

    Rollout rows are plan-major (R·S, scenario fastest), so view(R, S)
    pools each plan's weather ensemble — one batched rollout produces the
    traces that used to cost one rollout per displayed plan."""

    def __init__(self, plans: int, days: int, device) -> None:
        self.plans = plans
        self.accumulate = torch.zeros(plans, len(TRACE_CH), days,
                                      dtype=DTYPE, device=device)

    def __call__(self, t: int, state, fluxes: dict) -> None:
        r, s = self.plans, state.sw.shape[0] // self.plans
        self.accumulate[:, 0, t] = state.growing.view(r, s).to(DTYPE).mean(1)
        self.accumulate[:, 1, t] = fluxes["water_fac"].view(r, s).mean(1)
        self.accumulate[:, 2, t] = ((fluxes["water_fac"].view(r, s) < 0.5)
                                    & state.growing.view(r, s)).to(DTYPE).mean(1)
        self.accumulate[:, 3, t] = (fluxes["heat_dd"].view(r, s) > 0).to(DTYPE).mean(1)
        self.accumulate[:, 4, t] = fluxes["frost"].view(r, s).to(DTYPE).mean(1)

    def result(self) -> list[dict[str, list[float]]]:
        """Weekly bins — mean for states, max for the frost event channel."""
        w = math.ceil(self.accumulate.shape[2] / TRACE_BIN)
        pad = w * TRACE_BIN - self.accumulate.shape[2]
        b = torch.nn.functional.pad(
            self.accumulate, (0, pad)).unflatten(2, (w, TRACE_BIN))
        output = b.mean(-1)
        output[:, -1] = b[:, -1].amax(-1)
        return [{k: [round(float(v), 3) for v in output[p, i]]
                 for i, k in enumerate(TRACE_CH)}
                for p in range(self.plans)]


def _traces(model: WorldModel, field0: Field, plans: torch.Tensor,
            ensemble: Forcing, irrigation: float, fertilizer: float
            ) -> list[dict[str, list[float]]]:
    """Every plan's weekly ensemble signals in one rollout."""
    r, s = plans.shape[0], ensemble.batch
    field = field0.repeat(r * s) if field0.batch == 1 else field0
    a = expand_actions(plans, model.crop_table,
                       irr_mm_day=irrigation, n_fert_kg=fertilizer)
    actions = Actions(sow=a.sow.repeat_interleave(s, dim=0),
                      irrigation=a.irrigation.repeat_interleave(s, dim=0),
                      n_fert=a.n_fert.repeat_interleave(s, dim=0))
    traits = _Traces(r, ensemble.days, field.state.sw.device)
    with torch.inference_mode():
        model.rollout(field, ensemble.repeat(r), actions, observer=traits)
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


def _lake() -> Path | None:
    return (root := os.environ.get("NASA_FARM_LAKE")) and Path(root)


def _soil(p: Point) -> SoilProfile | None:
    """Stored lake soil (same product the residual trained on) — the
    offline-capable path; live SoilGrids is the fallback."""
    root, fid = _lake(), p.id
    if not (root and fid):
        return None
    from nasa_farm.lake import Lake
    from nasa_farm.optimize.data import soil
    try:
        with Lake(root, readonly=True) as L:
            return soil(L.metadata(fid).get("soil"))
    except (KeyError, FileNotFoundError, ValueError):
        return None


def _forcings(request: PlanRequest, p: Point) -> tuple[list, torch.Tensor | None]:
    """Daily forcing years + observed soil wetness (lake first, else POWER)."""
    if request.offline:
        return [synthetic_year(p.latitude, seed=y) for y in range(request.years)], None
    last = dt.date.today().year - 1
    years = list(range(last - request.years + 1, last + 1))
    if (root := _lake()) and p.id:
        from nasa_farm.optimize.data import forcings, wetness
        try:
            series = forcings(root, p.id, years)
            if series and (wet := wetness(root, p.id)) is not None:
                return series, torch.from_numpy(
                    wet[:len(years) * 365].copy())
        except (KeyError, FileNotFoundError, ValueError):
            pass
    if _in_nldas(p):
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
        source = (request.source
                  if not request.offline or request.source in ("power", "synth")
                  else "power")
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
            soil = _soil(point) or SoilGrids().profile(
                point.latitude, point.longitude)

        field = Field(state=initial_state(1, soil), soil=soil,
                      latitude=torch.tensor([point.latitude], dtype=DTYPE))
        model = WorldModel(crops, residual=request.residual)
        if request.residual and (root := _lake()):
            pt = root / "residual.pt"
            if pt.is_file():
                model.residual.load_state_dict(
                    torch.load(pt, weights_only=True))
        event = evaluate(model, field, plans, ens,
                      irr_mm_day=request.irrig, n_fert_kg=request.fertilizer,
                      workers=workers, device=device, compile=compile)
        record = recommend(event, weights=request.weights,
                        top_k=min(request.top_k, plans.shape[0]))
        mat, _ = objective_matrix(record["objectives"], record["metric_names"])
        return SiteResult(
            point=point, status="ok",
            plans_evaluated=int(plans.shape[0]), scenarios=int(ens.batch),
            objectives=[[round(float(v), 6) for v in row] for row in mat],
            pareto=record["pareto_idx"].tolist(),
            ranking=record["ranking"].tolist(),
            scores=[round(float(v), 4) for v in record["scores"]],
            traces=_traces(model, field, plans, ens,
                           request.irrig, request.fertilizer),
            history=_history(series, dt.date.today().year - request.years, sm))
    except Exception as exception:
        return SiteResult(point=point, status="error", detail=str(exception))


def _library() -> CropLibrary:
    """Crop trait table — with `lake/traits.json` calibration overrides
    applied when a lake is configured (`nasa-farm traits` writes it)."""
    crops = CropLibrary.default()
    if (root := _lake()) and (pt := root / "traits.json").is_file():
        crops = crops.calibrated(json.loads(pt.read_text()).get("traits", {}))
    return crops


def plan(request: PlanRequest, *, workers: int = 0, device: str = "auto",
         compile: str = "auto") -> PlanResponse:
    """Evaluate every plan × scenario at every point — request-level errors
    (unknown crop) raise, per-point failures degrade to status='error'."""
    crops = _library()
    unknown = [c for c in request.crops if c not in crops.names]
    if unknown:
        raise ValueError(
            f"unknown crops: {', '.join(unknown)} "
            f"(have: {', '.join(crops.names)})")
    plans = enumerate_rotations([crops.index(c) for c in request.crops],
                                request.horizon)
    depths = (tuple(float(d) for d in request.depths)
              if request.depths else DEPTHS)
    return PlanResponse(
        results=[_site(p, request, crops, plans, depths, workers, device,
                       compile) for p in request.points],
        columns=list(OBJECTIVE_DIRECTION),
        directions=list(OBJECTIVE_DIRECTION.values()),
        rotations=[[crops.names[c] for c in row.tolist()] for row in plans])
