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
from nasa_farm.acquire.power import PowerClient, series_to_forcing, series_to_wet
from nasa_farm.acquire.soilgrid import SoilGridClient
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
        self.acc = torch.zeros(len(TRACE_CH), days, dtype=DTYPE, device=device)

    def __call__(self, t: int, state, fluxes: dict) -> None:
        self.acc[0, t] = state.growing.to(DTYPE).mean()
        self.acc[1, t] = fluxes["water_fac"].mean()
        self.acc[2, t] = ((fluxes["water_fac"] < 0.5)
                          & state.growing).to(DTYPE).mean()
        self.acc[3, t] = (fluxes["heat_dd"] > 0).to(DTYPE).mean()
        self.acc[4, t] = fluxes["frost"].to(DTYPE).mean()

    def result(self) -> dict[str, list[float]]:
        """Weekly bins — mean for states, max for the frost event channel."""
        w = math.ceil(self.acc.shape[1] / TRACE_BIN)
        pad = w * TRACE_BIN - self.acc.shape[1]
        b = torch.nn.functional.pad(self.acc, (0, pad)).unflatten(1, (w, TRACE_BIN))
        out = b.mean(-1)
        out[-1] = b[-1].amax(-1)
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
    out = {
        "y0": y0,
        "years": len(series),
        "dry": _runs(pr < 1.0, 7),
        "heat": _runs(tmax > tmax.quantile(0.9), 2),
        "frost": _runs(tmin < 0.0, 1),
    }
    if sm is not None:
        out["sm"] = [round(float(v), 3) for v in sm]
    return out


def _in_nldas(p: Point) -> bool:
    from nasa_farm.acquire.nldas import covers
    return covers(p.lat, p.lon)


def _forcings(req: PlanRequest, p: Point) -> tuple[list, torch.Tensor | None]:
    """Daily forcing years + observed soil wetness (lake first, else POWER)."""
    if req.offline:
        return [synthetic_year(p.lat, seed=y) for y in range(req.years)], None
    last = dt.date.today().year - 1
    years = list(range(last - req.years + 1, last + 1))
    if (root := os.environ.get("NASA_FARM_LAKE")) and p.id:
        from nasa_farm.optimize.data import forcings, wetness
        try:
            series = forcings(Path(root), p.id, years)
            if series and (wet := wetness(Path(root), p.id)) is not None:
                return series, torch.from_numpy(wet[:len(years) * 365])
        except (KeyError, FileNotFoundError, ValueError):
            pass
    if not req.offline and _in_nldas(p):
        from nasa_farm.acquire.nldas import NldasClient
        try:
            return NldasClient().forcing_years(p.lat, p.lon, years), None
        except (OSError, KeyError, ValueError):
            pass
    client = PowerClient()
    raw = client.data_years(p.lat, p.lon, years)
    series = [series_to_forcing(d, y) for d, y in zip(raw, years)]
    sm = torch.cat([series_to_wet(d, y) for d, y in zip(raw, years)])
    return series, sm


def _site(point: Point, req: PlanRequest, crops: CropLibrary,
          plans: torch.Tensor, depths: tuple[float, ...],
          workers: int, device: str, compile: str) -> SiteResult:
    try:
        series, sm = _forcings(req, point)
        source = req.source if not req.offline else (
            "power" if req.source == "power" else "synth")
        ens = ensemble(source, point.lat, point.lon, (),
                       req.horizon, req.members, 0, hist=series)
        if req.delta_t or req.precip_scale != 1.0:
            ens = ens.perturb(req.delta_t, req.precip_scale)
        ens = dataclasses.replace(
            ens, co2=torch.full_like(ens.tmin, req.co2))

        if req.soil:
            soil = SoilProfile.preset(req.soil, depths=depths)
        elif req.offline:
            soil = SoilProfile.preset("loam", depths=depths)
        else:
            soil = SoilGridClient().profile(point.lat, point.lon)

        field = Field(state=initial_state(1, soil), soil=soil,
                      lat=torch.tensor([point.lat], dtype=DTYPE))
        model = WorldModel(crops, residual=req.residual)
        if req.residual and (root := os.environ.get("NASA_FARM_LAKE")):
            pt = Path(root) / "residual.pt"
            if pt.is_file():
                model.residual.load_state_dict(
                    torch.load(pt, weights_only=True))
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
            history=_history(series, dt.date.today().year - req.years, sm),
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
