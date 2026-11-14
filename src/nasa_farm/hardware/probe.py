"""Measured probing and its persistent cost cache.

Device selection is a measurement problem, not a heuristic: each lane's
real rollout path is timed once per workload signature, and the result —
rate plus the graph-vs-eager verdict — is persisted so the ~1 s probe is
paid once per (hardware, workload-shape) combination *ever*, not once
per process. A learned interpolation adds nothing over measurement on a
two-way choice; versioning the measurement is what makes it durable.

Cache location: ``$NASA_FARM_CACHE/probe.json`` (default
``~/.cache/nasa-farm``). Entries key on the workload signature, the
device identity, and the torch build, so a GPU/driver change or a model
shape change silently re-measures.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

import torch

_PROBE_DAYS = 200           # probe rollout length
_PROBE_ROWS = 8192          # probe plan-row cap
_MIN_FIELD_DAYS = 500_000   # below this, probing costs more than the run

_CACHE_FILE = "probe.json"


def cache_dir() -> Path:
    root = Path(os.environ.get("NASA_FARM_CACHE",
                               "~/.cache/nasa-farm")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def signature(model, plans, ensemble, workers: int) -> tuple:
    """Workload+build key — what a cached rate is valid for."""
    return (workers, plans.shape[0] // 512, plans.shape[1],
            ensemble.batch, ensemble.days, model.residual is not None,
            torch.__version__)


def worth(r: int, s: int, days: int) -> bool:
    """Whether probing pays for itself on this sweep."""
    return r * s * days >= _MIN_FIELD_DAYS


def _load() -> dict:
    try:
        return json.loads((cache_dir() / _CACHE_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save(d: dict) -> None:
    p = cache_dir() / _CACHE_FILE
    temporary = p.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(d))
        temporary.replace(p)
    except OSError:
        pass


def lookup(signal: tuple, lane_name: str) -> tuple[float, bool] | None:
    ent = _load().get(json.dumps(signal), {}).get(lane_name)
    return None if ent is None else (ent["rate"], ent["graph"])


def store(signal: tuple, lane_name: str, rate: float, graph: bool) -> None:
    d = _load()
    d.setdefault(json.dumps(signal), {})[lane_name] = {
        "rate": rate, "graph": graph}
    _save(d)


def time_fn(function: Callable[[], object]) -> float:
    """Wall seconds for one fn call, synchronizing accelerators."""
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()
    t0 = time.perf_counter()
    function()
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()
    return time.perf_counter() - t0


def measure(lane, probe_plans: torch.Tensor) -> tuple[float, bool]:
    """Measured (rows/s, use_graph) for one lane on a probe slice —
    warmup first so contexts/compiles are out of the timed path."""
    function = lane.function
    lane.use_graph = False
    try:
        function(lane, 0, probe_plans)                          # warmup
        t_eager = time_fn(lambda: function(lane, 0, probe_plans))
        rate_e = probe_plans.shape[0] / max(t_eager, 1e-9)
    except Exception:
        return 0.0, False
    if not lane.accelerator or lane.graphs is None:
        return rate_e, False
    lane.use_graph = True
    try:
        function(lane, 0, probe_plans)                          # capture+warmup
        t_graph = time_fn(lambda: function(lane, 0, probe_plans))
        rate_g = probe_plans.shape[0] / max(t_graph, 1e-9)
    except Exception:
        lane.use_graph = False
        return rate_e, False
    if rate_g > rate_e * 1.05:
        return rate_g, True
    lane.use_graph = False
    return rate_e, False


def probe_plans(plans: torch.Tensor, ensemble) -> torch.Tensor:
    return plans[: max(1, min(plans.shape[0], _PROBE_ROWS // ensemble.batch))]


def probe_ensemble(ensemble, vars: tuple[str, ...]):
    """First ``_PROBE_DAYS`` forcing columns — the probe slice."""
    days = min(_PROBE_DAYS, ensemble.days)
    return type(ensemble)(*(getattr(ensemble, k)[:, :days] for k in vars))
