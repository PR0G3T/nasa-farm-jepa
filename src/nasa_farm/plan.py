"""Planning over the world model — enumeration, batched evaluation, Pareto.

Rotation space is discrete and small (C^H sequences), so exhaustive
evaluation beats sampling-based search: exact Pareto front, zero
approximation error, fully vectorized.
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.heads import OBJECTIVE_DIRECTION, SeasonAccumulator, objectives
from nasa_farm.state.field import Field
from nasa_farm.state.crops import IDX
from nasa_farm.types import Actions, FORCING_VARS, Forcing


def enumerate_rotations(crop_indices: list[int], horizon_years: int) -> torch.Tensor:
    """(R, H) tensor of every crop sequence; R = C^H."""
    seqs = list(itertools.product(crop_indices, repeat=horizon_years))
    return torch.tensor(seqs, dtype=torch.long)


def expand_actions(plans: torch.Tensor, crop_table: torch.Tensor,
                   irr_mm_day: float = 0.0, irr_window: int = 150,
                   n_fert_kg: float = 0.0, fert_delay: int = 30) -> Actions:
    """(R, H) plans → daily Actions (R, H·365).

    Irrigation: `irr_mm_day` every day for `irr_window` days after sowing.
    Fertilizer: `n_fert_kg` applied `fert_delay` days after sowing.
    """
    r, h = plans.shape
    days = h * DAYS_PER_YEAR
    dev = plans.device
    sow = torch.full((r, days), -1, dtype=torch.long, device=dev)
    irr = torch.zeros(r, days, dtype=DTYPE, device=dev)
    fert = torch.zeros(r, days, dtype=DTYPE, device=dev)
    sow_doy = crop_table[plans, IDX["sow_doy"]].long()          # (R, H)
    for y in range(h):
        day0 = y * DAYS_PER_YEAR
        sow_day = day0 + (sow_doy[:, y] - 1).clamp(min=0, max=DAYS_PER_YEAR - 1)
        rows = torch.arange(r, device=dev)
        sow[rows, sow_day] = plans[:, y]
        if irr_mm_day > 0:
            for d in range(irr_window):
                idx = (sow_day + d).clamp(max=days - 1)
                after = sow_day + d
                mask = after < day0 + DAYS_PER_YEAR
                irr[rows[mask], idx[mask]] += irr_mm_day
        if n_fert_kg > 0:
            fday = (sow_day + fert_delay).clamp(max=days - 1)
            mask = sow_day + fert_delay < day0 + DAYS_PER_YEAR
            fert[rows[mask], fday[mask]] += n_fert_kg
    return Actions(sow=sow, irrigation=irr, n_fert=fert)


@dataclass(frozen=True, slots=True)
class Evaluation:
    metrics: torch.Tensor            # (R, S, H, M)
    objectives: dict[str, torch.Tensor]  # name → (R,)


_MIN_ROWS_PER_SHARD = 64   # below this, thread coordination costs more than it saves
_COMPILE_FIELD_DAYS = 30_000_000  # amortizes the ~10 s inductor graph build
_PROBE_DAYS = 200          # device probe rollout length
_PROBE_ROWS = 8192         # probe batch cap — saturates small GPUs already
_PROBE_MIN_FIELD_DAYS = 500_000  # below this, probing costs more than the run
_OOM_PATTERNS = ("out of memory", "can't allocate", "cannot allocate",
                 "cudaerror MemoryAllocation".lower())


def _physical_cores() -> int:
    """Unique (package, core_id) pairs — SMT siblings don't help
    dispatch-bound rollouts, so logical cpus overestimate."""
    try:
        ids = set()
        for cpu in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
            top = cpu / "topology"
            ids.add(((top / "physical_package_id").read_text().strip(),
                     (top / "core_id").read_text().strip()))
        if ids:
            return len(ids)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def _avail_ram() -> int:
    """Bytes of *currently available* RAM (Linux → POSIX → 2 GiB floor)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return (os.sysconf("SC_AVPHYS_PAGES") *
                os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 2 << 30


def _avail_mem(device: torch.device) -> int:
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return int(free)
    return _avail_ram()


def _row_bytes(days: int, horizon: int) -> int:
    """Per-rollout-row footprint: forcing+action columns dominate, then
    state, accumulator, and ~50 % headroom for temporaries."""
    return int(((10 * days + 14 + 9 * horizon) * 8) * 1.5)


def _resolve_device(device: str | torch.device) -> torch.device | None:
    """Explicit device, or None when 'auto' (caller probes the fastest)."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return None
    d = torch.device(device)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' but torch reports no usable GPU")
    return d


_DEVICE_CACHE: dict[tuple, torch.device] = {}


def _head(ens: Forcing, days: int) -> Forcing:
    """First `days` columns of every forcing channel — probe slice."""
    return Forcing(*(getattr(ens, k)[:, :days] for k in FORCING_VARS))


def _pick_device(model: WorldModel, field0: Field, plans: torch.Tensor,
                 ens: Forcing, workers: int) -> torch.device:
    """Measure the real execution path on each candidate device once per
    (workload-bucket, model-shape) and keep the winner — 'auto' that is
    actually measured, not assumed. Cached; ~1 s one-time cost."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    rows_days = plans.shape[0] * ens.batch * ens.days
    if rows_days < _PROBE_MIN_FIELD_DAYS:
        return torch.device("cpu")  # run is cheaper than measuring it
    key = (workers, plans.shape[0] // 512, plans.shape[1],
           ens.batch, ens.days, model.residual is not None)
    if key in _DEVICE_CACHE:
        return _DEVICE_CACHE[key]

    probe_plans = plans[: max(1, min(plans.shape[0], _PROBE_ROWS // ens.batch))]
    probe_ens = _head(ens, min(_PROBE_DAYS, ens.days))
    cpu, cuda = torch.device("cpu"), torch.device("cuda")
    times: dict[torch.device, float] = {cpu: float("inf"), cuda: float("inf")}
    for dev in (cpu, cuda):
        w = workers if dev == cpu else 1
        try:
            model.to(dev)
            _eval_block(model, field0.to(dev), probe_plans.to(dev),
                        probe_ens.to(dev), 0.0, 0.0, w)   # warmup (cuda ctx)
            if dev == cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _eval_block(model, field0.to(dev), probe_plans.to(dev),
                        probe_ens.to(dev), 0.0, 0.0, w)
            if dev == cuda:
                torch.cuda.synchronize()
            times[dev] = time.perf_counter() - t0
        except Exception:
            pass  # device probed-and-lost → stays at inf
    best = min(times, key=times.get)
    _DEVICE_CACHE[key] = best
    return best


def auto_workers(rows: int | None = None,
                 device: torch.device | None = None) -> int:
    """Physical cores on no-GIL CPU builds; 1 on GIL builds and on GPU
    (the device already parallelizes inside each op)."""
    if device is not None and device.type == "cuda":
        return 1
    gil = getattr(sys, "_is_gil_enabled", lambda: True)
    if gil():
        return 1
    cap = _physical_cores()
    if rows is not None:
        cap = min(cap, max(1, rows // _MIN_ROWS_PER_SHARD))
    return max(1, cap)


def _oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    cuda_oom = getattr(torch, "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exc, cuda_oom):
        return True
    msg = str(exc).lower()
    return any(p in msg for p in _OOM_PATTERNS)


def _rollout_metrics(model: WorldModel, field0: Field, plans: torch.Tensor,
                     forcing_ens: Forcing, irr_mm_day: float,
                     n_fert_kg: float) -> torch.Tensor:
    """One contiguous plan block → raw seasonal metrics (r_c·S, H, M)."""
    r, h = plans.shape
    s = forcing_ens.batch
    field = field0.repeat(r * s) if field0.batch == 1 else field0
    actions = expand_actions(plans, model.crop_table,
                             irr_mm_day=irr_mm_day, n_fert_kg=n_fert_kg)
    actions = Actions(actions.sow.repeat_interleave(s, dim=0),
                      actions.irrigation.repeat_interleave(s, dim=0),
                      actions.n_fert.repeat_interleave(s, dim=0))
    forcing = forcing_ens.repeat(r)
    acc = SeasonAccumulator(r * s, h, device=field.state.sw1.device)
    with torch.inference_mode():
        model.rollout(field, forcing, actions, observer=acc)
    return acc.result()


def _eval_block(model: WorldModel, field0: Field, plans: torch.Tensor,
                forcing_ens: Forcing, irr_mm_day: float, n_fert_kg: float,
                workers: int) -> torch.Tensor:
    """One plan block, optionally thread-sharded (no-GIL: real parallelism;
    `step` is pure — zero shared mutation between shards)."""
    if workers <= 1 or plans.shape[0] < 2:
        return _rollout_metrics(model, field0, plans, forcing_ens,
                                irr_mm_day, n_fert_kg)
    chunks = plans.chunk(workers)
    # Dispatch-bound ops: intra-op OMP only contends between workers.
    omp_prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(
                lambda p: _rollout_metrics(model, field0, p, forcing_ens,
                                           irr_mm_day, n_fert_kg),
                chunks))
    finally:
        torch.set_num_threads(omp_prev)
    return torch.cat(parts, dim=0)


def evaluate(model: WorldModel, field0: Field, plans: torch.Tensor,
             forcing_ens: Forcing, irr_mm_day: float = 0.0,
             n_fert_kg: float = 0.0, workers: int = 0,
             device: str | torch.device = "auto",
             compile: str = "auto", mem_budget: float = 0.6) -> Evaluation:
    """Roll out every plan under every scenario — fully hardware-adaptive.

    - device="auto": a one-time probe (~1 s, cached) times the real rollout
      on CPU vs GPU and keeps the winner — GPU loses on small batches
      (launch-bound) and wins large ones; measured, never assumed
    - workers=0: physical cores on no-GIL, 1 on GIL/GPU; floors on shard size
    - plan blocks sized to `mem_budget` of *free* RAM/VRAM — big sweeps stream
    - OOM ladder: block÷2 → workers÷2 → CUDA→CPU → raise. Non-OOM errors
      always propagate (a bug is never masked by lower settings).
    - compile="auto": torch.compile past ~30M field-days (build amortizes);
      "on" forces, "off" disables. Compiled graphs get dynamic shapes.
    """
    r, h = plans.shape
    s = forcing_ens.batch
    days = forcing_ens.days
    if days != h * DAYS_PER_YEAR:
        raise ValueError(
            f"forcing horizon {days}d ≠ plans horizon {h * DAYS_PER_YEAR}d")
    if field0.batch != 1:
        if field0.batch != r * s:
            raise ValueError(
                f"field batch {field0.batch} matches neither 1 nor "
                f"plans×scenarios {r * s}")
        if workers not in (0, 1):
            raise ValueError(
                "workers>1 requires a single template field (batch == 1); "
                "pre-batched fields only run serially")

    explicit = _resolve_device(device)
    orig_dev = model.crop_table.device
    orig_step = model.__dict__.get("step")
    do_compile = (compile == "on" or
                  (compile == "auto" and r * s * days >= _COMPILE_FIELD_DAYS))
    touched_cuda = False
    try:
        # probe benchmarks need the CPU candidate's worker count up front
        w = workers if workers != 0 else auto_workers(
            r * s, explicit or torch.device("cpu"))
        if field0.batch != 1:
            w = 1
        dev = explicit or _pick_device(model, field0, plans, forcing_ens, w)
        if workers == 0:
            w = 1 if field0.batch != 1 else auto_workers(r * s, dev)

        field0_d = field0.to(dev)
        ens_d = forcing_ens.to(dev)
        plans_d = plans.to(dev)
        model.to(dev)
        touched_cuda = dev.type == "cuda"
        if do_compile and orig_step is None:
            try:
                model.step = torch.compile(  # type: ignore[method-assign]
                    model.step, mode="max-autotune-no-cudagraphs", dynamic=True)
            except Exception:
                do_compile = False

        block0 = min(r, max(1, int(
            _avail_mem(dev) * mem_budget // _row_bytes(days, h)) // s))
        # Inductor pulls triton, which re-enables the GIL — a compiled step
        # parallelizes inside the kernels; extra threads would just contend.
        block, threads = block0, 1 if do_compile else w
        while True:
            try:
                parts = [
                    _eval_block(model, field0_d, blk, ens_d,
                                irr_mm_day, n_fert_kg, threads)
                    for blk in plans_d.split(block)
                ]
                season = torch.cat(parts, dim=0)
                break
            except Exception as exc:
                if not _oom(exc):
                    raise
                if block > 1:                      # stream smaller blocks
                    block = max(1, block // 2)
                elif threads > 1:                  # then drop concurrency
                    threads, block = 1, block0
                elif dev.type == "cuda":           # last resort: host RAM
                    dev = torch.device("cpu")
                    field0_d, ens_d, plans_d = (field0.to(dev),
                                                forcing_ens.to(dev),
                                                plans.to(dev))
                    model.to(dev)
                    block0 = min(r, max(1, int(
                        _avail_mem(dev) * mem_budget // _row_bytes(days, h)) // s))
                    block, threads = block0, w if field0.batch == 1 else 1
                    torch.cuda.empty_cache()
                else:
                    raise
    finally:
        if do_compile and orig_step is None:
            model.__dict__.pop("step", None)
        model.to(orig_dev)
        if touched_cuda:
            torch.cuda.empty_cache()

    metrics = season.reshape(r, s, h, -1)
    return Evaluation(metrics=metrics, objectives=objectives(season, r, s))


def objective_matrix(obj: dict[str, torch.Tensor],
                     keys: tuple[str, ...] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack objectives → (R, M) and direction signs (M,) (+1 maximize)."""
    keys = keys or tuple(obj.keys())
    mat = torch.stack([obj[k] for k in keys], dim=1)
    signs = torch.tensor([OBJECTIVE_DIRECTION[k] for k in keys], dtype=DTYPE)
    return mat, signs


def pareto_front(points: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Boolean (R,) mask of non-dominated points. points (R, M)."""
    z = points * signs
    ge = (z.unsqueeze(0) >= z.unsqueeze(1)).all(-1)  # ge[a,b]: b weakly ≥ a
    gt = (z.unsqueeze(0) > z.unsqueeze(1)).any(-1)
    dominated = (ge & gt).any(1)  # a dominated iff ∃ b strictly dominating it
    return ~dominated


def scalarize(points: torch.Tensor, signs: torch.Tensor,
              weights: torch.Tensor) -> torch.Tensor:
    """Range-normalized weighted score (R,). Higher is better."""
    z = points * signs
    span = (z.max(0).values - z.min(0).values).clamp(min=1e-9)
    zn = (z - z.min(0).values) / span
    return (zn * weights / weights.sum()).sum(-1)


def recommend(evaluation: Evaluation, weights: dict[str, float] | None = None,
              top_k: int = 5) -> dict:
    """Pareto set + priority-weighted ranking."""
    mat, signs = objective_matrix(evaluation.objectives)
    mask = pareto_front(mat, signs)
    w = torch.ones(mat.shape[1], dtype=DTYPE)
    if weights:
        keys = tuple(evaluation.objectives.keys())
        for k, v in weights.items():
            if k in keys:
                w[keys.index(k)] = float(v)
    score = scalarize(mat, signs, w)
    order = torch.argsort(score, descending=True)
    return {
        "pareto_idx": mask.nonzero().flatten(),
        "ranking": order[:top_k],
        "scores": score,
        "objectives": evaluation.objectives,
        "metric_names": tuple(evaluation.objectives.keys()),
    }
