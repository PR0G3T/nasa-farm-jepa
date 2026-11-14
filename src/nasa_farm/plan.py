"""Planning over the world model — enumeration, batched evaluation, Pareto.

Rotation space is discrete and small (C^H sequences), so exhaustive
evaluation beats sampling-based search: exact Pareto front, zero
approximation error, fully vectorized.

Execution is delegated to the hardware layer (`nasa_farm.hardware`): every
usable device is a co-executing lane, work is split into contiguous
cursor sized by *measured* per-lane rates, blocks stream through pinned
staging on dedicated streams, fixed-shape blocks replay through
accelerator graphs, and failures degrade lane-by-lane (OOM shrinks the
lane, it never silently changes semantics). Results assemble by plan-row
index, so co-execution stays bit-identical to serial.
"""

from __future__ import annotations

import copy
import itertools
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass

import torch

from nasa_farm.constants import DAYS_PER_YEAR, DTYPE
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.heads import (METRICS, OBJECTIVE_DIRECTION,
                             SeasonAccumulator, objectives)
from nasa_farm.hardware import capabilities as capabilities
from nasa_farm.hardware import memory as memory
from nasa_farm.hardware import probe
from nasa_farm.hardware import scheduler as schedule
from nasa_farm.hardware.telemetry import Monitor
from nasa_farm.state.crops import INDEX
from nasa_farm.state.field import Field
from nasa_farm.state.soil import SoilProfile
from nasa_farm.types import ACTION_VARS, Actions, FORCING_VARS, Forcing


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
    device = plans.device
    sow = torch.full((r, days), -1, dtype=torch.long, device=device)
    irrigation = torch.zeros(r, days, dtype=DTYPE, device=device)
    fertilizer = torch.zeros(r, days, dtype=DTYPE, device=device)
    sow_doy = crop_table[plans, INDEX["sow_doy"]].long()          # (R, H)
    for y in range(h):
        day0 = y * DAYS_PER_YEAR
        sow_day = day0 + (sow_doy[:, y] - 1).clamp(min=0, max=DAYS_PER_YEAR - 1)
        rows = torch.arange(r, device=device)
        sow[rows, sow_day] = plans[:, y]
        if irr_mm_day > 0:
            for d in range(irr_window):
                index = (sow_day + d).clamp(max=days - 1)
                after = sow_day + d
                mask = after < day0 + DAYS_PER_YEAR
                irrigation[rows[mask], index[mask]] += irr_mm_day
        if n_fert_kg > 0:
            fday = (sow_day + fert_delay).clamp(max=days - 1)
            mask = sow_day + fert_delay < day0 + DAYS_PER_YEAR
            fertilizer[rows[mask], fday[mask]] += n_fert_kg
    return Actions(sow=sow, irrigation=irrigation, n_fert=fertilizer)


@dataclass(frozen=True, slots=True)
class Evaluation:
    metrics: torch.Tensor            # (R, S, H, M)
    objectives: dict[str, torch.Tensor]  # name → (R,)


_COMPILE_FIELD_DAYS = 30_000_000  # amortizes the ~10 s inductor graph build
_LANE_MIN_SHARE = 0.05            # drop accel lanes below this share of top rate
_MAX_BLOCKS = 128                 # block-count target → tail + overhead bound
_BLOCK_MIN_S = 1.0                # seconds: shortest useful uniform block
_BLOCK_MAX_S = 30.0               # seconds: longest block (bounds stragglers)


def _row_bytes(days: int, horizon: int) -> int:
    """Per-rollout-row footprint: forcing+action columns dominate, then
    state, accumulator, and ~50 % headroom for temporaries."""
    return int(((10 * days + 14 + 9 * horizon) * 8) * 1.5)


def _resolve(device: str | torch.device) -> torch.device | None:
    """Explicit device (accelerators normalized to indexed form), or None
    when 'auto' — the scheduler then measures every usable lane."""
    if isinstance(device, torch.device):
        d = device
    elif device == "auto":
        return None
    else:
        d = torch.device(device)
    if d.type == "cpu":
        return d
    d = torch.device(d.type, 0 if d.index is None else d.index)
    if d not in capabilities.accelerators():
        raise RuntimeError(f"device {d} is not a usable accelerator")
    return d


def _core(model: WorldModel, field: Field, forcing: Forcing,
          actions: Actions, accumulate: SeasonAccumulator) -> torch.Tensor:
    """The capturable kernel of a block: reset the accumulator, roll out,
    return metrics. No host reads of device data — graph-safe."""
    accumulate.accumulate.zero_()
    with torch.inference_mode():
        model.rollout(field, forcing, actions, observer=accumulate)
    return accumulate.result()


def _expand_ri(plans: torch.Tensor, crop_table: torch.Tensor, s: int,
               irr_mm_day: float, n_fert_kg: float) -> Actions:
    """(k, H) plans → daily actions repeated per scenario (k·s, days)."""
    a = expand_actions(plans, crop_table,
                       irr_mm_day=irr_mm_day, n_fert_kg=n_fert_kg)
    return Actions(*(getattr(a, k).repeat_interleave(s, dim=0)
                     for k in ACTION_VARS))


def _rollout_metrics(model: WorldModel, field0: Field, plans: torch.Tensor,
                     forcing_ensemble: Forcing, irr_mm_day: float,
                     n_fert_kg: float) -> torch.Tensor:
    """One contiguous plan block → raw seasonal metrics (r_c·S, H, M)."""
    r, h = plans.shape
    s = forcing_ensemble.batch
    field = field0.repeat(r * s) if field0.batch == 1 else field0
    actions = _expand_ri(plans, model.crop_table, s, irr_mm_day, n_fert_kg)
    forcing = forcing_ensemble.repeat(r)
    accumulate = SeasonAccumulator(r * s, h, device=field.state.sw.device)
    return _core(model, field, forcing, actions, accumulate)


def _shard(model: WorldModel, field0: Field, plans: torch.Tensor,
           ensemble: Forcing, irr_mm_day: float,
           n_fert_kg: float) -> torch.Tensor:
    """Pool-mappable block worker — module-level so interpreter pools can
    pickle it. OMP is capped per shard: dispatch-bound ops only contend."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _rollout_metrics(model, field0, plans, ensemble,
                                irr_mm_day, n_fert_kg)
    finally:
        torch.set_num_threads(previous)


def _field_slice(field: Field, a: int, b: int) -> Field:
    """Row-slice views [a, b) of a pre-batched field."""
    soil = field.soil
    return Field(
        state=field.state._map(lambda v: v[a:b]),
        soil=SoilProfile(
            depth=soil.depth, sand=soil.sand[a:b], clay=soil.clay[a:b],
            organic=soil.organic[a:b], sat=soil.sat[a:b], fc=soil.fc[a:b],
            wp=soil.wp[a:b], air=soil.air[a:b]),
        latitude=field.latitude[a:b],
        elevation=field.elevation[a:b] if field.elevation is not None else None,
        context=field.context[a:b] if field.context is not None else None)


def _block_ctx(lane: schedule.Lane, start: int, k: int, s: int,
               h: int) -> tuple[Field, Forcing, SeasonAccumulator]:
    """Per-block expansion: repeated template field or pre-batched slice,
    scenario-repeated forcing, and the metric accumulator.

    Template-field blocks cache the expansion per row count — content is
    plan-independent, so one repeat serves every block of that size and
    its addresses stay stable for graph capture (the graph's Replayer
    also keeps the entry alive through the captured closure).
    """
    key = (0 if lane.field.batch == 1 else start, k)
    ent = lane.context.get(key)
    if ent is None:
        field = (lane.field.repeat(k * s) if lane.field.batch == 1
               else _field_slice(lane.field, start * s, (start + k) * s))
        forcing = lane.ensemble.repeat(k)
        accumulate = SeasonAccumulator(k * s, h, device=lane.device)
        if len(lane.context) >= 4:
            lane.context.pop(next(iter(lane.context)))
        lane.context[key] = ent = (field, forcing, accumulate)
    return ent


def _graphed(lane: schedule.Lane, start: int, k: int, s: int, h: int):
    """Closure captured into a graph: lazily pins this shape's ctx, then
    rebuilds Actions around the graph's static input buffers."""
    state: dict = {}

    def function(*t: torch.Tensor) -> torch.Tensor:
        if "ctx" not in state:
            state["ctx"] = _block_ctx(lane, start, k, s, h)
        field, forcing, accumulate = state["ctx"]
        return _core(lane.model, field, forcing, Actions(*t), accumulate)
    return function


def _accelerator_block(lane: schedule.Lane, start: int, block: torch.Tensor,
                 irr_mm_day: float, n_fert_kg: float) -> torch.Tensor:
    """Accelerator lane work: stage plans, expand actions eagerly (the
    data-dependent indexing is not capturable), then run the core —
    replayed when this shape has a captured graph."""
    cm = (torch.cuda.stream(lane.comp_s)
          if lane.comp_s is not None else nullcontext())
    with cm:
        k, h = block.shape
        s = lane.ensemble.batch
        plans_d = lane.stage.in_(block)
        acts = _expand_ri(plans_d, lane.model.crop_table, s,
                          irr_mm_day, n_fert_kg)
        acts_t = tuple(getattr(acts, n) for n in ACTION_VARS)
        if lane.use_graph and lane.graphs is not None \
                and lane.field.batch == 1:
            rep = lane.graphs.get((k, s, lane.ensemble.days),
                                  _graphed(lane, start, k, s, h), acts_t)
            if rep is not None:
                return rep(*acts_t)
        field, forcing, accumulate = _block_ctx(lane, start, k, s, h)
        return _core(lane.model, field, forcing, acts, accumulate)


def _processor_block(lane: schedule.Lane, start: int, block: torch.Tensor,
               irr_mm_day: float, n_fert_kg: float) -> torch.Tensor:
    """CPU lane work: pool-sharded when a pool exists, serial otherwise.
    `step` is pure — zero shared mutation between shards."""
    if lane.pool is not None and block.shape[0] >= 2:
        chunks = block.chunk(lane.workers)
        parts = lane.pool.map(
            _shard, itertools.repeat(lane.model),
            itertools.repeat(lane.field), chunks,
            itertools.repeat(lane.ensemble), itertools.repeat(irr_mm_day),
            itertools.repeat(n_fert_kg))
        return torch.cat(list(parts), dim=0)
    s = lane.ensemble.batch
    field = (lane.field if lane.field.batch == 1
           else _field_slice(lane.field, start * s,
                             (start + block.shape[0]) * s))
    return _rollout_metrics(lane.model, field, block, lane.ensemble,
                            irr_mm_day, n_fert_kg)


def _lane(model: WorldModel, field0: Field, ensemble: Forcing,
          device: torch.device, workers: int, do_compile: bool,
          mem_budget: float, r: int, s: int, h: int, days: int,
          irr_mm_day: float, n_fert_kg: float) -> schedule.Lane:
    """Build one compute endpoint: private model copy on the device,
    per-lane streams/staging/pool, memory-sized block cap, work fn."""
    ml = copy.deepcopy(model).to(device)
    if do_compile:
        _load_compile_cache()
        try:
            ml.step = torch.compile(  # type: ignore[method-assign]
                ml.step, mode="max-autotune-no-cudagraphs", dynamic=True)
        except Exception:
            pass  # compiled step is an optimization; eager is equivalent
    lane = schedule.make_lane(device, workers)
    lane.model = ml
    lane.field = field0.to(device)
    lane.ensemble = ensemble.to(device)
    lane.block_rows = lane.cap_rows = min(r, max(1, int(
        memory.available(device, mem_budget) // _row_bytes(days, h)) // s))
    lane.function = ((lambda l, start, blk: _accelerator_block(
        l, start, blk, irr_mm_day, n_fert_kg)) if lane.accelerator
        else (lambda l, start, blk: _processor_block(
            l, start, blk, irr_mm_day, n_fert_kg)))
    return lane


def _measure(lanes: list[schedule.Lane], model: WorldModel,
             plans: torch.Tensor, ensemble: Forcing, workers: int) -> None:
    """Fill lane.rate / lane.use_graph — from the persistent probe cache
    when the signature is known, else by timing the real block path."""
    signal = probe.signature(model, plans, ensemble, workers)
    pplans = probe.probe_plans(plans, ensemble)
    head = probe.probe_ensemble(ensemble, FORCING_VARS)
    for lane in lanes:
        hit = probe.lookup(signal, lane.name)
        if hit is not None:
            lane.rate, lane.use_graph = hit
            continue
        full = lane.ensemble
        lane.ensemble = head.to(lane.device)
        try:
            lane.rate, lane.use_graph = probe.measure(lane, pplans)
        finally:
            lane.ensemble = full
            lane.context.clear()          # probe-horizon ctx must not leak
            if lane.graphs is not None:
                lane.graphs = type(lane.graphs)(lane.device)
        if lane.rate > 0:
            probe.store(signal, lane.name, lane.rate, lane.use_graph)


def _prune(lanes: list[schedule.Lane]) -> list[schedule.Lane]:
    """Drop lanes that failed their probe and accel lanes too slow to pay
    their own setup. The CPU lane is never pruned on slowness — it is the
    universal fallback tail of the OOM ladder."""
    live = [l for l in lanes if l.rate > 0]
    for l in lanes:
        if l.rate <= 0:
            l.close()
    if not live:
        return []
    top = max(l.rate for l in live)
    keep = [l for l in live
            if not l.accelerator or l.rate >= _LANE_MIN_SHARE * top]
    for l in live:
        if l not in keep:
            l.close()
    return keep


def _size_blocks(lanes: list[schedule.Lane], r: int, probe_days: int,
                 days: int) -> None:
    """Uniform per-lane block rows ∝ measured rate — equal *block times*
    bound the tail straggler, uniformity maximizes graph/ctx reuse."""
    est = r / max(sum(l.rate for l in lanes), 1e-9) * (days / probe_days)
    t = min(_BLOCK_MAX_S, max(_BLOCK_MIN_S, est / _MAX_BLOCKS))
    scale = probe_days / days
    for l in lanes:
        l.block_rows = l.cap_rows = min(
            l.block_rows, r, max(1, int(l.rate * scale * t)))


def _load_compile_cache() -> None:
    """Persisted Inductor artifacts — turns the ~10 s compile warmup into
    a disk read on every process after the first."""
    try:
        p = probe.cache_dir() / "compile.mega"
        if p.exists():
            torch.compiler.load_cache_artifacts(p.read_bytes())
    except Exception:
        pass


def _save_compile_cache() -> None:
    try:
        art = torch.compiler.save_cache_artifacts()
        if art:
            (probe.cache_dir() / "compile.mega").write_bytes(art)
    except Exception:
        pass


def _telemetry(lanes: list[schedule.Lane], mons: dict[str, Monitor],
               r: int, s: int, days: int) -> dict:
    fd = r * s * days
    j = sum(m.joules() for m in mons.values())
    return {
        "field_days": fd,
        "joules": j,
        "j_per_field_day": j / fd if fd else 0.0,
        "lanes": {l.name: {"rate_plans_s": l.rate,
                           "graph": l.use_graph, "alive": l.alive,
                           "block_rows": l.block_rows}
                  for l in lanes},
        "accel": {n: m.stats() for n, m in mons.items()},
    }


def evaluate(model: WorldModel, field0: Field, plans: torch.Tensor,
             forcing_ensemble: Forcing, irr_mm_day: float = 0.0,
             n_fert_kg: float = 0.0, workers: int = 0,
             device: str | torch.device = "auto",
             compile: str = "auto", mem_budget: float = 0.6,
             telemetry: dict | None = None) -> Evaluation:
    """Roll out every plan under every scenario — fully hardware-adaptive.

    - device="auto": every usable accelerator plus the CPU co-executes;
      lanes are measured once per workload signature (cached on disk),
      pruned below 5 % of the top rate, then receive a contiguous share
      of plan rows proportional to their rate — uniform block times, a
      deterministic assignment, bit-identical reruns.
    - device explicit / small sweeps: single lane, blocks memory-sized.
    - workers=0: physical cores on the CPU lane (pinned), 1 on accel
      lanes and when a compiled step parallelizes inside its kernels.
    - Blocks: sized to `mem_budget` of free device RAM, capped so block
      *time* is uniform across lanes (bounds the tail straggler).
    - Graphs: fixed-shape blocks on CUDA replay through captured graphs
      when measurement says they win (probe decides, cache remembers).
    - OOM ladder per lane: block÷2 → CPU workers÷2 → lane dies →
      survivors absorb rows → all lanes dead → original error raised.
      Non-OOM errors abort the whole sweep immediately.
    - telemetry dict (optional): energy (J), J/field-day, per-lane rates.
    """
    r, h = plans.shape
    s = forcing_ensemble.batch
    days = forcing_ensemble.days
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

    explicit = _resolve(device)
    do_compile = (compile == "on" or
                  (compile == "auto" and r * s * days >= _COMPILE_FIELD_DAYS))
    cpu_w = 1 if field0.batch != 1 or do_compile else (
        workers if workers else capabilities.processor_workers(r * s))

    keywords = dict(do_compile=do_compile, mem_budget=mem_budget,
              r=r, s=s, h=h, days=days,
              irr_mm_day=irr_mm_day, n_fert_kg=n_fert_kg)
    if explicit is not None:
        lanes = [_lane(model, field0, forcing_ensemble, explicit,
                       cpu_w if explicit.type == "cpu" else 1, **keywords)]
        for l in lanes:
            l.use_graph = l.accelerator and l.graphs is not None
    elif not probe.worth(r, s, days):
        lanes = [_lane(model, field0, forcing_ensemble, torch.device("cpu"),
                       cpu_w, **keywords)]
    else:
        lanes = [_lane(model, field0, forcing_ensemble, d,
                       cpu_w if d.type == "cpu" else 1, **keywords)
                 for d in capabilities.lanes(DTYPE)]
        _measure(lanes, model, plans, forcing_ensemble, cpu_w)
        lanes = _prune(lanes)
        if not lanes:
            raise RuntimeError("every compute lane failed its probe")
        _size_blocks(lanes, r, min(probe._PROBE_DAYS, days), days)

    accelerator = any(l.accelerator for l in lanes)
    output = torch.empty(
        (r * s, h, len(METRICS)), dtype=DTYPE,
        pin_memory=accelerator and torch.cuda.is_available())
    mons = {l.name: Monitor(l.device) for l in lanes if l.accelerator}
    try:
        with ExitStack() as stack:
            for l in lanes:
                if l.name in mons:
                    l.telemetry = stack.enter_context(mons[l.name])
            schedule.run(lanes, plans, output, s)
        if do_compile:
            _save_compile_cache()
    finally:
        for l in lanes:
            l.close()
    if telemetry is not None:
        telemetry.update(_telemetry(lanes, mons, r, s, days))

    metrics = output.reshape(r, s, h, -1)
    return Evaluation(metrics=metrics, objectives=objectives(output, r, s))


def objective_matrix(object: dict[str, torch.Tensor],
                     keys: tuple[str, ...] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack objectives → (R, M) and direction signs (M,) (+1 maximize)."""
    keys = keys or tuple(object.keys())
    mat = torch.stack([object[k] for k in keys], dim=1)
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
