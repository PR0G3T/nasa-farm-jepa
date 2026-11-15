"""Co-execution scheduler: every lane works, measured, never assumed.

A ``Lane`` is one compute endpoint — a CPU worker pool or one
accelerator device. Work is a row-space of plans; each lane gets a
contiguous share proportional to its measured rate, chunked into
uniform-time blocks — faster lanes simply get larger shares, and the
assignment is deterministic, so reruns are bit-identical.

Failure ladder, per lane: OOM → halve the pull → shrink workers (CPU)
→ lane dies → its range is re-split among survivors → only when no lane
is left does the original error propagate. Non-OOM errors abort the
whole sweep immediately: a bug is never masked by lower settings.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from dataclasses import field as _dc_field

import torch

from nasa_farm.hardware import graphs as _graphs
from nasa_farm.hardware import memory as memory
from nasa_farm.hardware import pool as _pool
from nasa_farm.hardware import streams as streams
from nasa_farm.hardware import telemetry as _thermo

_OOM_PATTERNS = ("out of memory", "can't allocate", "cannot allocate", "cudaerror memoryallocation")


def out_of_memory(exception: BaseException) -> bool:
    if isinstance(exception, MemoryError):
        return True
    cuda_oom = getattr(torch, "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exception, cuda_oom):
        return True
    message = str(exception).lower()
    return any(p in message for p in _OOM_PATTERNS)


@dataclass
class Lane:
    """One compute endpoint. ``fn`` does the real work:
    ``fn(lane, plans_block) -> (k·s, H, M) metrics`` on `device`."""

    name: str
    device: torch.device
    workers: int = 1
    block_rows: int = 1 << 30
    cap_rows: int = 1 << 30
    rate: float = 0.0
    use_graph: bool = False
    function: Callable | None = None
    model: object = None
    field: object = None
    ensemble: object = None
    pool: Executor | None = None
    comp_s: object = None
    copy_s: object = None
    stage: streams.Stage | None = None
    graphs: _graphs.Cache | None = None
    context: dict = _dc_field(default_factory=dict)
    mem_pool: object = None
    telemetry: _thermo.Monitor | None = None
    alive: bool = True

    @property
    def accelerator(self) -> bool:
        return self.device.type != "cpu"

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)


def make_lane(device: torch.device, workers: int, telemetry: _thermo.Monitor | None = None) -> Lane:
    """Wire a lane's hardware-facing resources: streams + pinned stage +
    graph cache + OOM reserve pool on accelerators, a pinned worker pool
    on CPU."""
    comp_s, copy_s = streams.streams(device)
    lane = Lane(
        name=device_label(device),
        device=device,
        workers=workers,
        comp_s=comp_s,
        copy_s=copy_s,
        stage=streams.Stage(device, copy_s, comp_s),
        graphs=(_graphs.Cache(device) if device.type == "cuda" else None),
        mem_pool=memory.reserve_pool(device),
        telemetry=telemetry,
    )
    if device.type == "cpu" and workers > 1:
        lane.pool = _pool.make(workers)
    return lane


def device_label(device: torch.device) -> str:
    from nasa_farm.hardware.capabilities import name

    return name(device)


def _shrink(lane: Lane) -> bool:
    """OOM recovery. True → the lane keeps working with less."""
    if lane.block_rows > 1:
        lane.block_rows = max(1, lane.block_rows // 2)
        return True
    if lane.device.type == "cpu" and lane.pool is not None and lane.workers > 1:
        lane.workers = max(1, lane.workers // 2)
        lane.pool.shutdown(wait=True)
        lane.pool = _pool.make(lane.workers) if lane.workers > 1 else None
        lane.block_rows = lane.cap_rows
        return True
    lane.alive = False
    memory.empty(lane.device)
    return False


def _split(lanes: list[Lane], ranges: list[tuple[int, int]]) -> list[tuple[Lane, list]]:
    """Contiguous rate-proportional apportion of pending row ranges.
    Deterministic: identical rates → identical row→lane assignment."""
    total = sum(hi - lo for lo, hi in ranges)
    rates = [lane.rate for lane in lanes]
    if sum(rates) <= 0:
        rates = [1.0] * len(lanes)
    scale = total / sum(rates)
    want = [rate * scale for rate in rates]
    counts = [int(w) for w in want]
    order = sorted(range(len(lanes)), key=lambda i: want[i] - counts[i], reverse=True)
    for i in order[: total - sum(counts)]:
        counts[i] += 1
    out = [(lane, []) for lane in lanes]
    spans = iter(ranges)
    lo, hi = next(spans)
    for (_, parts), n in zip(out, counts, strict=True):
        while n > 0:
            if lo == hi:
                lo, hi = next(spans)
            take = min(n, hi - lo)
            parts.append((lo, lo + take))
            lo += take
            n -= take
    return out


def _drive(
    lane: Lane,
    ranges: list[tuple[int, int]],
    plans: torch.Tensor,
    output: torch.Tensor,
    scen: int,
    pending: list,
    lock: threading.Lock,
    abort: threading.Event,
    errors: list,
    deaths: list,
) -> None:
    """Serial driver over a lane's contiguous ranges, chunked by
    block_rows. OOM shrinks and retries in place; lane death releases the
    lane's remaining range into `pending` for survivors."""
    for lo, hi in ranges:
        start = lo
        while start < hi and not abort.is_set():
            k = min(lane.block_rows, hi - start)
            try:
                result = lane.function(lane, start, plans[start : start + k])
            except Exception as exception:
                if not out_of_memory(exception):
                    errors.append(exception)
                    abort.set()
                    return
                if not _shrink(lane):
                    deaths.append((lane.name, exception))
                    with lock:
                        pending.append((start, hi))
                    return
                continue
            try:
                _store(lane, output, start, k, scen, result)
            except Exception as exception:
                errors.append(exception)
                abort.set()
                return
            start += k


def run(lanes: list[Lane], plans: torch.Tensor, output: torch.Tensor, scen: int) -> None:
    """Drive `lanes` over the rows of `plans`, writing each block's
    metrics into ``output[start*s : end*s]``.

    Rows split contiguously ∝ each lane's measured rate — identical rates
    give an identical row→lane assignment, so reruns are bit-identical
    (per-device float rounding aside). Lanes chunk their own range by
    block_rows — uniform block *times* bound the tail straggler. When a
    lane dies, its uncovered range is re-split among survivors; only when
    every lane is dead does the original error raise.
    """
    lock = threading.Lock()
    abort = threading.Event()
    errors: list[BaseException] = []
    deaths: list[tuple[str, BaseException]] = []
    pending = [(0, plans.shape[0])]
    while pending and not abort.is_set():
        active = [lane for lane in lanes if lane.alive]
        if not active:
            break
        pending_next = pending
        pending = []
        drivers = [
            threading.Thread(
                target=_drive,
                args=(lane, parts, plans, output, scen, pending, lock, abort, errors, deaths),
                daemon=True,
                name=f"nasa-farm-{lane.name}",
            )
            for lane, parts in _split(active, pending_next)
        ]
        for t in drivers:
            t.start()
        for t in drivers:
            t.join()
    for lane in lanes:
        if lane.copy_s is not None:
            lane.copy_s.synchronize()
        if lane.comp_s is not None:
            lane.comp_s.synchronize()
    if errors:
        raise errors[0]
    if pending:
        raise deaths[-1][1] if deaths else RuntimeError(f"{pending} plan rows left unevaluated")


def _store(
    lane: Lane, output: torch.Tensor, start: int, k: int, scen: int, result: torch.Tensor
) -> None:
    destination = output[start * scen : (start + k) * scen]
    if lane.copy_s is not None:
        result.record_stream(lane.copy_s)
        with torch.cuda.stream(lane.copy_s):
            lane.copy_s.wait_stream(lane.comp_s)
            destination.copy_(result, non_blocking=True)
        lane.copy_s.synchronize()
    else:
        destination.copy_(result)
