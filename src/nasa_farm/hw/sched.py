"""Co-execution scheduler: every lane works, measured, never assumed.

A ``Lane`` is one compute endpoint — a CPU worker pool or one
accelerator device. Work is a row-space of plans; lanes pull blocks from
a shared cursor guided by self-scheduling (pulls shrink toward the end,
bounding tail latency), so faster lanes naturally absorb more work and a
slowing lane simply pulls less — co-execution without a fixed split.

Failure ladder, per lane: OOM → halve the pull → shrink workers (CPU)
→ lane dies → survivors absorb remaining rows → only when no lane is
left does the original error propagate. Non-OOM errors abort the whole
sweep immediately: a bug is never masked by lower settings.
"""

from __future__ import annotations

import threading
from concurrent.futures import Executor
from dataclasses import dataclass, field as _dc_field
from typing import Callable

import torch

from nasa_farm.hw import graphs as _graphs
from nasa_farm.hw import mem
from nasa_farm.hw import pipe
from nasa_farm.hw import pool as _pool
from nasa_farm.hw import thermo as _thermo

_OOM_PATTERNS = ("out of memory", "can't allocate", "cannot allocate",
                 "cudaerror memoryallocation")
_MIN_PULL = 1             # plans; end-of-sweep granularity bound


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    cuda_oom = getattr(torch, "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exc, cuda_oom):
        return True
    msg = str(exc).lower()
    return any(p in msg for p in _OOM_PATTERNS)


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
    fn: Callable | None = None
    model: object = None
    field: object = None
    ens: object = None
    pool: Executor | None = None
    comp_s: object = None
    copy_s: object = None
    stage: pipe.Stage | None = None
    graphs: _graphs.Cache | None = None
    ctx: dict = _dc_field(default_factory=dict)
    mem_pool: object = None
    thermo: _thermo.Monitor | None = None
    alive: bool = True

    @property
    def accel(self) -> bool:
        return self.device.type != "cpu"

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)


def make_lane(device: torch.device, workers: int,
              thermo: _thermo.Monitor | None = None) -> Lane:
    """Wire a lane's hardware-facing resources: streams + pinned stage +
    graph cache + OOM reserve pool on accelerators, a pinned worker pool
    on CPU."""
    comp_s, copy_s = pipe.streams(device)
    lane = Lane(
        name=device_label(device), device=device, workers=workers,
        comp_s=comp_s, copy_s=copy_s,
        stage=pipe.Stage(device, copy_s, comp_s),
        graphs=(_graphs.Cache(device) if device.type == "cuda"
                else None),
        mem_pool=mem.reserve_pool(device),
        thermo=thermo,
    )
    if device.type == "cpu" and workers > 1:
        lane.pool = _pool.make(workers)
    return lane


def device_label(device: torch.device) -> str:
    from nasa_farm.hw.caps import name
    return name(device)


def run(lanes: list[Lane], plans: torch.Tensor,
        out: torch.Tensor, scen: int) -> None:
    """Drive `lanes` over the rows of `plans`, writing each block's
    metrics into ``out[start*s : end*s]``.

    Lanes pull blocks concurrently; ``out`` slots are disjoint per row
    range so assembly is order-free and bit-identical to serial.
    """
    n_rows = plans.shape[0]
    lock = threading.Lock()
    abort = threading.Event()
    errors: list[BaseException] = []
    pos = [0]

    def take(lane: Lane) -> tuple[int, int]:
        with lock:
            if pos[0] >= n_rows:
                return 0, 0
            live = max(1, sum(1 for l in lanes if l.alive))
            k = min(lane.block_rows, n_rows - pos[0],
                    max(_MIN_PULL, (n_rows - pos[0]) // (2 * live)))
            start = pos[0]
            pos[0] += k
            return start, k

    def give_back(k: int) -> None:
        with lock:
            pos[0] -= k

    def shrink(lane: Lane) -> bool:
        """OOM recovery. True → lane keeps working with less."""
        if lane.block_rows > 1:
            lane.block_rows = max(1, lane.block_rows // 2)
            return True
        if lane.device.type == "cpu" and lane.pool is not None \
                and lane.workers > 1:
            lane.workers = max(1, lane.workers // 2)
            lane.pool.shutdown(wait=True)
            lane.pool = (_pool.make(lane.workers)
                         if lane.workers > 1 else None)
            lane.block_rows = lane.cap_rows
            return True
        lane.alive = False
        mem.empty(lane.device)
        return False

    def drive(lane: Lane) -> None:
        while not abort.is_set():
            start, k = take(lane)
            if k <= 0:
                return
            block = plans[start:start + k]
            try:
                res = lane.fn(lane, start, block)
            except Exception as exc:
                if not is_oom(exc):
                    errors.append(exc)
                    abort.set()
                    return
                give_back(k)
                if not shrink(lane) and not any(l.alive for l in lanes):
                    errors.append(exc)
                    abort.set()
                continue
            try:
                _store(lane, out, start, k, scen, res)
            except Exception as exc:
                errors.append(exc)
                abort.set()
                return

    drivers = [threading.Thread(target=drive, args=(l,), daemon=True,
                                name=f"nasa-farm-{l.name}")
               for l in lanes if l.alive]
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


def _store(lane: Lane, out: torch.Tensor, start: int, k: int,
           scen: int, res: torch.Tensor) -> None:
    dst = out[start * scen:(start + k) * scen]
    if lane.copy_s is not None:
        res.record_stream(lane.copy_s)
        with torch.cuda.stream(lane.copy_s):
            lane.copy_s.wait_stream(lane.comp_s)
            dst.copy_(res, non_blocking=True)
        lane.copy_s.synchronize()
    else:
        dst.copy_(res)
