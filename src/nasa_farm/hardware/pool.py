"""Worker-pool backend: threads on free-threaded CPython, isolated
interpreters on GIL builds.

On a no-GIL interpreter, threads give real parallelism with zero
marshalling. On a GIL build, ``concurrent.futures.InterpreterPoolExecutor``
(3.14+) restores per-core parallelism — each worker owns its own
interpreter and GIL, so dispatch-bound tensor work scales the same way.
Arguments and results cross by pickle; per-shard floors elsewhere keep
payloads big enough to amortize it.

Workers pin themselves to distinct physical cores (``hw.topo``), so a
pool of N never lets two shards fight over one core.
"""

from __future__ import annotations

import sys
from concurrent.futures import Executor, ThreadPoolExecutor

from nasa_farm.hardware.topology import pin, worker_processors


def gil_enabled() -> bool:
    return getattr(sys, "_is_gil_enabled", lambda: True)()


def backend() -> str:
    """'interpreters' on GIL builds, 'threads' on free-threaded ones."""
    if not gil_enabled():
        return "threads"
    import concurrent.futures as counterfactual
    return "interpreters" if hasattr(counterfactual, "InterpreterPoolExecutor") \
        else "threads"


def make(workers: int) -> Executor:
    """Worker executor for the chosen backend, threads pinned to
    physical cores where the platform allows it."""
    if workers <= 1:
        raise ValueError("workers must be >= 2 for a pool")
    if backend() == "interpreters":
        import concurrent.futures as counterfactual
        return counterfactual.InterpreterPoolExecutor(max_workers=workers)
    pool = ThreadPoolExecutor(max_workers=workers)
    _pin_all(pool, workers)
    return pool


def _pin_all(pool: ThreadPoolExecutor, workers: int) -> None:
    """One pin task per worker: each pool thread claims its physical
    core once; later tasks inherit the affinity for free."""
    import threading
    barrier = threading.Barrier(workers)

    def claim(i):
        pin(worker_processors(i, workers))
        barrier.wait()

    list(pool.map(claim, range(workers)))
