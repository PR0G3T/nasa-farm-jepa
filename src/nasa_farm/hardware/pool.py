"""Worker pool: OS threads pinned to distinct physical cores.

Threads are the right primitive for tensor-bound shards: PyTorch's
kernels release the GIL inside every C++ op, so a ThreadPoolExecutor
yields real parallelism on both free-threaded and GIL builds — and on
free-threaded builds there is no serialization at all. Subinterpreters
cannot host ``torch._C`` (single-phase extension init), and process
pools would pay pickling on every shard; threads share the model and
the read-only forcing/plan blocks at zero marshalling cost.

Workers pin themselves to distinct physical cores (``topology``), so
a pool of N never lets two shards fight over one core.
"""

from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor

from nasa_farm.hardware.topology import pin, worker_processors


def make(workers: int) -> Executor:
    """Pinned thread pool; workers >= 2 (a serial lane needs no pool)."""
    if workers <= 1:
        raise ValueError("workers must be >= 2 for a pool")
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
