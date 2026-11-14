"""Accelerator graph capture/replay for fixed-shape work blocks.

A captured block replays its entire kernel sequence with one launch:
all dispatch overhead — ~50-100 launches per simulated day — collapses
to a single submission. Inputs land in static buffers; outputs are read
from stable graph-pool storage.

Graphs share one memory pool per lane (sequential lanes never run two
graphs at once), so an extra graph costs ~zero extra VRAM. Any capture
failure degrades to eager: a graph is an optimization, never semantics.
"""

from __future__ import annotations

import torch


class Replayer:
    """One captured graph: copy inputs into static buffers, replay."""

    def __init__(self, graph, static_in: tuple[torch.Tensor, ...],
                 out: torch.Tensor, keep=None) -> None:
        self.g = graph
        self.static_in = static_in
        self.out = out
        # `keep` holds the captured closure: the graph references ctx
        # tensors (field/forcing/acc) by raw address — keeping `fn`
        # alive keeps those addresses valid for the replayer's life.
        self.keep = keep

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        for dst, src in zip(self.static_in, inputs):
            dst.copy_(src, non_blocking=True)
        self.g.replay()
        return self.out


def pool_handle(device: torch.device):
    """Shared graph memory pool for a lane (CUDA only; None elsewhere)."""
    if device.type == "cuda":
        return torch.cuda.graph_pool_handle()
    return None


def capture(fn, inputs: tuple[torch.Tensor, ...],
            device: torch.device, pool=None) -> Replayer | None:
    """Capture `fn(*inputs)` for replay. Returns None on any failure —
    the caller falls back to eager execution of the same `fn`.

    Two warmup iterations on a side stream settle lazy initialization
    (CUDA contexts, compiled-kernel autotuning) before capture.
    """
    try:
        static_in = tuple(t.clone() for t in inputs)
        side = torch.cuda.Stream(device) if device.type == "cuda" else None
        if side is not None:
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                for _ in range(2):
                    warm_out = fn(*static_in)
            torch.cuda.current_stream(device).wait_stream(side)
        else:
            for _ in range(2):
                warm_out = fn(*static_in)

        g = _new_graph(device)
        kw = {} if pool is None else {"pool": pool}
        if device.type == "cuda":
            with torch.cuda.graph(g, **kw):
                out = fn(*static_in)
        else:
            with g:
                out = fn(*static_in)
        return Replayer(g, static_in, out, keep=fn)
    except Exception:
        return None


def _new_graph(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.CUDAGraph()
    return torch.accelerator.Graph()


class Cache:
    """Per-lane shape-keyed graph store. Entries share `pool` so the
    cache's memory footprint stays ~one working set."""

    def __init__(self, device: torch.device, max_entries: int = 8) -> None:
        self.device = device
        self.pool = pool_handle(device)
        self.max = max_entries
        self._d: dict[tuple, Replayer | None] = {}

    def get(self, key: tuple, fn,
            inputs: tuple[torch.Tensor, ...]) -> Replayer | None:
        """Replayer for `key` — None marker means 'already tried, use
        eager', so capture is attempted at most once per shape."""
        if key in self._d:
            return self._d[key]
        if len(self._d) >= self.max:
            return None
        try:
            rep = capture(fn, inputs, self.device, pool=self.pool)
        except Exception:
            rep = None   # capture machinery itself failed → eager
        self._d[key] = rep
        return rep
