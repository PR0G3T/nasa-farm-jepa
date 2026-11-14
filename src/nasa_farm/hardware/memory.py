"""Memory: per-device availability and OOM headroom.

Each accelerator lane keeps a small ``MemPool(use_on_oom=True)`` reserve:
when the regular allocator would fail, the caching allocator dips into
this pool — a last-resort cushion that turns a hard OOM into a
recoverable, ladder-managed event. CPU lanes budget from host RAM.
"""

from __future__ import annotations

import torch

from nasa_farm.hardware.capabilities import device_memory

_RESERVE_CAP = 512 << 20      # never more than 512 MiB held back
_RESERVE_FRAC = 0.02          # ~2 % of free VRAM


def available(device: torch.device, budget: float = 1.0) -> int:
    """Usable bytes on `device` — `budget` fraction of *currently free*."""
    free, _ = device_memory(device)
    return int(free * budget)


def reserve_pool(device: torch.device,
                 reserve: int | None = None) -> "torch.cuda.MemPool | None":
    """Emergency headroom pool for a CUDA lane. The pool holds a freed
    block of `reserve` bytes: on allocator OOM it is consumed instead of
    failing. Returns None on non-CUDA devices or unsupported builds."""
    if device.type != "cuda" or not hasattr(torch.cuda, "MemPool"):
        return None
    free, _ = device_memory(device)
    n = reserve if reserve is not None else int(
        min(_RESERVE_CAP, free * _RESERVE_FRAC))
    if n <= 0:
        return None
    try:
        pool = torch.cuda.MemPool(use_on_oom=True)
        with torch.cuda.use_mem_pool(pool, device):
            block = torch.empty(n, dtype=torch.uint8, device=device)
            del block
        return pool
    except (RuntimeError, AttributeError, TypeError):
        return None


def empty(device: torch.device) -> None:
    """Release cached-but-unused memory on `device` (post-ladder GC)."""
    try:
        torch.accelerator.empty_cache(device)
    except (RuntimeError, AttributeError, TypeError):
        pass
