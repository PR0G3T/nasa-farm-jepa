"""Import-time environment bootstrap for hardware-aware execution.

Sets allocator and host-side defaults before the first CUDA allocation
and before torch's allocator options are read. Every knob is merged into
the user's existing configuration — explicitly set values always win.

- ``PYTORCH_ALLOC_CONF=expandable_segments:True``: virtual-memory-backed
  CUDA segments grow in place instead of allocating one cudaMalloc region
  per size class — fragmentation (the dominant OOM cause under streamed,
  variable-size plan blocks) is structurally impossible across segments.
- ``THP_MEM_ALLOC_ENABLE=1``: CPU tensor allocations >= 2 MiB are backed
  by transparent huge pages (``madvise``) — fewer TLB misses and faster
  first-touch for the large forcing/action/state slabs.
- ``PYTORCH_NVML_BASED_CUDA_CHECK=1``: ``is_available`` checks go through
  NVML rather than the CUDA runtime, keeping ``fork()`` unpolluted for
  subprocess-spawning services.
"""

from __future__ import annotations

import os

_ALLOC_DEFAULTS = {"expandable_segments": "True"}


def _merge_allocation_config() -> None:
    current = os.environ.get("PYTORCH_ALLOC_CONF") or \
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    kv = dict(
        pair.split(":", 1) for pair in current.split(",") if ":" in pair)
    if kv.get("backend") == "cudaMallocAsync":
        return  # foreign allocator — native-only keys must not be injected
    kv = {**_ALLOC_DEFAULTS, **kv}
    os.environ["PYTORCH_ALLOC_CONF"] = ",".join(
        f"{k}:{v}" for k, v in kv.items())


def bootstrap() -> None:
    _merge_allocation_config()
    os.environ.setdefault("THP_MEM_ALLOC_ENABLE", "1")
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")


bootstrap()
