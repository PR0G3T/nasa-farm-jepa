"""Hardware-adaptive execution — one concept per module.

- ``bootstrap``    — allocator/env bootstrap at import (expandable segments, THP)
- ``capabilities`` — compute-lane discovery across every accelerator backend
- ``topology``     — physical-core topology, NUMA map, worker affinity
- ``memory``       — per-device memory + emergency OOM reserve pools
- ``pool``         — worker executors: threads (no-GIL) or interpreters (GIL)
- ``streams``      — per-lane compute/copy streams + pinned staging
- ``graphs``       — accelerator graph capture/replay for fixed-shape blocks
- ``probe``        — measured device rates + persistent cost cache
- ``telemetry``    — optional NVML power/temperature/throttle monitor
- ``scheduler``    — co-execution scheduler over all lanes + OOM ladder
"""

from nasa_farm.hardware import bootstrap as _bootstrap  # noqa: F401  (env)
