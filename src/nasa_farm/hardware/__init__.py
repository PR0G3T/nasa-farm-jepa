"""Hardware-adaptive execution — one concept per module.

- ``boot``   — allocator/env bootstrap at import (expandable segments, THP)
- ``caps``   — compute-lane discovery across every accelerator backend
- ``topo``   — physical-core topology, NUMA map, worker affinity
- ``mem``    — per-device memory + emergency OOM reserve pools
- ``pool``   — worker executors: threads (no-GIL) or interpreters (GIL)
- ``pipe``   — per-lane compute/copy streams + pinned staging
- ``graphs`` — accelerator graph capture/replay for fixed-shape blocks
- ``probe``  — measured device rates + persistent cost cache
- ``thermo`` — optional NVML power/temperature/throttle monitor
- ``sched``  — co-execution scheduler over all lanes + OOM ladder
"""

from nasa_farm.hw import boot as _boot  # noqa: F401  (env side effects)
