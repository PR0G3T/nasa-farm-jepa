"""CPU topology: physical cores, NUMA grouping, worker affinity.

Dispatch-bound rollouts gain nothing from SMT siblings, so workers are
counted and pinned on physical cores only. Topology is read from sysfs
(Linux) with portable fallbacks; affinity uses ``os.sched_setaffinity``
per worker thread and silently degrades where unsupported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_SYS = Path("/sys/devices/system/cpu")


@dataclass(frozen=True, slots=True)
class Topo:
    """Physical-core map. ``ids`` = one cpu index per physical core,
    grouped by NUMA node (inner lists share a memory domain)."""
    ids: tuple[int, ...]
    numa: tuple[tuple[int, ...], ...]

    @property
    def cores(self) -> int:
        return len(self.ids)


def _sysfs_topo() -> Topo | None:
    """(package, core_id) → representative cpu index, NUMA-grouped."""
    try:
        seen: dict[tuple[str, str], int] = {}
        node_of: dict[int, int] = {}
        for cpu in _SYS.glob("cpu[0-9]*"):
            top = cpu / "topology"
            pkg = (top / "physical_package_id").read_text().strip()
            core = (top / "core_id").read_text().strip()
            i = int(cpu.name[3:])
            seen.setdefault((pkg, core), i)
            nodes = list(cpu.glob("node[0-9]*"))
            node_of[i] = int(nodes[0].name[4:]) if nodes else 0
        if not seen:
            return None
        groups: dict[int, list[int]] = {}
        for i in seen.values():
            groups.setdefault(node_of.get(i, 0), []).append(i)
        numa = tuple(tuple(sorted(v)) for _, v in sorted(groups.items()))
        return Topo(ids=tuple(sorted(seen.values())), numa=numa)
    except (OSError, ValueError):
        return None


def topo() -> Topo:
    """Best-known physical topology (cached)."""
    global _TOPO
    if _TOPO is None:
        t = _sysfs_topo()
        if t is None:
            n = max(1, (os.cpu_count() or 2) // 2)
            t = Topo(ids=tuple(range(n)), numa=(tuple(range(n)),))
        _TOPO = t
    return _TOPO


_TOPO: Topo | None = None


def physical_cores() -> int:
    """Number of physical cores — what thread-parallel work should use."""
    return topo().cores


def pin(cpus: tuple[int, ...]) -> bool:
    """Pin the calling thread to the given cpu ids. Returns success."""
    if not cpus:
        return False
    try:
        os.sched_setaffinity(0, set(cpus))
        return True
    except (OSError, AttributeError):
        return False


def worker_cpus(worker: int, workers: int) -> tuple[int, ...]:
    """The physical core for worker ``worker`` — round-robin across NUMA
    nodes first (memory bandwidth), then depth-first within each node."""
    nodes = topo().numa
    order = [c for i in range(max(len(n) for n in nodes))
             for n in nodes if i < len(n) for c in (n[i],)]
    return (order[worker % len(order)],)
