"""Compute-lane discovery: every usable device as a first-class endpoint.

A ``Lane`` is one place work can run — a CPU worker pool or one
accelerator device (CUDA/ROCm, XPU, MPS, …) — carrying the capabilities
the scheduler needs: dtype support, free memory, and a measured rate.
Discovery goes through ``torch.accelerator`` so every backend PyTorch
ships is considered, not just CUDA.
"""

from __future__ import annotations

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.hardware.topology import physical_cores


def accelerators() -> list[torch.device]:
    """Every accelerator device visible at runtime, fully indexed."""
    if not torch.accelerator.is_available():
        return []
    accumulate = torch.accelerator.current_accelerator()
    if accumulate is None:
        return []
    return [torch.device(accumulate.type, i)
            for i in range(torch.accelerator.device_count())]


def supports(dtype: torch.dtype, device: torch.device) -> bool:
    """Whether `device` can natively run `dtype` (MPS lacks float64; an
    actual probe alloc settles anything the table doesn't know)."""
    if device.type == "mps" and dtype in (torch.float64, torch.complex128):
        return False
    return True


def device_memory(device: torch.device) -> tuple[int, int]:
    """(free, total) bytes on `device`; (0, 0) when the backend can't say."""
    if device.type == "cpu":
        return _avail_ram(), _avail_ram()
    try:
        free, total = torch.accelerator.memory.get_memory_info(device)
        return int(free), int(total)
    except (AttributeError, RuntimeError, TypeError):
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            return int(free), int(total)
        return 0, 0


def _avail_ram() -> int:
    """Bytes of *currently available* host RAM (Linux → POSIX → 2 GiB)."""
    import os
    from pathlib import Path
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return (os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 2 << 30


def name(device: torch.device) -> str:
    """Stable human label for a device (cache keys, telemetry)."""
    if device.type == "cpu":
        return "cpu"
    try:
        props = torch.cuda.get_device_properties(device) \
            if device.type == "cuda" else None
        return f"{device.type}:{device.index} {props.name}" \
            if props else f"{device.type}:{device.index}"
    except RuntimeError:
        return f"{device.type}:{device.index}"


def lanes(dtype: torch.dtype = DTYPE) -> list[torch.device]:
    """Usable compute devices, accelerators first then CPU — CPU is the
    always-present tail lane so a failed accelerator never dead-ends."""
    devs = [d for d in accelerators() if supports(dtype, d)]
    devs.append(torch.device("cpu"))
    return devs


def processor_workers(rows: int | None = None, *,
                min_rows_per_shard: int = 64) -> int:
    """Physical cores on no-GIL builds; 1 on GIL builds (the interpreter
    pool restores parallelism there — see `hw.pool`). Floored so shards
    stay bigger than coordination cost."""
    import sys
    cap = physical_cores()
    if rows is not None:
        cap = min(cap, max(1, rows // min_rows_per_shard))
    return max(1, cap)
