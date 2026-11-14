"""Per-lane streams and pinned staging.

Each accelerator lane runs two streams: a compute stream and a copy
stream. Block inputs ride the copy stream (pinned host buffers →
non-blocking H2D), then the compute stream waits on a copy event —
input transfer of block i+1 overlaps compute of block i, and results
drain back over the copy stream while the next block runs.

``NASA_FARM_SMS`` optionally partitions the lane's SMs through a CUDA
green context (driver ≥ 13.1, ``cuda.bindings`` required): the lane
then shares the device with other work at guaranteed QoS instead of
competing for every SM.
"""

from __future__ import annotations

import os

import torch


def streams(device: torch.device) -> tuple[object, object]:
    """(compute, copy) streams for a lane — CPU lanes use Nones (host
    work needs no stream). Honors NASA_FARM_SMS=<n> SM partitioning via
    green contexts when the driver/bindings support it."""
    if device.type == "cpu":
        return None, None
    comp = _green_stream(device)
    if comp is None:
        comp = _stream(device)
    return comp, _stream(device)


def _stream(device: torch.device):
    """Dedicated stream where the backend exposes them (CUDA today;
    other accelerators fall back to the default stream — copy ordering
    is then handled by plain synchronization)."""
    return torch.cuda.Stream(device) if device.type == "cuda" else None


def _green_stream(device: torch.device):
    sms = os.environ.get("NASA_FARM_SMS")
    if not sms or device.type != "cuda":
        return None
    try:
        from torch.cuda.green_contexts import GreenContext
        ctx = GreenContext(num_sms=int(sms), device_id=device.index)
        return ctx.Stream()
    except Exception:
        return None


class Stage:
    """Pinned staging for one lane's block inputs.

    ``in_`` copies a host plans block into a pinned buffer then on the
    compute stream — the caller only gets a device tensor, transfer
    ordering is internal. Buffers grow to the largest block seen and are
    reused for the whole sweep.
    """

    def __init__(self, device: torch.device, copy_stream,
                 compute_stream) -> None:
        self.device = device
        self.copy_s = copy_stream
        self.comp_s = compute_stream
        self._pin: torch.Tensor | None = None

    def in_(self, block: torch.Tensor) -> torch.Tensor:
        """Host block → device tensor ready on the compute stream."""
        if self.device.type == "cpu" or self.copy_s is None:
            return block.to(self.device, non_blocking=True)
        if self._pin is None or self._pin.shape[0] < block.shape[0]:
            self._pin = torch.empty(
                block.shape, dtype=block.dtype, pin_memory=True)
        buf = self._pin[: block.shape[0]]
        buf.copy_(block)
        with torch.cuda.stream(self.copy_s):
            dev = buf.to(self.device, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(self.copy_s)
        self.comp_s.wait_event(ev)
        dev.record_stream(self.comp_s)
        return dev
