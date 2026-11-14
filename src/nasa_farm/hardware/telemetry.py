"""Accelerator telemetry: power, temperature, throttling, energy.

A daemon thread samples the device via NVML (``nvidia-ml-py``, optional
extra ``nasa-farm[gpu]``) for the duration of a sweep. Two things come
out of it:

- ``joules()`` — energy actually consumed, so cost can be reported per
  field-day simulated, not just seconds elapsed.
- ``throttled`` — set when hardware reports thermal/power cap events;
  the scheduler watches it and re-probes lane rates mid-run instead of
  trusting measurements taken before the laptop started throttling.

Everything is optional: no NVML, no monitor, no overhead — the sampler
is injected in tests so the logic runs without a GPU.
"""

from __future__ import annotations

import threading
import time


class Monitor:
    """Periodic NVML sampler for one accelerator device."""

    def __init__(self, device, hz: float = 2.0,
                 sampler=None) -> None:
        self.device = device
        self.dt = 1.0 / max(hz, 0.1)
        self.throttled = threading.Event()
        self._stop = threading.Event()
        self._samples: list[tuple[float, float, float, int]] = []
        self._thread: threading.Thread | None = None
        self._nv, self._handle = (None, None)
        self._sampler = sampler or self._nvml_sampler()

    def _nvml_sampler(self):
        if self.device.type != "cuda":
            return None
        try:
            import pynvml as nv
            nv.nvmlInit()
            h = nv.nvmlDeviceGetHandleByIndex(self.device.index or 0)
        except Exception:
            return None

        def sample() -> tuple[float, float, int] | None:
            """(watts, celsius, throttle_reason_bits)."""
            try:
                w = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                w = 0.0
            try:
                t = nv.nvmlDeviceGetTemperature(h, 0)
            except Exception:
                t = 0.0
            try:
                r = nv.nvmlDeviceGetCurrentClocksThrottleReasons(h)
            except Exception:
                r = 0
            return w, t, int(r)
        return sample

    @property
    def available(self) -> bool:
        return self._sampler is not None

    def __enter__(self) -> "Monitor":
        if self._sampler is not None:
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="nasa-farm-thermo")
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            s = self._sampler()
            if s is not None:
                w, t, reasons = s
                self._samples.append((time.monotonic(), w, t, reasons))
                if reasons:
                    self.throttled.set()
            self._stop.wait(self.dt)

    def joules(self) -> float:
        """Integrated power over the sampling window (J)."""
        s = self._samples
        return sum(0.5 * (s[i][1] + s[i - 1][1]) * (s[i][0] - s[i - 1][0])
                   for i in range(1, len(s)))

    def stats(self) -> dict:
        s = self._samples
        temps = [x[2] for x in s] or [0.0]
        return {
            "joules": self.joules(),
            "temp_max_c": max(temps),
            "temp_last_c": temps[-1],
            "throttle_events": sum(1 for x in s if x[3]),
            "samples": len(s),
        }
