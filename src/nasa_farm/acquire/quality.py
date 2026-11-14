"""Product quality flags → per-pixel retrieval confidence.

Every granule product declares its QA band here once: flag values map to
a confidence in [0, 1] that scales the extraction kernel — recommended
retrievals count fully, uncertain ones partially, unusable ones not at
all. Products without a QA band fall back to finite-data confidence, so
they never fake a quality signal and never silently drop coverage.

SMAP: retrieval_qual_flag bits 0–2 (recommended / attempted / successful)
— values 0 and 8 are recommended quality (8 = freeze/thaw retrieval
failed, which does not degrade the soil-moisture retrieval).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as numpy
import xarray as xarray


def _smap_conf(flag: numpy.ndarray) -> numpy.ndarray:
    """bit0 recommended, bit1 attempted, bit2 successful → {1.0, 0.4, 0.0}."""
    f = numpy.asarray(flag, numpy.int64)
    output = numpy.where((f & 7) == 0, 1.0, numpy.where((f & 2) == 0, 0.4, 0.0))
    return numpy.where((f >= 0) & (f < 256), output, 0.0)


def _strict_conf(flag: numpy.ndarray) -> numpy.ndarray:
    """Generic pass/flag band: 0 → full confidence, else uncertain."""
    f = numpy.asarray(flag, numpy.float64)
    return numpy.where(f == 0, 1.0, numpy.where(numpy.isfinite(f), 0.4, 0.0))


@dataclass(frozen=True, slots=True)
class Quality:
    """QA band declaration: candidate names + flag → confidence map."""

    vars: tuple[str, ...]
    to_conf: Callable[[numpy.ndarray], numpy.ndarray] = _strict_conf

    def band(self, ds: xarray.Dataset) -> str | None:
        """First QA band name present, so callers can slice before masking."""
        return next((v for v in self.vars if v in ds), None)

    def mask(self, ds: xarray.Dataset) -> numpy.ndarray | None:
        """(H, W) confidence map, or None when the product has no QA band."""
        for v in self.vars:
            if v in ds:
                return numpy.asarray(self.to_conf(ds[v].values), numpy.float64)
        return None


SMAP_QC = Quality(
    vars=("retrieval_qual_flag", "retrieval_quality_flag",
          "retrieval_qual_flag_pm"),
    to_conf=_smap_conf)
"""SPL3SMP_E: recommended 0/8 → 1.0, attempted-uncertain → 0.4, skip → 0."""

FLAG_QC = Quality(vars=("QC", "qc", "quality_flag", "mask"))
"""Generic single-band QA (ECOSTRESS, NISAR): clean → 1.0, flagged → 0.4."""
