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

import numpy as np
import xarray as xr


def _smap_conf(flag: np.ndarray) -> np.ndarray:
    """bit0 recommended, bit1 attempted, bit2 successful → {1.0, 0.4, 0.0}."""
    f = np.asarray(flag, np.int64)
    # bit0=0 recommended → 1.0; attempted (bit1=0) AND successful (bit2=0)
    # but not recommended → 0.4; anything else → 0.
    out = np.where((f & 7) == 0, 1.0,
                   np.where((f & 6) == 0, 0.4, 0.0))
    return np.where((f >= 0) & (f < 256), out, 0.0)


def _strict_conf(flag: np.ndarray) -> np.ndarray:
    """Generic pass/flag band: 0 → full confidence, else uncertain."""
    f = np.asarray(flag, np.float64)
    return np.where(f == 0, 1.0, np.where(np.isfinite(f), 0.4, 0.0))


@dataclass(frozen=True, slots=True)
class QcSpec:
    """QA band declaration: candidate names + flag → confidence map."""

    vars: tuple[str, ...]
    to_conf: Callable[[np.ndarray], np.ndarray] = _strict_conf

    def mask(self, ds: xr.Dataset) -> np.ndarray | None:
        """(H, W) confidence map, or None when the product has no QA band."""
        for v in self.vars:
            if v in ds:
                return np.asarray(self.to_conf(ds[v].values), np.float64)
        return None


SMAP_QC = QcSpec(
    vars=("retrieval_qual_flag", "retrieval_quality_flag",
          "retrieval_qual_flag_pm"),
    to_conf=_smap_conf)
"""SPL3SMP_E: recommended 0/8 → 1.0, attempted-uncertain → 0.4, skip → 0."""

FLAG_QC = QcSpec(vars=("QC", "qc", "quality_flag", "mask"))
"""Generic single-band QA (NISAR): clean → 1.0, flagged → 0.4.
Not valid for bit-packed bands (ECOSTRESS `QC`): use CLOUD_QC there."""


def _cloud_conf(flag: np.ndarray) -> np.ndarray:
    """Binary cloud mask: 0 clear → 1.0; cloud or missing → 0."""
    f = np.asarray(flag, np.float64)
    return np.where(f == 0, 1.0, 0.0)


CLOUD_QC = QcSpec(vars=("cloud",), to_conf=_cloud_conf)
"""ECOSTRESS v002 tiles: `_cloud.tif` asset (0 clear / 1 cloud)."""
