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
import xarray as xarray


def _smap_conf(flag: np.ndarray) -> np.ndarray:
    """bit0 recommended, bit1 attempted, bit2 successful → {1.0, 0.4, 0.0}."""
    f = np.asarray(flag, np.int64)
    output = np.where((f & 7) == 0, 1.0, np.where((f & 2) == 0, 0.4, 0.0))
    return np.where((f >= 0) & (f < 256), output, 0.0)


def _strict_conf(flag: np.ndarray) -> np.ndarray:
    """Generic pass/flag band: 0 → full confidence, else uncertain."""
    f = np.asarray(flag, np.float64)
    return np.where(f == 0, 1.0, np.where(np.isfinite(f), 0.4, 0.0))


@dataclass(frozen=True, slots=True)
class Quality:
    """QA band declaration: candidate names + flag → confidence map."""

    variables: tuple[str, ...]
    to_conf: Callable[[np.ndarray], np.ndarray] = _strict_conf

    def band(self, ds: xarray.Dataset) -> str | None:
        """First QA band name present, so callers can slice before masking."""
        return next((v for v in self.variables if v in ds), None)

    def mask(self, ds: xarray.Dataset) -> np.ndarray | None:
        """(H, W) confidence map, or None when the product has no QA band."""
        for v in self.variables:
            if v in ds:
                return np.asarray(self.to_conf(ds[v].values), np.float64)
        return None


SMAP_QC = Quality(
    variables=("retrieval_qual_flag", "retrieval_quality_flag", "retrieval_qual_flag_pm"),
    to_conf=_smap_conf,
)
"""SPL3SMP_E: recommended 0/8 → 1.0, attempted-uncertain → 0.4, skip → 0."""

FLAG_QC = Quality(variables=("QC", "qc", "quality_flag", "mask"))
"""Generic single-band QA (ECOSTRESS, NISAR): clean → 1.0, flagged → 0.4."""


def _fmask_conf(flag: np.ndarray) -> np.ndarray:
    """HLS Fmask bits: 1 cloud, 2 adjacent, 3 shadow, 4 snow/ice."""
    f = np.asarray(flag, np.int64)
    return np.where(
        (f >> 1) & 1, 0.0, np.where(((f >> 3) & 1) | ((f >> 4) & 1) | ((f >> 2) & 1), 0.4, 1.0)
    )


def _cloud_conf(flag: np.ndarray) -> np.ndarray:
    """ECOSTRESS final cloud mask / QC: 0 clear → 1.0; cloudy → 0."""
    f = np.asarray(flag, np.float64)
    return np.where(f == 0, 1.0, 0.0)


FMASK_QC = Quality(variables=("Fmask", "fmask"), to_conf=_fmask_conf)
"""HLSL30_VI/HLSS30_VI Fmask: clear land → 1.0, shadow/snow/adjacent →
0.4, cloud → 0."""
CLOUD_QC = Quality(variables=("cloud", "cloud_mask", "Cloud"), to_conf=_cloud_conf)
"""ECOSTRESS JET/LSTE cloud mask: clear → 1.0, else 0 (an instantaneous
TIR retrieval through cloud is unusable, not just degraded)."""
