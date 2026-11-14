"""Experiment harness — reproducible, serialized research runs.

Registered runners (`twin`, `bench`, `assim`, `jepa`, `interv`) each
consume one `Spec` and return a metrics tree; `run` wraps it in a `Report`
with full provenance.
"""

from nasa_farm.exp.run import Report, experiments, register, run
from nasa_farm.exp.spec import Spec
from nasa_farm.exp import assim, bench, interv, jepa, twin  # noqa: F401

__all__ = [
    "Report", "Spec", "assim", "bench", "experiments", "interv", "jepa",
    "register", "run", "twin",
]
