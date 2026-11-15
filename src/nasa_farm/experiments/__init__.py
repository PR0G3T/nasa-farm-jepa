"""Experiment harness — reproducible, serialized research runs.

Registered runners (`twin`, `bench`, `assim`, `jepa`, `interv`, `adapt`,
`cf`, `discover`, `scale`, `design`) each consume one `Spec` and return a metrics
tree; `run` wraps it in a `Report` with full provenance.
"""

from nasa_farm.experiments import (
    adapt,
    assimilation,
    benchmark,
    counterfactual,
    design,
    discover,
    embedding,
    intervention,
    realdata,
    scale,
    twin,
)
from nasa_farm.experiments.run import Report, experiments, register, run
from nasa_farm.experiments.specification import Specification

__all__ = [
    "Report",
    "Specification",
    "adapt",
    "assimilation",
    "benchmark",
    "counterfactual",
    "design",
    "discover",
    "embedding",
    "experiments",
    "intervention",
    "realdata",
    "register",
    "run",
    "scale",
    "twin",
]
