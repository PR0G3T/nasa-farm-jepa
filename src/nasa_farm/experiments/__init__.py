"""Experiment harness — reproducible, serialized research runs.

Registered runners (`twin`, `bench`, `assim`, `jepa`, `interv`, `adapt`,
`cf`, `discover`, `scale`) each consume one `Spec` and return a metrics
tree; `run` wraps it in a `Report` with full provenance.
"""

from nasa_farm.experiments.run import Report, experiments, register, run
from nasa_farm.experiments.specification import Spec
from nasa_farm.experiments import (  # noqa: F401
    adapt, assimilation, benchmark, counterfactual, discover, embedding,
    intervention, scale, twin,
)

__all__ = [
    "Report", "Spec", "adapt", "assimilation", "benchmark",
    "counterfactual", "discover", "embedding", "experiments",
    "intervention", "register", "run", "scale", "twin",
]
