"""Data assimilation — ensemble analyses and observation-schedule design.

`kalman`: stochastic/deterministic EnKF + the online assimilation driver.
`score`: kernel ensemble score filtering (EnSF) transport analysis.
`design`: Bayesian experimental design — pick observation days/channels by
expected information gain.
"""

from nasa_farm.assim.kalman import (
    Observation, STATE_VARS, assimilate, enkf_update,
)
from nasa_farm.assim.score import ensf_update
from nasa_farm.assim.design import design_schedule, eig

__all__ = [
    "Observation", "STATE_VARS", "assimilate", "design_schedule", "eig",
    "enkf_update", "ensf_update",
]
