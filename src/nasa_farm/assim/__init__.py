"""Data assimilation — ensemble analyses and observation-schedule design.

`kalman`: stochastic/deterministic EnKF + the online assimilation driver.
`score`: kernel ensemble score filtering (EnSF) transport analysis.
`latent`: EnSF-LR nonlinear obs-space analysis + learned energy-score
filter (FilterNet/fit_filter).
`langevin`: physics-exact gradient-guided posterior on initial states.
`design`: Bayesian experimental design — EIG ranking and ordered
acquisition calendars.
"""

from nasa_farm.assim.kalman import (
    Observation, STATE_VARS, assimilate, enkf_update,
)
from nasa_farm.assim.score import ensf_update
from nasa_farm.assim.latent import (
    FilterNet, ensf_lr_update, fit_filter, net_update,
)
from nasa_farm.assim.langevin import ula_posterior
from nasa_farm.assim.design import design_schedule, eig, plan_schedule

__all__ = [
    "FilterNet", "Observation", "STATE_VARS", "assimilate",
    "design_schedule", "eig", "enkf_update", "ensf_lr_update",
    "ensf_update", "fit_filter", "net_update", "plan_schedule",
    "ula_posterior",
]
