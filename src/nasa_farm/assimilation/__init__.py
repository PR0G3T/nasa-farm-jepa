"""Data assimilation — ensemble analyses and observation-schedule design.

`kalman`: stochastic/deterministic EnKF + the online assimilation driver.
`score`: kernel ensemble score filtering (EnSF) transport analysis.
`latent`: EnSF-LR nonlinear obs-space analysis + learned energy-score
filter (FilterNet/fit_filter).
`langevin`: physics-exact gradient-guided posterior on initial states.
`design`: Bayesian experimental design — EIG ranking and ordered
acquisition calendars.
`smooth`: ensemble-subspace variational smoother (joint state+trait
4DEnVar through the differentiable rollout).
"""

from nasa_farm.assimilation.kalman import (
    Observation, STATE_VARS, assimilate, enkf_update,
)
from nasa_farm.assimilation.error import RStat, aoei
from nasa_farm.assimilation.score import ensf_update
from nasa_farm.assimilation.latent import (
    FilterNet, ensf_lr_update, fit_filter, net_update,
)
from nasa_farm.assimilation.langevin import ula_posterior
from nasa_farm.assimilation.design import design_schedule, eig, plan_schedule
from nasa_farm.assimilation.smooth import esv_smooth

__all__ = [
    "FilterNet", "Observation", "RStat", "STATE_VARS", "aoei",
    "assimilate", "design_schedule", "eig", "enkf_update",
    "ensf_lr_update", "ensf_update", "esv_smooth", "fit_filter",
    "net_update", "plan_schedule", "ula_posterior",
]
