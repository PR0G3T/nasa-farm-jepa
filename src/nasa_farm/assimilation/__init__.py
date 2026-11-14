"""Data assimilation — ensemble analyses and observation-schedule design.

`kalman`: stochastic/deterministic EnKF + the online assimilation driver.
`score`: ensemble score filter (EnSF) — backward-SDE analysis with a
training-free Monte-Carlo score (Bao, Zhang & Zhang 2023).
`latent`: EnSF-LR nonlinear obs-space analysis + learned energy-score
filter (Filter/fit_filter).
`langevin`: physics-exact gradient-guided posterior on initial states.
`design`: Bayesian experimental design — EIG ranking and ordered
acquisition calendars.
`smooth`: ensemble-subspace variational smoother (joint state+trait
4DEnVar through the differentiable rollout).
"""

from nasa_farm.assimilation.kalman import (
    Observation, STATE_VARS, assimilate, kalman_update,
)
from nasa_farm.assimilation.error import Statistic, inflate
from nasa_farm.assimilation.score import score_update
from nasa_farm.assimilation.latent import (
    Filter, fit_filter, network_update, score_langevin_update,
)
from nasa_farm.assimilation.langevin import langevin_posterior
from nasa_farm.assimilation.design import (
    design_schedule, eigen, fisher_design, plan_schedule,
)
from nasa_farm.assimilation.smooth import ensemble_smooth

__all__ = [
    "Filter", "Observation", "STATE_VARS", "Statistic", "assimilate",
    "design_schedule", "eigen", "ensemble_smooth", "fisher_design",
    "fit_filter", "inflate", "kalman_update", "langevin_posterior",
    "network_update", "plan_schedule", "score_langevin_update",
    "score_update",
]
