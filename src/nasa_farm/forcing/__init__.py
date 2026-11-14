"""Forcing ensembles — scenario construction for counterfactual rollouts.

`base`: deterministic climatology + seeded perturbation ensembles.
`gen`: learned flow-matching generator (AR-conditioned daily weather).
`provide`: provider registry — one call dispatches to POWER, synthetic,
file or generated ensembles.
`analog`: historical-year weighting by climate-analogue features.
`bias`: quantile-mapping bias correction for forecast products.
"""

from nasa_farm.forcing.base import (
    augment, climatology_ensemble, ensemble_of, ou_noise, synthetic_year,
    tile_to_horizon,
)
from nasa_farm.forcing.analog import analog_weights, weighted_ensemble
from nasa_farm.forcing.bias import apply_qmap, fit_qmap
from nasa_farm.forcing.gen import FlowGen, fit_flow, sample_flow
from nasa_farm.forcing.provide import ensemble, register

__all__ = [
    "FlowGen", "analog_weights", "apply_qmap", "augment",
    "climatology_ensemble", "ensemble", "ensemble_of", "fit_flow",
    "fit_qmap", "ou_noise", "register", "sample_flow", "synthetic_year",
    "tile_to_horizon", "weighted_ensemble",
]
