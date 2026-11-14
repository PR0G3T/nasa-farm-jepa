"""Forcing ensembles — scenario construction for counterfactual rollouts.

`base`: deterministic climatology + seeded perturbation ensembles.
`gen`: energy-score-trained noise-injection generator (AR daily weather).
`provide`: provider registry — one call dispatches to POWER, synthetic,
file or generated ensembles.
`analog`: historical-year weighting by climate-analogue features.
`bias`: quantile-mapping bias correction for forecast products, plus
ensemble probabilistic correction (mean offset + spread recalibration).
"""

from nasa_farm.forcing.base import (
    augment, climatology_ensemble, ensemble_of, ou_noise, synthetic_year,
    tile_to_horizon,
)
from nasa_farm.forcing.analog import analog_weights, weighted_ensemble
from nasa_farm.forcing.bias import (
    EnsBC, apply_ensbc, apply_qmap, fit_ensbc, fit_qmap,
)
from nasa_farm.forcing.gen import WxGen, fit_gen, sample_gen
from nasa_farm.forcing.provide import ensemble, register

__all__ = [
    "EnsBC", "WxGen", "analog_weights", "apply_ensbc", "apply_qmap",
    "augment", "climatology_ensemble", "ensemble", "ensemble_of",
    "fit_ensbc", "fit_gen", "fit_qmap", "ou_noise", "register",
    "sample_gen", "synthetic_year", "tile_to_horizon", "weighted_ensemble",
]
