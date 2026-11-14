"""Forcing ensembles — scenario construction for counterfactual rollouts.

`base`: deterministic climatology + seeded perturbation ensembles.
`gen`: conditional-flow-matching generator (AR daily weather).
`provide`: provider registry — one call dispatches to POWER, synthetic,
file or generated ensembles.
`analog`: historical-year weighting by climate-analogue features.
`bias`: quantile-mapping bias correction for forecast products, plus
ensemble probabilistic correction (mean offset + spread recalibration).
`reconcile`: precision-weighted multi-source consensus + spread.
`shuffle`: ensemble copula coupling — restores cross-variable and
temporal dependence to marginally postprocessed members.
`seam`: forecast-to-climatology stitching — analog-weighted member
continuations align operational ensembles to any horizon.
"""

from nasa_farm.forcing.base import (
    augment, climatology_ensemble, ensemble_of, ou_noise, synthetic_year,
    tile_to_horizon,
)
from nasa_farm.forcing.analog import analog_weights, weighted_ensemble
from nasa_farm.forcing.bias import (
    EnsBC, apply_ensbc, apply_qmap, fit_ensbc, fit_qmap,
)
from nasa_farm.forcing.seam import bridge, roll_to_doy, span
from nasa_farm.forcing.generation import WxGen, fit_gen, sample_gen
from nasa_farm.forcing.provide import ensemble, register
from nasa_farm.forcing.reconcile import consensus
from nasa_farm.forcing.shuffle import ecc, ranks, shuffle_forcing

__all__ = [
    "EnsBC", "WxGen", "analog_weights", "apply_ensbc", "apply_qmap",
    "augment", "bridge", "climatology_ensemble", "consensus", "ecc",
    "ensemble", "ensemble_of", "fit_ensbc", "fit_gen", "fit_qmap",
    "ou_noise", "ranks", "register", "roll_to_doy", "sample_gen",
    "shuffle_forcing", "span", "synthetic_year", "tile_to_horizon",
    "weighted_ensemble",
]
