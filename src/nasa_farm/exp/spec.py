"""Experiment specification — the whole protocol in one immutable object.

A `Spec` fully determines an experiment: site, soil, crop, forcing source
(real POWER years or seeded synthetic), ensemble sizes, observation
channels and cadence, truth-model perturbations, fit budget and seed.
`as_dict`/`from_dict` make it JSON/TOML-serializable, so every `Report`
carries its complete provenance and any run can be replayed bit-for-bit.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

from nasa_farm.state.soil import DEPTHS


@dataclass(frozen=True, slots=True)
class Spec:
    lat: float = 42.0
    lon: float = -93.6
    soil: str = "silt_loam"
    depths: tuple[float, ...] = DEPTHS
    crop: str = "maize"
    horizon: int = 1                     # simulated years per rollout
    years: tuple[int, ...] = ()          # POWER years; () → synthetic
    realizations: int = 1                # held-out truth realizations scored
    members: int = 16                    # forcing-ensemble / DA members
    obs_every: int = 10                  # days between obs (assim)
    noise: float = 0.05                  # relative observation noise
    perturb: dict[str, float] = field(default_factory=dict)
    """truth = prior traits × scale, e.g. {"rue": 0.8} — the twin gap."""
    obs_channels: tuple[str, ...] = ("ndvi", "sm1", "vwc")
    """satellite channels for assimilation (observe.CHANNELS names)."""
    channels: tuple[str, ...] = ("yield_kg",)
    """seasonal METRICS channels fitted and scored (heads.METRICS names)."""
    alphas: tuple[float, ...] = (0.1,)   # interval levels for PICP/scores
    latent_dim: int = 8                  # JEPA latent size (0 disables)
    ema: float = 0.996                   # target-encoder momentum
    sig_weight: float = 1.0              # SIGReg strength
    epochs: int = 120
    lr: float = 0.05
    seed: int = 0

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Spec":
        known = {f.name for f in dataclasses.fields(cls)}
        kw = {k: (tuple(v) if isinstance(v, list) else v)
              for k, v in d.items() if k in known}
        return cls(**kw)

    def to_json(self) -> str:
        return json.dumps(self.as_dict())

    @classmethod
    def from_json(cls, s: str) -> "Spec":
        return cls.from_dict(json.loads(s))
