"""Crop parameter library.

All crop traits live in a single (C, P) tensor so every downstream module
indexes it uniformly — adding a crop is a data change, never a code change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nasa_farm.constants import DTYPE

PARAMS: tuple[str, ...] = (
    "tbase",        # °C — GDD base temperature
    "topt",         # °C — optimal growth temperature
    "gdd_em",       # °C·d — GDD to emergence
    "gdd_lai_max",  # °C·d — GDD to full canopy
    "gdd_sen",      # °C·d — GDD to senescence onset
    "gdd_mat",      # °C·d — GDD to maturity
    "lai_max",      # m²/m²
    "k_ext",        # — light extinction coefficient
    "rue",          # g DM / MJ intercepted PAR
    "kc_max",       # — peak crop coefficient
    "root_frac2",   # — fraction of water uptake from layer 2
    "legume",       # 0/1
    "fix_frac",     # — max fraction of N demand met by fixation
    "n_demand",     # kg N per kg biomass
    "harvest_index",# — grain yield / above-ground biomass
    "residue_frac", # — fraction of non-yield biomass retained on field
    "sow_doy",      # — default day-of-year of sowing
)
P = len(PARAMS)
IDX = {name: i for i, name in enumerate(PARAMS)}

_LIBRARY: dict[str, list[float]] = {
    #            tbase topt gdd_em gdd_lai gdd_sen gdd_mat lai  kext rue  kc   rf2  leg fix  ndem   HI   res  sow
    "maize":    [8.0,  30.0, 80.0, 700.0, 1200.0, 1750.0, 5.5, 0.6, 3.9, 1.2, 0.6, 0.0, 0.0, 0.018, 0.52, 0.6, 120],
    "soybean":  [10.0, 30.0, 90.0, 800.0, 1100.0, 1650.0, 5.0, 0.5, 2.4, 1.15, 0.55, 1.0, 0.75, 0.055, 0.40, 0.5, 140],
    "wheat":    [0.0,  25.0, 100.0, 900.0, 1500.0, 2100.0, 6.0, 0.5, 2.8, 1.15, 0.6, 0.0, 0.0, 0.022, 0.46, 0.4, 80],
    "alfalfa":  [5.0,  30.0, 60.0, 600.0, 1e9,    1e9,    6.5, 0.55, 2.6, 1.2, 0.7, 1.0, 0.80, 0.030, 0.90, 0.9, 100],
    "sorghum":  [8.0,  32.0, 80.0, 650.0, 1100.0, 1650.0, 4.5, 0.55, 3.6, 1.05, 0.65, 0.0, 0.0, 0.016, 0.48, 0.5, 135],
    "clover":   [4.0,  24.0, 70.0, 500.0, 1e9,    1e9,    4.0, 0.6, 2.2, 1.0, 0.5, 1.0, 0.85, 0.040, 0.0, 1.0, 250],
    "fallow":   [0.0,  25.0, 1e9,  1e9,   1e9,    1e9,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   0.0, 0.0, 1],
}

PERENNIAL = {"alfalfa", "clover"}  # gdd_sen/gdd_mat=1e9 → never matures in-season


@dataclass(frozen=True, slots=True)
class CropLibrary:
    """Immutable crop trait table (C, P) plus name index."""

    table: torch.Tensor  # (C, P)
    names: tuple[str, ...]

    @classmethod
    def default(cls) -> "CropLibrary":
        names = tuple(_LIBRARY)
        table = torch.tensor([_LIBRARY[n] for n in names], dtype=DTYPE)
        return cls(table=table, names=names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def get(self, name: str) -> torch.Tensor:
        return self.table[self.index(name)]

    def col(self, param: str) -> torch.Tensor:
        return self.table[:, IDX[param]]

    @property
    def n_crops(self) -> int:
        return self.table.shape[0]

    def is_perennial(self, crop_idx: torch.Tensor) -> torch.Tensor:
        return self.table[crop_idx.clamp(min=0), IDX["gdd_mat"]] > 1e8
