"""Crop parameter library.

All crop traits live in a single (C, P) tensor so every downstream module
indexes it uniformly — adding a crop is a data change, never a code change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    "pp_sens",      # — photoperiod sensitivity (development delay per h past critical)
    "crit_daylen",  # h — critical day length for photoperiod response
    "vern_req",     # d — chilling days required before reproductive growth
    "gdd_fl",       # — fraction of gdd_mat where the reproductive window opens
    "heat_t",       # °C — tmax threshold for reproductive heat damage
    "heat_sus",     # — harvest-index decay rate per °C·d above heat_t
    "t_kill",       # °C — tmin below which the standing crop is killed
    "co2_resp",     # — RUE elasticity to CO2 (C3 ≈ 0.2, C4 ≈ 0.05)
)
P = len(PARAMS)
INDEX = {name: i for i, name in enumerate(PARAMS)}

_LIBRARY: dict[str, list[float]] = {
    #            tbase topt gdd_em gdd_lai gdd_sen gdd_mat lai  kext rue  kc   rf2  leg fix  ndem   HI   res  sow  pp   crit vern fl   heat sus   kill  co2
    "maize":    [8.0,  30.0, 80.0, 700.0, 1500.0, 1750.0, 5.5, 0.6, 3.9, 1.2, 0.6, 0.0, 0.0, 0.018, 0.52, 0.6, 120, 0.08, 12.5, 0.0, 0.60, 34.0, 0.04, -2.0, 0.05],
    "soybean":  [10.0, 30.0, 90.0, 800.0, 1100.0, 1650.0, 5.0, 0.5, 2.4, 1.15, 0.55, 1.0, 0.75, 0.055, 0.40, 0.5, 140, 0.30, 13.5, 0.0, 0.55, 35.0, 0.03, -2.0, 0.20],
    "wheat":    [0.0,  21.0, 100.0, 600.0, 1900.0, 2100.0, 6.0, 0.5, 2.8, 1.15, 0.6, 0.0, 0.0, 0.022, 0.46, 0.4, 285, 0.15, 14.0, 45.0, 0.62, 31.0, 0.05, -15.0, 0.20],
    "alfalfa":  [5.0,  30.0, 60.0, 600.0, 1e9,    1e9,    6.5, 0.55, 2.6, 1.2, 0.7, 1.0, 0.80, 0.030, 0.90, 0.9, 100, 0.10, 13.0, 0.0, 0.60, 36.0, 0.0,  -20.0, 0.22],
    "sorghum":  [8.0,  32.0, 80.0, 650.0, 1100.0, 1650.0, 4.5, 0.55, 3.6, 1.05, 0.65, 0.0, 0.0, 0.016, 0.48, 0.5, 135, 0.25, 12.5, 0.0, 0.60, 36.0, 0.02, -2.0, 0.05],
    "clover":   [4.0,  24.0, 70.0, 500.0, 1e9,    1e9,    4.0, 0.6, 2.2, 1.0, 0.5, 1.0, 0.85, 0.040, 0.0, 1.0, 250, 0.10, 13.0, 0.0, 0.60, 33.0, 0.0,  -20.0, 0.20],
    "fallow":   [0.0,  25.0, 1e9,  1e9,   1e9,    1e9,    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   0.0, 0.0, 1,   0.0,  12.0, 0.0, 1.00, 99.0, 0.0,  -99.0, 0.0],
}

PERENNIAL = {"alfalfa", "clover"}  # gdd_sen/gdd_mat=1e9 → never matures in-season


@dataclass(frozen=True, slots=True)
class CropLibrary:
    """Immutable crop trait table (C, P) plus name index.

    `var`: optional (C, Dv) variety embedding table — the G×E axis.
    Cultivar identity rides the latent's ctx input (WorldModel.variety_embedding)
    so two cultivars of the same crop can learn different residuals and
    dynamics without touching the physics traits."""

    table: torch.Tensor  # (C, P)
    names: tuple[str, ...]
    variety: torch.Tensor | None = None  # (C, Dv) variety embeddings

    @classmethod
    def default(cls) -> "CropLibrary":
        names = tuple(_LIBRARY)
        table = torch.tensor([_LIBRARY[n] for n in names], dtype=DTYPE)
        return cls(table=table, names=names)

    def with_var(self, dim: int,
                 seed: int | None = None) -> "CropLibrary":
        """Attach a small learnable variety embedding per crop."""
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        variety = 0.02 * torch.randn(self.table.shape[0], dim,
                                 generator=g, dtype=DTYPE)
        return replace(self, variety=variety)

    def calibrated(self, overrides: dict[str, dict[str, float]]) -> "CropLibrary":
        """New library with per-crop trait overrides applied — the
        `{crop: {param: value}}` contract `nasa-farm traits` writes to
        `lake/traits.json`. Unknown crops/params are ignored."""
        table = self.table.clone()
        for crop, params in overrides.items():
            if crop not in self.names:
                continue
            for param, value in params.items():
                if param in INDEX:
                    table[self.index(crop), INDEX[param]] = float(value)
        return replace(self, table=table)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def get(self, name: str) -> torch.Tensor:
        return self.table[self.index(name)]

    def column(self, param: str) -> torch.Tensor:
        return self.table[:, INDEX[param]]

    @property
    def n_crops(self) -> int:
        return self.table.shape[0]

    def is_perennial(self, crop_idx: torch.Tensor) -> torch.Tensor:
        return self.table[crop_idx.clamp(min=0), INDEX["gdd_mat"]] > 1e8

    def mixture(self, name: str, a: str, b: str,
                fraction: float = 0.5) -> "CropLibrary":
        """Append a composite cultivar row = frac·a + (1−frac)·b.

        Mixtures and intercrops enter the model as data, not code: the
        blended row is sown by its own index like any crop. Continuous
        traits interpolate; sow_doy rounds to the weighted day.
        """
        i, j = self.index(a), self.index(b)
        row = fraction * self.table[i] + (1.0 - fraction) * self.table[j]
        row = row.clone()
        row[INDEX["sow_doy"]] = row[INDEX["sow_doy"]].round()
        table = torch.cat([self.table, row.unsqueeze(0)], dim=0)
        variety = None if self.variety is None else torch.cat(
            [self.variety, (fraction * self.variety[i] + (1.0 - fraction) * self.variety[j])
             .unsqueeze(0)], dim=0)
        return CropLibrary(table=table, names=self.names + (name,), variety=variety)
