import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.model import WorldModel
from nasa_farm.forcing import synthetic_year
from nasa_farm.state.crops import CropLibrary
from nasa_farm.state.field import Field, initial_state
from nasa_farm.state.soil import SoilProfile
from nasa_farm.types import Actions, Forcing


@pytest.fixture
def soil() -> SoilProfile:
    return SoilProfile.preset("loam")


@pytest.fixture
def crops() -> CropLibrary:
    return CropLibrary.default()


@pytest.fixture
def field(soil) -> Field:
    return Field(
        state=initial_state(1, soil), soil=soil, latitude=torch.tensor([42.0], dtype=DTYPE)
    )


@pytest.fixture
def physics_model(crops) -> WorldModel:
    return WorldModel(crops, residual=False)


@pytest.fixture
def hybrid_model(crops) -> WorldModel:
    return WorldModel(crops, residual=True)


@pytest.fixture
def warm_year() -> Forcing:
    return synthetic_year(42.0, tmean_annual=14.0, seed=7)


def idle(days: int, batch: int = 1) -> Actions:
    return Actions.idle(batch, days)


def day_forcing(
    tmin=5.0, tmax=20.0, srad=20.0, precip=0.0, rh=70.0, u2=2.5, doy=150.0, batch: int = 1
) -> dict:
    def t(v):
        return torch.full((batch,), float(v), dtype=DTYPE)

    return {
        "tmin": t(tmin),
        "tmax": t(tmax),
        "srad": t(srad),
        "precip": t(precip),
        "rh": t(rh),
        "u2": t(u2),
        "doy": t(doy),
    }


def day_action(sow: int = -1, irrigation: float = 0.0, n_fert: float = 0.0, batch: int = 1) -> dict:
    return {
        "sow": torch.full((batch,), sow, dtype=torch.long),
        "irrigation": torch.full((batch,), irrigation, dtype=DTYPE),
        "n_fert": torch.full((batch,), n_fert, dtype=DTYPE),
    }
