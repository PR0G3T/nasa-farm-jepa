"""nasa-farm — physics-informed latent world model for crop-rotation
decision support on NASA Earth observations."""

from nasa_farm.dynamics.model import WorldModel
from nasa_farm.landscape import FieldGraph, Landscape
from nasa_farm.state.crops import CropLibrary
from nasa_farm.state.field import Field, FieldState, initial_state
from nasa_farm.state.soil import SoilProfile
from nasa_farm.types import Actions, Forcing

__version__ = "0.1.0"

__all__ = [
    "Actions", "CropLibrary", "Field", "FieldGraph", "FieldState",
    "Forcing", "Landscape", "SoilProfile", "WorldModel", "initial_state",
]
