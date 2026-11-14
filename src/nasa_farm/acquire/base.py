"""Acquisition contracts — one uniform interface per data source.

`fetch` is intentionally generic: each source returns its natural payload
(xarray Dataset for gridded products, Forcing for POWER). Callers never
branch on source identity beyond the declared return type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class BBox:
    west: float
    south: float
    east: float
    north: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)


@dataclass(frozen=True)
class Temporal:
    start: str  # YYYY-MM-DD
    end: str


@runtime_checkable
class DataSource(Protocol):
    """One dataset = one class implementing this. DRY/SRP boundary."""

    name: str
    requires_auth: bool

    def fetch(self, aoi: BBox | None, temporal: Temporal, **kw) -> Any: ...
