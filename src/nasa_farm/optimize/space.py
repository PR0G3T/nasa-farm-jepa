"""Search-space description + samplers — shared by black-box HPO and NAS.

Sampler ladder: Sobol (quasi-random, space-filling — strictly better than
random for coverage at fixed budget) first, plain random for the tail.
Categorical dimensions are drawn uniformly; there is no meaningful low-
discrepancy ordering on unordered categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class Parameter:
    """One dimension of the search space."""

    name: str
    low: float | None = None
    high: float | None = None
    log: bool = False                 # continuous: sample in log space
    integer: bool = False             # continuous range rounded to int
    choices: tuple[Any, ...] | None = None  # categorical

    def __post_init__(self) -> None:
        is_cat = self.choices is not None
        if is_cat != (self.low is None):  # exactly one of choices/(low,high)
            raise ValueError(
                f"Param {self.name!r}: either choices or (low, high) required")
        if not is_cat and self.high <= self.low:
            raise ValueError(f"Param {self.name!r}: high <= low")
        if not is_cat and self.log and self.low <= 0:
            raise ValueError(f"Param {self.name!r}: log space needs low > 0")

    @property
    def categorical(self) -> bool:
        return self.choices is not None

    def map(self, u: float) -> Any:
        """Unit-cube coordinate → parameter value."""
        if self.categorical:
            return self.choices[min(int(u * len(self.choices)), len(self.choices) - 1)]
        low, high = (torch.log10(torch.tensor(v)).item() for v in (self.low, self.high)) \
            if self.log else (self.low, self.high)
        v = low + u * (high - low)
        v = 10 ** v if self.log else v
        return int(round(v)) if self.integer else float(v)


class Space:
    """Ordered parameter set; `sample` returns name → value configs."""

    def __init__(self, parameters: list[Parameter]) -> None:
        self.parameters = parameters
        self.continuous = [p for p in parameters if not p.categorical]
        self.categorical = [p for p in parameters if p.categorical]

    @property
    def ndim(self) -> int:
        return len(self.parameters)

    def sample(self, n: int, seed: int = 0) -> list[dict[str, Any]]:
        """n configs: Sobol on continuous dims, uniform on categorical."""
        generation = torch.Generator().manual_seed(seed)
        cont_u = torch.rand(n, len(self.continuous), generator=generation) \
            if self.continuous else None
        if self.continuous:
            try:
                eng = torch.quasirandom.SobolEngine(
                    dimension=len(self.continuous), scramble=True, seed=seed)
                cont_u = eng.draw(n)
            except (RuntimeError, NotImplementedError):
                pass  # Sobol cap reached → uniform fallback
        cat_u = torch.rand(n, len(self.categorical), generator=generation) \
            if self.categorical else None
        output = []
        for i in range(n):
            config = {}
            for j, p in enumerate(self.continuous):
                config[p.name] = p.map(float(cont_u[i, j]))
            for j, p in enumerate(self.categorical):
                config[p.name] = p.map(float(cat_u[i, j]))
            output.append(config)
        return output
