"""Learnable residual correction on top of the physics core.

Zero-initialized output layer ⇒ the hybrid model is exactly the physics core
at init (verified by tests). Outputs are tanh-bounded per variable and
re-clamped to hard physical bounds by the caller, so conservation-relevant
invariants cannot be violated arbitrarily.
"""

from __future__ import annotations

import torch
import torch.nn as nn

RESIDUAL_VARS: tuple[str, ...] = ("sw1", "sw2", "n_min", "lai", "biomass")
BOUND_SCALE = {"sw1": 5.0, "sw2": 5.0, "n_min": 2.0, "lai": 0.3, "biomass": 60.0}

N_FEATURES = 14  # see featurize()


def featurize(state_fields: dict[str, torch.Tensor],
              forcing: dict[str, torch.Tensor],
              action: dict[str, torch.Tensor]) -> torch.Tensor:
    """(B, N_FEATURES) input vector for the residual net."""
    cols = [
        state_fields["sw1"], state_fields["sw2"], state_fields["n_min"],
        state_fields["lai"], state_fields["biomass"], state_fields["gdd"],
        state_fields["growing"].to(state_fields["sw1"].dtype),
        forcing["tmin"], forcing["tmax"], forcing["srad"], forcing["precip"],
        action["irrigation"], action["n_fert"],
        action["sow"].to(state_fields["sw1"].dtype),
    ]
    scales = torch.tensor(
        [300.0, 500.0, 80.0, 6.0, 20_000.0, 2500.0, 1.0,
         30.0, 40.0, 30.0, 50.0, 20.0, 100.0, 6.0],
        dtype=state_fields["sw1"].dtype,
    )
    return torch.stack(cols, dim=-1) / scales


class ResidualNet(nn.Module):
    """MLP producing bounded per-variable deltas. Identity at init."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, len(RESIDUAL_VARS)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.to(torch.float64)
        scale = torch.tensor([BOUND_SCALE[v] for v in RESIDUAL_VARS], dtype=torch.float64)
        self.register_buffer("scale", scale)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(feats)) * self.scale
