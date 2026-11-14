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
        dtype=state_fields["sw1"].dtype, device=state_fields["sw1"].device,
    )
    return torch.stack(cols, dim=-1) / scales


ACTS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU, "gelu": nn.GELU, "elu": nn.ELU, "tanh": nn.Tanh,
}


class ResidualNet(nn.Module):
    """Configurable MLP producing bounded per-variable deltas.

    Defaults reproduce the original head (2×64 SiLU hidden layers). The
    output layer is zero-initialized ⇒ identity at init regardless of the
    hidden architecture (verified by tests).
    """

    def __init__(self, hidden: int = 64, depth: int = 2,
                 activation: str = "silu", layernorm: bool = False,
                 skip: bool = False, zero_init: bool = True) -> None:
        super().__init__()
        act = ACTS[activation]
        layers: list[nn.Module] = []
        d = N_FEATURES
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            if layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(act())
            d = hidden
        layers.append(nn.Linear(d, len(RESIDUAL_VARS)))
        self.body = nn.Sequential(*layers[:-1])
        self.head = layers[-1]
        # input → last-hidden shortcut (real projection, dims differ)
        self.short = (nn.Linear(N_FEATURES, hidden, bias=False)
                      if skip else None)
        if zero_init:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        scale = torch.tensor([BOUND_SCALE[v] for v in RESIDUAL_VARS],
                             dtype=torch.float64)
        self.register_buffer("scale", scale)
        self.to(torch.float64)

    @property
    def net(self) -> nn.Sequential:
        """Full Sequential view (kept for tests/tooling)."""
        return nn.Sequential(*self.body, self.head)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        h = self.body(feats)
        if self.short is not None:
            h = h + self.short(feats)
        return torch.tanh(self.head(h)) * self.scale
