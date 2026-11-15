"""MRV — monitoring, reporting and verification of climate outcomes.

Converts modeled stock changes and emissions into CO2-equivalent with
conformal-consistent uncertainty bounds, the reporting unit every carbon
program (Verra VT0014 and successors) requires: conservative crediting
means reporting the *lower* bound of a one-sided interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nasa_farm.constants import DTYPE

GWP_N2O: float = 273.0  # AR6 100-yr GWP of N2O
N_TO_N2O: float = 44.0 / 28.0
C_TO_CO2: float = 44.0 / 12.0


@dataclass(frozen=True, slots=True)
class CarbonAccount:
    """t CO2e ha-1 for one comparison (baseline vs practice change)."""

    soc_co2e: torch.Tensor  # (B,) soil C change → CO2e removed
    n2o_co2e: torch.Tensor  # (B,) N2O emitted → CO2e released
    net_co2e: torch.Tensor  # soc_co2e - n2o_co2e (positive = removal)


def account(
    soc0: torch.Tensor,
    soc1: torch.Tensor,
    n2o0: torch.Tensor,
    n2o1: torch.Tensor,
    area_ha: float | torch.Tensor = 1.0,
) -> CarbonAccount:
    """soc_i: kg C m-2 (soc_f + soc_s) at end of period; n2o_i: kg N ha-1
    cumulative. Positive ΔSOC counts as removal; ΔN2O as emission."""
    dsoc = (soc1 - soc0) * 10.0 * C_TO_CO2  # kg C m-2 → t CO2 ha-1
    dn2o = (n2o1 - n2o0) * N_TO_N2O / 1000.0 * GWP_N2O  # kg N → t CO2e ha-1
    a = torch.as_tensor(area_ha, dtype=DTYPE)
    soc_e, n2o_e = dsoc * a, dn2o * a
    return CarbonAccount(soc_co2e=soc_e, n2o_co2e=n2o_e, net_co2e=soc_e - n2o_e)


def creditable(net: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Conservative credit: the one-sided lower confidence bound of the net
    CO2e distribution (scenario ensemble or posterior samples), dim 0 =
    sample axis."""
    return torch.quantile(net, alpha, dim=0)
