"""Ensemble copula coupling — restore dependence to postprocessed members.

Marginal postprocessing (QMap, EnsBC) fixes each variable at each day but
destroys the joint structure: a member's day-to-day autocorrelation and
cross-variable dependence come out white. ECC reorders each column's
marginal values along the rank order of a template ensemble — observed
years or the raw forecast — transferring the template's full dependence
pattern while preserving the corrected marginals exactly (each column
keeps its multiset of values, hence its univariate distribution).

Schaake shuffle = ECC with a historical-observation template.
"""

from __future__ import annotations

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.types import FORCING_VARS, Forcing


def ranks(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Rank 0..N−1 of each element along `dim` (ties break by position —
    deterministic, which is all the reordering needs)."""
    return x.argsort(dim).argsort(dim).to(DTYPE)


def ecc(fcst: torch.Tensor, tmpl: torch.Tensor) -> torch.Tensor:
    """(S, T) forecast × (R, T) template → reordered forecast (S, T).

    Forecast member m pairs with template member τ(m); at every column it
    receives the order statistic whose normalized rank equals the paired
    template member's rank there. S=R pairs 1:1 and preserves each
    column's multiset exactly; S≠R interpolates the sorted marginal.
    """
    s, r = fcst.shape[0], tmpl.shape[0]
    edited = fcst.sort(0).values                                    # (S, T)
    sel = (torch.arange(s, device=fcst.device) * (r - 1)
           / max(s - 1, 1)).round().long()                      # (S,) τ(m)
    p = ranks(tmpl, 0)[sel] * (s - 1) / max(r - 1, 1)           # (S, T)
    low = p.floor().clamp(max=s - 1).long()
    high = p.ceil().long()
    fraction = p - low.to(DTYPE)
    return (edited.gather(0, low)
            + (edited.gather(0, high) - edited.gather(0, low)) * fraction)


def shuffle_forcing(fcst: Forcing, tmpl: Forcing) -> Forcing:
    """ECC per weather variable; doy/co2 pass through unshuffled."""
    keywords = {v: (ecc(getattr(fcst, v), getattr(tmpl, v)) if fcst.batch > 1
              else getattr(fcst, v))
          for v in FORCING_VARS if v not in ("doy", "co2")}
    return Forcing(**keywords, doy=fcst.doy, co2=fcst.co2)
