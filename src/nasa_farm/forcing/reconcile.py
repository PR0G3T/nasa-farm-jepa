"""Multi-source forcing consensus — precision-weighted fusion.

POWER, NLDAS-3, reanalysis products: each carries its own bias, so the
consensus is a per-variable weighted mean across sources with the
weighted std returned as per-day forcing uncertainty — usable directly
as observation-equivalent spread downstream. `prec` maps source → weight
(inverse error variance); per-variable overrides cover sources known to
be better for, e.g., radiation than precipitation.
"""

from __future__ import annotations

import torch

from nasa_farm.constants import DTYPE
from nasa_farm.types import Forcing

W_VARS: tuple[str, ...] = ("tmin", "tmax", "srad", "precip", "rh", "u2")


def consensus(
    srcs: dict[str, Forcing],
    prec: dict[str, float] | None = None,
    var_prec: dict[str, dict[str, float]] | None = None,
) -> tuple[Forcing, dict[str, torch.Tensor]]:
    """{source: Forcing} → (consensus Forcing (1,T), {var: (T,) spread}).

    Each source contributes its member mean; weights normalize per
    variable so a source missing a weight drops out of that variable.
    Physical contract enforced after fusion (tmax ≥ tmin, bounds).
    """
    names = list(srcs)
    values, spread = {}, {}
    for v in W_VARS:
        w = torch.tensor(
            [
                (
                    (var_prec or {}).get(v, {}).get(s)
                    if var_prec and v in var_prec
                    else (prec or {}).get(s, 1.0)
                )
                for s in names
            ],
            dtype=DTYPE,
        )
        x = torch.stack([getattr(srcs[s], v).to(DTYPE).mean(0) for s in names])  # (S, T)
        wm = w / w.sum().clamp(min=1e-12)
        values[v] = (wm.unsqueeze(-1) * x).sum(0)
        spread[v] = ((wm.unsqueeze(-1) * (x - values[v].unsqueeze(0)) ** 2).sum(0)).sqrt()
    values["precip"] = values["precip"].clamp(min=0.0)
    values["srad"] = values["srad"].clamp(min=0.0)
    values["rh"] = values["rh"].clamp(0.0, 100.0)
    values["u2"] = values["u2"].clamp(min=0.0)
    tmin = torch.minimum(values["tmin"], values["tmax"])
    tmax = torch.maximum(values["tmin"], values["tmax"])
    doy = srcs[names[0]].doy[:1]
    f = Forcing(
        tmin=tmin.unsqueeze(0),
        tmax=tmax.unsqueeze(0),
        srad=values["srad"].unsqueeze(0),
        precip=values["precip"].unsqueeze(0),
        rh=values["rh"].unsqueeze(0),
        u2=values["u2"].unsqueeze(0),
        doy=doy,
    )
    return f, spread
