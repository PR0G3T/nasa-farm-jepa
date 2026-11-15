"""Interventions — partial state edits and learned coupled responses.

`edit_state` writes user-supplied values on a masked subset of state
channels, respecting physical bounds. `ResponseNet` learns how an edit
propagates: trained on the model's own counterfactuals, it predicts the
one-step response of every *unmasked* channel — the masked-response scheme
that turns a forward simulator into a what-if engine without a rollout per
query.

`whatif` composes both: roll to day d, edit, optionally fold in the learned
response, roll on — an auditable do-operator on the world model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nasa_farm.constants import DTYPE
from nasa_farm.dynamics.residual import N_FEATURES
from nasa_farm.state.field import Field, FieldState
from nasa_farm.state.soil import SoilProfile
from nasa_farm.types import Actions, Forcing

EDIT_VARS: tuple[str, ...] = (
    "sw0",
    "swr",
    "n_min",
    "lai",
    "biomass",
    "litter",
    "soc_f",
    "soc_s",
    "tws",
    "gdd",
)
"""Channels a user may set directly (sw0 = surface layer, swr = root-zone
total). The response net predicts deltas on this same channel set for the
unmasked complement."""

EDIT_SCALE: dict[str, float] = {
    "sw0": 300.0,
    "swr": 600.0,
    "n_min": 80.0,
    "lai": 6.0,
    "biomass": 20_000.0,
    "litter": 5_000.0,
    "soc_f": 3_000.0,
    "soc_s": 15_000.0,
    "tws": 200.0,
    "gdd": 2_500.0,
}


def get_variables(state: FieldState) -> torch.Tensor:
    """(B, E) editable-channel view of the state."""
    return torch.stack(
        [
            state.sw[:, 0],
            state.sw[:, 1:].sum(-1),
            state.n_min,
            state.lai,
            state.biomass,
            state.litter,
            state.soc_f,
            state.soc_s,
            state.tws,
            state.gdd,
        ],
        dim=-1,
    )


def put_variables(
    state: FieldState, values: torch.Tensor, soil: SoilProfile | None = None
) -> FieldState:
    """(B, E) channel values → new state; sw splits layer0/root-zone.

    When `soil` is given, water values are clamped to hydraulic bounds and
    the root-zone total is redistributed proportional to current layer
    contents (uniform when dry)."""
    v = dict(zip(EDIT_VARS, values.unbind(-1), strict=True))
    sw0 = v["sw0"].clamp(min=0.0)
    swr = v["swr"].clamp(min=0.0)
    if soil is not None:
        sw0 = sw0.clamp(min=soil.air[:, 0], max=soil.sat[:, 0])
        cap_r = soil.sat[:, 1:].sum(-1)
        swr = swr.clamp(max=cap_r)
    if state.n_layers > 1:
        root = state.sw[:, 1:]
        w = root / root.sum(-1, keepdim=True).clamp(min=1e-9)
        w = torch.where(root.sum(-1, keepdim=True) > 1e-9, w, torch.full_like(w, 1.0 / w.shape[-1]))
        root = swr.unsqueeze(-1) * w
        if soil is not None:
            root = root.clamp(max=soil.sat[:, 1:])
            swr = root.sum(-1)
        sw = torch.cat([sw0.unsqueeze(1), root], dim=1)
    else:
        sw = sw0.unsqueeze(1)
    return state.replace(
        sw=sw,
        n_min=v["n_min"].clamp(min=0.0),
        lai=v["lai"].clamp(min=0.0),
        biomass=v["biomass"].clamp(min=0.0),
        litter=v["litter"].clamp(min=0.0),
        soc_f=v["soc_f"].clamp(min=0.0),
        soc_s=v["soc_s"].clamp(min=0.0),
        tws=v["tws"],
        gdd=v["gdd"].clamp(min=0.0),
    )


def edit_state(
    state: FieldState, mask: torch.Tensor, values: torch.Tensor, soil: SoilProfile | None = None
) -> FieldState:
    """Do-operator: set EDIT_VARS[mask] to `vals`, keep the rest.

    mask: (E,) bool or indices; vals: (B, E) or broadcastable, read only on
    masked positions. Complement channels pass through untouched — a bare
    edit carries no learned response; `ResponseNet` adds propagation."""
    current = get_variables(state)
    m = (
        torch.zeros(len(EDIT_VARS), dtype=torch.bool, device=current.device)
        if mask.dtype != torch.bool
        else mask.to(current.device)
    )
    if mask.dtype != torch.bool:
        m[mask.long()] = True
    v = values.to(DTYPE).expand_as(current)
    return put_variables(state, torch.where(m, v, current), soil)


class Response(nn.Module):
    """Learned coupled response to a partial edit.

    Input: state features + normalized edit values + edit mask.
    Output: per-channel additive deltas applied on the *complement* of the
    mask only — the edited channels keep exactly what the user set.
    Zero-init ⇒ no response at init, the honest prior."""

    def __init__(self, hidden: int = 64, depth: int = 2, context_dimension: int = 0) -> None:
        super().__init__()
        e = len(EDIT_VARS)
        self.context_dimension = context_dimension
        layers: list[nn.Module] = []
        d = N_FEATURES + 2 * e + context_dimension
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, e)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        scale = torch.tensor([EDIT_SCALE[v] for v in EDIT_VARS])
        self.register_buffer("scale", scale.to(DTYPE))
        self.to(DTYPE)

    def forward(
        self,
        features: torch.Tensor,
        edit: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = self.scale.to(edit.device)
        mb = mask.to(edit.dtype)
        if mb.dim() == 1:
            mb = mb.expand(features.shape[0], -1)
        x = [features, edit / scale, mb]
        if self.context_dimension:
            c = (
                context
                if context is not None
                else torch.zeros(
                    features.shape[0],
                    self.context_dimension,
                    dtype=features.dtype,
                    device=features.device,
                )
            )
            x.append(c.to(features.dtype))
        output = torch.tanh(self.net(torch.cat(x, dim=-1))) * scale
        return output * (1.0 - mb)


def apply_response(
    state: FieldState,
    response: Response,
    features: torch.Tensor,
    mask: torch.Tensor,
    soil: SoilProfile | None = None,
    context: torch.Tensor | None = None,
) -> FieldState:
    """edit_state + predicted propagation on the unmasked complement."""
    delta = response(features, get_variables(state), mask, context)
    return put_variables(state, get_variables(state) + delta, soil)


def whatif(
    model,
    field: Field,
    forcing: Forcing,
    actions: Actions,
    day: int,
    mask: torch.Tensor,
    values: torch.Tensor,
    response: Response | None = None,
    observer=None,
) -> Field:
    """Counterfactual query: roll to `day`, apply the edit (+ optional
    learned response), roll to the end. Returns the final field."""
    for t in range(day):
        field, fluxes = model.step(field, actions.day(t), forcing.day(t), t)
        if observer is not None:
            observer(t, field.state, fluxes)
    s = edit_state(field.state, mask, values, field.soil)
    if response is not None:
        forcing_day = forcing.day(min(day, forcing.days - 1))
        action = actions.day(min(day, actions.days - 1))
        features = model._features(s, forcing_day, action)
        s = apply_response(s, response, features, mask, field.soil, field.context)
    field = Field(
        state=s,
        soil=field.soil,
        latitude=field.latitude,
        elevation=field.elevation,
        context=field.context,
    )
    for t in range(day, forcing.days):
        field, fluxes = model.step(field, actions.day(t), forcing.day(t), t)
        if observer is not None:
            observer(t, field.state, fluxes)
    return field


def fit_response(
    model,
    response: Response,
    field: Field,
    forcing: Forcing,
    actions: Actions,
    days: tuple[int, ...] | None = None,
    edits_per_day: int = 8,
    epochs: int = 100,
    rate: float = 1e-2,
    seed: int = 0,
) -> dict:
    """Self-supervised masked-response training on model counterfactuals.

    For each sampled day and random (mask, values) edit, the target is the
    model's own one-step response: Δ = step(edit(s)) − step(s) on unmasked
    channels. The net distills the model's intervention semantics, so
    `whatif` answers match rolled-out counterfactuals.
    """
    generation = torch.Generator().manual_seed(seed)
    snapshot: dict[int, tuple] = {}
    candidate = sorted(
        set(days) if days is not None else range(5, forcing.days - 1, max(1, forcing.days // 24))
    )

    f = field
    with torch.inference_mode():
        for t in range(forcing.days - 1):
            f, _ = model.step(f, actions.day(t), forcing.day(t), t)
            if t in candidate:
                snapshot[t] = (f.state.clone(), forcing.day(t), actions.day(t))

    parameters = {f"p{i}": p for i, p in enumerate(response.parameters())}
    leaves = list(parameters.values())
    opt = torch.optim.Adam(leaves, lr=rate)
    e = len(EDIT_VARS)
    history = []
    for _ in range(epochs):
        total = torch.zeros((), dtype=DTYPE)
        for t, (saved, forcing_day, action) in snapshot.items():
            features = model._features(saved, forcing_day, action).detach()
            for _ in range(edits_per_day):
                m = torch.rand(e, generator=generation) < 0.4
                if not m.any() or m.all():
                    continue
                v = get_variables(saved) * (
                    1.0 + 0.5 * torch.randn((saved.batch, e), generator=generation, dtype=DTYPE)
                )
                with torch.inference_mode():
                    base_f = Field(
                        state=saved,
                        soil=field.soil,
                        latitude=field.latitude,
                        elevation=field.elevation,
                        context=field.context,
                    )
                    edit_f = Field(
                        state=edit_state(saved, m, v, field.soil),
                        soil=field.soil,
                        latitude=field.latitude,
                        elevation=field.elevation,
                        context=field.context,
                    )
                    target = get_variables(
                        model.step(edit_f, action, forcing_day, t)[0].state
                    ) - get_variables(model.step(base_f, action, forcing_day, t)[0].state)
                edited = torch.where(m, v, get_variables(saved))
                predicted = response(features, edited, m, field.context)
                total = (
                    total + (((predicted - target) * (~m).to(DTYPE)) / EDIT_SCALE_T).pow(2).mean()
                )
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(leaves, 1.0)
        opt.step()
        history.append(float(total.detach()))
    return {"history": history, "final": history[-1] if history else 0.0}


EDIT_SCALE_T = torch.tensor([EDIT_SCALE[v] for v in EDIT_VARS], dtype=DTYPE)
