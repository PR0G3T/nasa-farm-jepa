"""nasa-farm CLI — fetch → build field → enumerate rotations → Pareto.

Single command = full pipeline: POWER climatology ensemble × rotation
enumeration → exact Pareto front + priority-weighted ranking.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tomllib

import torch

from nasa_farm import CropLibrary, Field, WorldModel, initial_state
from nasa_farm.acquire.power import PowerClient
from nasa_farm.acquire.soilgrids import SoilGridsClient
from nasa_farm.constants import DTYPE
from nasa_farm.forcing import climatology_ensemble, synthetic_year, tile_to_horizon
from nasa_farm.optimize.hpo import gradient_hpo
from nasa_farm.optimize.nas import nas_search
from nasa_farm.optimize.targets import (
    season_metrics, synthetic_observation, trajectory_loss,
)
from nasa_farm.optimize.trainer import fit, leaf
from nasa_farm.plan import enumerate_rotations, evaluate, recommend
from nasa_farm.state.crops import IDX
from nasa_farm.state.soil import SoilProfile


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nasa-farm")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("plan", help="rank rotations for a field")
    pl.add_argument("--lat", type=float, required=True)
    pl.add_argument("--lon", type=float, required=True)
    pl.add_argument("--horizon", type=int, default=3, help="rotation length, years")
    pl.add_argument("--crops", default="maize,soybean,wheat,clover,fallow")
    pl.add_argument("--years", type=int, default=5, help="climatology depth")
    pl.add_argument("--delta-t", type=float, default=0.0, help="+°C scenario shift")
    pl.add_argument("--precip-scale", type=float, default=1.0)
    pl.add_argument("--irrig", type=float, default=0.0, help="mm/day post-sowing")
    pl.add_argument("--fert", type=float, default=0.0, help="kg N/ha at sow+30d")
    pl.add_argument("--soil", default=None, help="texture preset override")
    pl.add_argument("--top-k", type=int, default=5)
    pl.add_argument("--workers", type=int, default=0,
                    help="rollout threads; 0 = auto (physical cores, no-GIL)")
    pl.add_argument("--device", default="auto",
                    help="auto | cpu | cuda — auto picks the GPU when usable")
    pl.add_argument("--compile", nargs="?", const="on", default="auto",
                    choices=("auto", "on", "off"),
                    help="torch.compile the step; auto engages on large sweeps")
    pl.add_argument("--config", default=None,
                    help="TOML file; CLI flags override config values")
    pl.add_argument("--offline", action="store_true", help="synthetic forcing, no network")
    pl.add_argument("--no-residual", action="store_true")

    cal = sub.add_parser("calibrate", help="gradient HPO on a trait cell")
    cal.add_argument("--lat", type=float, required=True)
    cal.add_argument("--lon", type=float, required=True)
    cal.add_argument("--crop", default="maize")
    cal.add_argument("--param", default="rue", help="trait name to perturb+recover")
    cal.add_argument("--scale", type=float, default=0.8,
                   help="synthetic truth = trait × scale")
    cal.add_argument("--channel", default="yield_kg", help="METRICS channel")
    cal.add_argument("--epochs", type=int, default=120)
    cal.add_argument("--lr", type=float, default=0.05)

    nas = sub.add_parser("nas", help="residual architecture search")
    nas.add_argument("--lat", type=float, default=42.0)
    nas.add_argument("--lon", type=float, default=-93.6)
    nas.add_argument("--candidates", type=int, default=24)
    nas.add_argument("--top-k", type=int, default=8)
    nas.add_argument("--rungs", default="10,30,90", help="fit budgets, comma-sep")
    nas.add_argument("--seed", type=int, default=0)
    return p


def _argv_with_config(argv: list[str]) -> list[str]:
    """`--config file.toml` → flags injected after the subcommand, so any
    flag the user typed still wins (argparse takes the last occurrence).
    Section = subcommand name (`[plan]`, `[calibrate]`, `[nas]`); flat keys
    also work."""
    if "--config" not in argv or not argv or argv[0] not in (
            "plan", "calibrate", "nas"):
        return argv
    i = argv.index("--config")
    with open(argv[i + 1], "rb") as fh:
        cfg = tomllib.load(fh)
    argv = argv[:i] + argv[i + 2:]
    flags = []
    for k, v in cfg.get(argv[0], cfg).items():
        key = "--" + k.replace("_", "-")
        match v:
            case bool() if v:  flags.append(key)
            case bool():       pass
            case _:            flags += [key, str(v)]
    return [argv[0], *flags, *argv[1:]]


def _forcings(args) -> list:
    if args.offline:
        return [synthetic_year(args.lat, seed=y) for y in range(args.years)]
    last = dt.date.today().year - 1
    return PowerClient().forcing_years(
        args.lat, args.lon, list(range(last - args.years + 1, last + 1)))


def cmd_plan(args) -> int:
    crops = CropLibrary.default()
    crop_idx = [crops.index(c) for c in args.crops.split(",")]
    plans = enumerate_rotations(crop_idx, args.horizon)

    ens = climatology_ensemble(_forcings(args), args.horizon,
                               delta_t=args.delta_t,
                               precip_scale=args.precip_scale)

    if args.soil:
        soil = SoilProfile.preset(args.soil)
    elif args.offline:
        soil = SoilProfile.preset("loam")
    else:
        soil = SoilGridsClient().profile(args.lat, args.lon)

    field = Field(state=initial_state(1, soil), soil=soil,
                  lat=torch.tensor([args.lat], dtype=DTYPE))

    model = WorldModel(crops, residual=not args.no_residual)
    ev = evaluate(model, field, plans, ens,
                  irr_mm_day=args.irrig, n_fert_kg=args.fert,
                  workers=args.workers, device=args.device,
                  compile=args.compile)
    rec = recommend(ev, top_k=min(args.top_k, plans.shape[0]))

    out = {
        "plans_evaluated": int(plans.shape[0]),
        "scenarios": int(ens.batch),
        "pareto": [
            {"rotation": [crops.names[c] for c in plans[i].tolist()],
             "rank": int((rec["ranking"] == i).nonzero().flatten()[0])
             if (rec["ranking"] == i).any() else None,
             **{k: round(float(v[i]), 2)
                for k, v in rec["objectives"].items()}}
            for i in rec["pareto_idx"].tolist()
        ],
        "top": [
            {"rotation": [crops.names[c] for c in plans[i].tolist()],
             "score": round(float(rec["scores"][i]), 4),
             **{k: round(float(v[i]), 2)
                for k, v in rec["objectives"].items()}}
            for i in rec["ranking"].tolist()
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _maize_actions(crops, days: int, crop: str = "maize"):
    """One sow event at the crop's default DOY — synthetic demo driver."""
    from nasa_farm.types import Actions
    a = Actions.idle(1, days)
    sow = a.sow.clone()
    sow[0, int(crops.get(crop)[IDX["sow_doy"]]) - 1] = crops.index(crop)
    return Actions(sow, a.irrigation, a.n_fert)


def cmd_calibrate(args) -> int:
    """Identifiability demo of the exact pipeline real observations will use:
    truth = trait×scale → obs → gradient-fit the cell → recovery %."""
    crops = CropLibrary.default()
    i_c, i_p = crops.index(args.crop), IDX[args.param]

    truth = WorldModel(crops, residual=False)
    tbl_t = truth.crop_table.detach().clone()
    tbl_t[i_c, i_p] *= args.scale
    truth.crop_table = tbl_t

    soil = SoilProfile.preset("silt_loam")
    field = Field(initial_state(1, soil), soil,
                  torch.tensor([args.lat], dtype=DTYPE))
    forcing = tile_to_horizon(synthetic_year(args.lat, seed=0), 1)
    actions = _maize_actions(crops, forcing.days, args.crop)
    obs = synthetic_observation(truth, field, forcing, actions, (args.channel,))

    cand = WorldModel(crops, residual=False)
    base = cand.crop_table.detach().clone()
    p = leaf(base[i_c, i_p])

    def loss():
        tbl = base.clone()
        tbl[i_c, i_p] = p
        cand.crop_table = tbl
        return trajectory_loss(
            season_metrics(cand, field, forcing, actions), obs)

    res = gradient_hpo(loss, {"v": p}, epochs=args.epochs, lr=args.lr)
    truth_v, start_v, fit_v = (float(tbl_t[i_c, i_p]),
                               float(base[i_c, i_p]), float(p.detach()))
    span = truth_v - start_v
    out = {
        "param": args.param, "crop": args.crop, "channel": args.channel,
        "start": start_v, "truth": truth_v, "fitted": fit_v,
        "recovery_pct": round(100 * (fit_v - start_v) / span, 1)
        if abs(span) > 1e-12 else 100.0,
        "best_loss": round(res.best_loss, 6), "epochs_run": res.epochs_run,
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_nas(args) -> int:
    """NAS on the residual head, on a learnable synthetic signal: reference
    model carries a small non-zero residual — what real obs will replace."""
    crops = CropLibrary.default()
    soil = SoilProfile.preset("silt_loam")
    field = Field(initial_state(1, soil), soil,
                  torch.tensor([args.lat], dtype=DTYPE))
    forcing = tile_to_horizon(synthetic_year(args.lat, seed=0), 1)
    actions = _maize_actions(crops, forcing.days)

    ref = WorldModel(crops, residual=True)
    with torch.no_grad():
        ref.residual.head.weight.normal_(0, 0.1,
                                         generator=torch.Generator().manual_seed(123))
    obs = synthetic_observation(ref, field, forcing, actions,
                                ("yield_kg", "biomass_end"))

    def objective(net, budget):
        m = WorldModel(crops, residual=True)
        m.residual = net
        params = {f"p{i}": p for i, p in enumerate(net.parameters())}
        r = fit(lambda: trajectory_loss(
            season_metrics(m, field, forcing, actions), obs),
            params, epochs=int(budget), lr=0.02, patience=int(budget))
        return r.best_loss

    res = nas_search(objective, n_candidates=args.candidates,
                     proxy_top_k=args.top_k,
                     rungs=tuple(int(x) for x in args.rungs.split(",")),
                     seed=args.seed)
    out = {
        "best_config": res.best_config,
        "best_loss": round(res.best_value, 5),
        "leaderboard": [
            {"cfg": c, "zero_cost": round(s, 3)} for c, s in res.leaderboard],
        "asha_rungs": [t.rung for t in res.trials],
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = _argv_with_config(list(sys.argv[1:] if argv is None else argv))
    args = _build_parser().parse_args(argv)
    match args.cmd:
        case "plan":
            return cmd_plan(args)
        case "calibrate":
            return cmd_calibrate(args)
        case "nas":
            return cmd_nas(args)
        case _:
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
