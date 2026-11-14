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
from nasa_farm.forcing import climatology_ensemble, synthetic_year
from nasa_farm.plan import enumerate_rotations, evaluate, recommend
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
                    help="rollout threads; 0 = auto (no-GIL build only)")
    pl.add_argument("--compile", action="store_true",
                    help="torch.compile the step — pays off on large sweeps")
    pl.add_argument("--config", default=None,
                    help="TOML file; CLI flags override config values")
    pl.add_argument("--offline", action="store_true", help="synthetic forcing, no network")
    pl.add_argument("--no-residual", action="store_true")
    return p


def _argv_with_config(argv: list[str]) -> list[str]:
    """`--config file.toml` → flags injected after the subcommand, so any
    flag the user typed still wins (argparse takes the last occurrence)."""
    if "--config" not in argv or (argv and argv[0] != "plan"):
        return argv
    i = argv.index("--config")
    with open(argv[i + 1], "rb") as fh:
        cfg = tomllib.load(fh)
    argv = argv[:i] + argv[i + 2:]
    flags = []
    for k, v in cfg.get("plan", cfg).items():
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
    if args.compile:
        model.enable_compile(mode="max-autotune-no-cudagraphs")
    ev = evaluate(model, field, plans, ens,
                  irr_mm_day=args.irrig, n_fert_kg=args.fert,
                  workers=args.workers)
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


def main(argv: list[str] | None = None) -> int:
    argv = _argv_with_config(list(sys.argv[1:] if argv is None else argv))
    args = _build_parser().parse_args(argv)
    match args.cmd:
        case "plan":
            return cmd_plan(args)
        case _:
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
