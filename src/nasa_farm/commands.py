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
from nasa_farm.constants import DTYPE
from nasa_farm.control import mppi
from nasa_farm.experiments import Spec, experiments
from nasa_farm.experiments import run as exp_run
from nasa_farm.forcing import augment, synthetic_year, tile_to_horizon
from nasa_farm.optimize.hyperparameter import gradient_hpo
from nasa_farm.optimize.architecture import nas_search
from nasa_farm.optimize.targets import (
    season_metrics, synthetic_observation, trajectory_loss,
)
from nasa_farm.optimize.trainer import fit, leaf
from nasa_farm.service import SCENARIO_KEYS, PlanRequest, Point, plan
from nasa_farm.state.crops import IDX
from nasa_farm.state.soil import SoilProfile

DEFAULT_CROPS_ARG = "maize,soybean,wheat,clover,fallow"


def _plan_flags(pl: argparse.ArgumentParser) -> None:
    """Shared compute flags for `plan` and `export` — one definition."""
    pl.add_argument("--horizon", type=int, default=3, help="rotation length, years")
    pl.add_argument("--crops", default=DEFAULT_CROPS_ARG)
    pl.add_argument("--years", type=int, default=5, help="climatology depth")
    pl.add_argument("--delta-t", type=float, default=0.0, help="+°C scenario shift")
    pl.add_argument("--precip-scale", type=float, default=1.0)
    pl.add_argument("--irrig", type=float, default=0.0, help="mm/day post-sowing")
    pl.add_argument("--fert", type=float, default=0.0, help="kg N/ha at sow+30d")
    pl.add_argument("--soil", default=None, help="texture preset override")
    pl.add_argument("--depths", default=None,
                    help="layer thicknesses m, comma-sep (e.g. 0.3,0.6,0.9)")
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
    pl.add_argument("--co2", type=float, default=420.0, help="ppm scenario")
    pl.add_argument("--offline", action="store_true", help="synthetic forcing, no network")
    pl.add_argument("--source", default="power",
                    choices=("power", "ens", "seas5", "synth"),
                    help="forcing ensemble source: power climatology, "
                         "operational IFS ENS / SEAS5 (Open-Meteo), synthetic")
    pl.add_argument("--members", type=int, default=16,
                    help="ensemble members for forecast sources (≤51)")
    pl.add_argument("--no-residual", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nasa-farm")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("plan", help="rank rotations for a field")
    _plan_flags(pl)
    pl.add_argument("--lat", type=float, required=True)
    pl.add_argument("--lon", type=float, required=True)

    ex2 = sub.add_parser("export", help="batch plan → static JSON for the app")
    _plan_flags(ex2)
    ex2.add_argument("--fields", required=True,
                     help="fields JSON path, or '-' for stdin")
    ex2.add_argument("--out", required=True, help="output JSON path")
    ex2.add_argument("--scenario", action="append", default=None,
                     help="name[:k=v,...] — repeatable; k ∈ "
                          + ",".join(sorted(SCENARIO_KEYS)))

    cr = sub.add_parser("crawl", help="batch acquisition → data lake")
    cr.add_argument("--fields", required=True,
                    help="fields JSON path, or '-' for stdin")
    cr.add_argument("--out", default="lake", help="lake root dir")
    cr.add_argument("--years", type=int, default=5)
    cr.add_argument("--sources", default="power,soil",
                    help="power,soil,smap,ecostress-seb,ecostress-met")
    cr.add_argument("--workers", type=int, default=8)

    sv = sub.add_parser("serve", help="HTTP API (granian + fastapi)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--workers", type=int, default=1,
                    help="granian worker processes")
    sv.add_argument("--tunnel", action="store_true",
                    help="cloudflared quick tunnel → public URL")

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

    ct = sub.add_parser("control", help="MPPI irrigation/N scheduling")
    ct.add_argument("--lat", type=float, default=42.0)
    ct.add_argument("--crop", default="maize")
    ct.add_argument("--scenarios", type=int, default=8,
                    help="forcing ensemble size")
    ct.add_argument("--samples", type=int, default=64)
    ct.add_argument("--iters", type=int, default=6)
    ct.add_argument("--weights", default=None,
                    help="reward mix, e.g. yield_kg=1.0,n_leached=-0.5")
    ct.add_argument("--risk", default="mean",
                    choices=("mean", "cvar", "conf"),
                    help="scenario risk aggregation")
    ct.add_argument("--alpha", type=float, default=0.2,
                    help="CVaR tail level / conformal miscoverage")
    ct.add_argument("--sow-shifts", default="0",
                    help="sow-day shifts searched, comma-sep (e.g. -14,0,14)")
    ct.add_argument("--seed", type=int, default=0)

    ex = sub.add_parser("exp", help="run a research experiment")
    ex.add_argument("name", choices=experiments())
    ex.add_argument("--lat", type=float, default=42.0)
    ex.add_argument("--lon", type=float, default=-93.6)
    ex.add_argument("--soil", default="silt_loam")
    ex.add_argument("--crop", default="maize")
    ex.add_argument("--horizon", type=int, default=1)
    ex.add_argument("--years", default="",
                    help="POWER years, comma-sep; empty = synthetic forcing")
    ex.add_argument("--realizations", type=int, default=1,
                    help="held-out truth realizations scored")
    ex.add_argument("--members", type=int, default=16,
                    help="forcing-ensemble / DA ensemble size")
    ex.add_argument("--obs-every", type=int, default=10,
                    help="days between assimilated observations")
    ex.add_argument("--noise", type=float, default=0.05,
                    help="relative observation noise")
    ex.add_argument("--perturb", default="",
                    help="truth trait shifts, e.g. rue:0.8,gdd_mat:0.95")
    ex.add_argument("--channels", default="yield_kg",
                    help="seasonal METRICS channels, comma-sep")
    ex.add_argument("--obs-channels", default="ndvi,sm1",
                    help="satellite channels assimilated, comma-sep")
    ex.add_argument("--alphas", default="0.1",
                    help="interval levels for PICP/interval score")
    ex.add_argument("--latent-dim", type=int, default=8,
                    help="JEPA latent size")
    ex.add_argument("--hier", default="",
                    help="latent strides, comma-sep; empty = flat JEPA")
    ex.add_argument("--ema", type=float, default=0.996,
                    help="JEPA target-encoder momentum")
    ex.add_argument("--sig-weight", type=float, default=1.0,
                    help="SIGReg regularization strength")
    ex.add_argument("--epochs", type=int, default=120)
    ex.add_argument("--lr", type=float, default=0.05)
    ex.add_argument("--seed", type=int, default=0)
    ex.add_argument("--out", default=None, help="save report JSON here")
    ex.add_argument("--latex", action="store_true",
                    help="also print booktabs rows of scalar results")
    ex.add_argument("--config", default=None,
                    help="TOML file; CLI flags override config values")
    return p


def _argv_with_config(argv: list[str]) -> list[str]:
    """`--config file.toml` → flags injected after the subcommand, so any
    flag the user typed still wins (argparse takes the last occurrence).
    Section = subcommand name (`[plan]`, `[export]`, `[calibrate]`, `[nas]`,
    `[control]`, `[exp]`); flat keys also work."""
    if "--config" not in argv or not argv or argv[0] not in (
            "plan", "export", "calibrate", "nas", "control", "exp"):
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


def _req(args, points: list[Point], **over) -> PlanRequest:
    kw = dict(horizon=args.horizon, crops=args.crops.split(","),
              years=args.years, delta_t=args.delta_t,
              precip_scale=args.precip_scale, irrig=args.irrig,
              fert=args.fert, co2=args.co2, soil=args.soil,
              source=getattr(args, "source", "power"),
              members=getattr(args, "members", 16),
              residual=not args.no_residual, offline=args.offline,
              top_k=args.top_k,
              depths=([float(x) for x in args.depths.split(",")]
                      if args.depths else None))
    kw.update(over)
    return PlanRequest(points=points, **kw)


def cmd_plan(args) -> int:
    res = plan(_req(args, [Point(lat=args.lat, lon=args.lon)]),
               workers=args.workers, device=args.device,
               compile=args.compile).results[0]
    if res.status != "ok":
        sys.stderr.write(f"{res.detail}\n")
        return 1
    json.dump(res.model_dump(
        include={"plans_evaluated", "scenarios", "pareto", "top"}),
        sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _scenarios(specs: list[str] | None) -> dict[str, dict[str, float]]:
    """`name:k=v,...` specs → ordered {name: overrides}; bare name → {}."""
    out: dict[str, dict[str, float]] = {"baseline": {}}
    for spec in specs or ():
        name, _, rest = spec.partition(":")
        kv = dict((k, float(v)) for k, _, v in
                  (p.partition("=") for p in rest.split(",") if p))
        bad = set(kv) - SCENARIO_KEYS
        if bad:
            raise SystemExit(f"scenario {name!r}: keys {sorted(bad)} "
                             f"∉ {sorted(SCENARIO_KEYS)}")
        out[name] = kv
    return out


def cmd_export(args) -> int:
    from pathlib import Path

    from nasa_farm import __version__

    src = (sys.stdin.read() if args.fields == "-"
           else Path(args.fields).read_text())
    raw = json.loads(src)
    items = raw if isinstance(raw, list) else raw.get(
        "fields", raw.get("points", []))
    points = [Point(id=str(f["id"]), lat=f["lat"], lon=f["lon"])
              for f in items]
    meta = {str(f["id"]): {k: f[k] for k in ("name", "crop", "ha") if k in f}
            for f in items}

    scenarios = _scenarios(args.scenario)
    out = {"version": __version__,
           "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
           "request": {"horizon": args.horizon, "crops": args.crops.split(","),
                       "years": args.years, "irrig": args.irrig,
                       "fert": args.fert, "co2": args.co2,
                       "offline": args.offline},
           "fields": meta, "scenarios": {}}
    for name, over in scenarios.items():
        res = plan(_req(args, points, **over),
                   workers=args.workers, device=args.device,
                   compile=args.compile)
        fields = {}
        for i, r in enumerate(res.results):
            if r.point.id:
                fields[r.point.id] = r.model_dump(exclude={"point", "history"})
                if r.history and "history" not in out["fields"].get(r.point.id, {}):
                    out["fields"].setdefault(r.point.id, {})["history"] = r.history
            sys.stderr.write(
                f"{name} [{i + 1}/{len(res.results)}] "
                f"{r.point.id or r.point.lat} {r.status}"
                f"{f' — {r.detail}' if r.detail else ''}\n")
        out["scenarios"][name] = {"params": over, "fields": fields}
    Path(args.out).write_text(json.dumps(out) + "\n")
    sys.stderr.write(f"wrote {args.out}\n")
    return 0


def cmd_crawl(args) -> int:
    from pathlib import Path

    from nasa_farm.acquire.crawl import crawl

    src = (sys.stdin.read() if args.fields == "-"
           else Path(args.fields).read_text())
    raw = json.loads(src)
    items = raw if isinstance(raw, list) else raw.get(
        "fields", raw.get("points", []))
    fields = [{"id": str(f["id"]), "lat": f["lat"], "lon": f["lon"],
               "name": f.get("name", "")} for f in items]
    last = dt.date.today().year - 1
    crawl(fields, args.out,
          list(range(last - args.years + 1, last + 1)),
          sources=tuple(args.sources.split(",")), workers=args.workers,
          log=lambda m: sys.stderr.write(f"{m}\n"))
    sys.stderr.write(f"wrote {args.out}\n")
    return 0


def cmd_serve(args) -> int:
    try:
        from granian import Granian
        from granian.constants import Interfaces
    except ImportError:
        sys.stderr.write("serve extra required: uv sync --extra serve\n")
        return 2
    from nasa_farm.server import tunnel
    proc = tunnel(args.port) if args.tunnel else None
    try:
        Granian("nasa_farm.server:app", address=args.host, port=args.port,
                interface=Interfaces.ASGI, workers=args.workers).serve()
    finally:
        if proc is not None:
            proc.terminate()
    return 0


def _demo_actions(crops, days: int, crop: str = "maize"):
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
    actions = _demo_actions(crops, forcing.days, args.crop)
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
    actions = _demo_actions(crops, forcing.days)

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


def cmd_control(args) -> int:
    """Scenario-robust MPPI: optimize daily irrigation/N against a forcing
    ensemble; report the schedule and its yield uplift vs rainfed."""
    from nasa_farm.heads import METRICS, SeasonAccumulator

    crops = CropLibrary.default()
    soil = SoilProfile.preset("silt_loam")
    field = Field(initial_state(1, soil), soil,
                  torch.tensor([args.lat], dtype=DTYPE))
    ens = augment(synthetic_year(args.lat, seed=args.seed),
                  args.scenarios - 1, seed=args.seed)
    base = _demo_actions(crops, ens.days, args.crop)
    model = WorldModel(crops)
    weights = (None if args.weights is None else
               {k: float(v) for k, v in
                (w.split("=") for w in args.weights.split(","))})
    res = mppi(model, field, ens, base, iters=args.iters,
               samples=args.samples, weights=weights, risk=args.risk,
               alpha=args.alpha,
               sow_shifts=tuple(int(x) for x in args.sow_shifts.split(",")),
               seed=args.seed)

    idx = {n: i for i, n in enumerate(METRICS)}

    def _yield(acts) -> float:
        acc = SeasonAccumulator(ens.batch, ens.days // 365)
        with torch.inference_mode():
            model.rollout(field.repeat(ens.batch), ens,
                          acts.repeat(ens.batch), observer=acc)
        return float(acc.result()[..., idx["yield_kg"]].mean())

    y_opt, y_rf = _yield(res.actions), _yield(base)
    out = {
        "crop": args.crop, "scenarios": ens.batch,
        "reward": round(res.reward, 2),
        "reward_history": [round(h, 2) for h in res.history],
        "irrig_total_mm": round(float(res.actions.irrigation.sum()), 1),
        "fert_total_kg": round(float(res.actions.n_fert.sum()), 1),
        "yield_kg_mean": round(y_opt, 0),
        "yield_vs_rainfed_pct": round(100 * (y_opt - y_rf)
                                      / max(abs(y_rf), 1e-9), 1),
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_exp(args) -> int:
    perturb = {k: float(v) for k, v in
               (p.split(":") for p in args.perturb.split(",") if p)}
    spec = Spec(
        lat=args.lat, lon=args.lon, soil=args.soil, crop=args.crop,
        horizon=args.horizon,
        years=tuple(int(y) for y in args.years.split(",") if y),
        realizations=args.realizations,
        members=args.members, obs_every=args.obs_every,
        noise=args.noise, perturb=perturb,
        obs_channels=tuple(args.obs_channels.split(",")),
        channels=tuple(args.channels.split(",")),
        alphas=tuple(float(a) for a in args.alphas.split(",")),
        latent_dim=args.latent_dim,
        hier=tuple(int(s) for s in args.hier.split(",") if s),
        ema=args.ema,
        sig_weight=args.sig_weight,
        epochs=args.epochs, lr=args.lr, seed=args.seed)
    rep = exp_run(args.name, spec)
    out = {"experiment": rep.name, "seconds": rep.seconds,
           "spec": rep.spec, "meta": rep.meta, "results": rep.results}
    if args.out:
        out["saved"] = str(rep.save(args.out))
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if args.latex:
        sys.stdout.write(rep.latex() + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = _argv_with_config(list(sys.argv[1:] if argv is None else argv))
    args = _build_parser().parse_args(argv)
    match args.cmd:
        case "plan":
            return cmd_plan(args)
        case "export":
            return cmd_export(args)
        case "crawl":
            return cmd_crawl(args)
        case "serve":
            return cmd_serve(args)
        case "calibrate":
            return cmd_calibrate(args)
        case "nas":
            return cmd_nas(args)
        case "control":
            return cmd_control(args)
        case "exp":
            return cmd_exp(args)
        case _:
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
