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
from nasa_farm.control import path_integral
from nasa_farm.experiments import Specification, experiments
from nasa_farm.experiments import run as exp_run
from nasa_farm.forcing import augment, synthetic_year, tile_to_horizon
from nasa_farm.optimize.hyperparameter import gradient
from nasa_farm.optimize.architecture import search
from nasa_farm.optimize.targets import (
    season_metrics, synthetic_observation, trajectory_loss,
)
from nasa_farm.optimize.trainer import fit, leaf
from nasa_farm.service import SCENARIO_KEYS, PlanRequest, Point, plan
from nasa_farm.state.crops import INDEX
from nasa_farm.state.soil import SoilProfile

DEFAULT_CROPS_ARG = "maize,soybean,wheat,clover,fallow"


def _plan_flags(parser: argparse.ArgumentParser) -> None:
    """Shared compute flags for `plan` and `export` — one definition."""
    parser.add_argument("--horizon", type=int, default=3, help="rotation length, years")
    parser.add_argument("--crops", default=DEFAULT_CROPS_ARG)
    parser.add_argument("--years", type=int, default=5, help="climatology depth")
    parser.add_argument("--delta-t", type=float, default=0.0, help="+°C scenario shift")
    parser.add_argument("--precip-scale", type=float, default=1.0)
    parser.add_argument("--irrig", type=float, default=0.0, help="mm/day post-sowing")
    parser.add_argument("--fert", type=float, default=0.0, help="kg N/ha at sow+30d")
    parser.add_argument("--soil", default=None, help="texture preset override")
    parser.add_argument("--depths", default=None,
                    help="layer thicknesses m, comma-sep (e.g. 0.3,0.6,0.9)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0,
                    help="rollout threads; 0 = auto (physical cores, no-GIL)")
    parser.add_argument("--device", default="auto",
                    help="auto | cpu | cuda — auto picks the GPU when usable")
    parser.add_argument("--compile", nargs="?", const="on", default="auto",
                    choices=("auto", "on", "off"),
                    help="torch.compile the step; auto engages on large sweeps")
    parser.add_argument("--config", default=None,
                    help="TOML file; CLI flags override config values")
    parser.add_argument("--co2", type=float, default=420.0, help="ppm scenario")
    parser.add_argument("--offline", action="store_true", help="synthetic forcing, no network")
    parser.add_argument("--source", default="power",
                    choices=("power", "ens", "seas5", "synth"),
                    help="forcing ensemble source: power climatology, "
                         "operational IFS ENS / SEAS5 (Open-Meteo), synthetic")
    parser.add_argument("--members", type=int, default=16,
                    help="ensemble members for forecast sources (≤51)")
    parser.add_argument("--no-residual", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nasa-farm")
    sub = p.add_subparsers(dest="cmd", required=True)

    parser = sub.add_parser("plan", help="rank rotations for a field")
    _plan_flags(parser)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)

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
    ex.add_argument("--obs-channels", dest="observed_channels", default="ndvi,sm1",
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


def _arguments_with_config(argv: list[str]) -> list[str]:
    """`--config file.toml` → flags injected after the subcommand, so any
    flag the user typed still wins (argparse takes the last occurrence).
    Section = subcommand name (`[plan]`, `[export]`, `[calibrate]`, `[nas]`,
    `[control]`, `[exp]`); flat keys also work."""
    if "--config" not in argv or not argv or argv[0] not in (
            "plan", "export", "calibrate", "nas", "control", "exp"):
        return argv
    i = argv.index("--config")
    with open(argv[i + 1], "rb") as fh:
        config = tomllib.load(fh)
    argv = argv[:i] + argv[i + 2:]
    flags = []
    for k, v in config.get(argv[0], config).items():
        key = "--" + k.replace("_", "-")
        match v:
            case bool() if v:  flags.append(key)
            case bool():       pass
            case _:            flags += [key, str(v)]
    return [argv[0], *flags, *argv[1:]]


def _request(arguments, points: list[Point], **over) -> PlanRequest:
    keywords = dict(horizon=arguments.horizon, crops=arguments.crops.split(","),
              years=arguments.years, delta_t=arguments.delta_t,
              precip_scale=arguments.precip_scale, irrig=arguments.irrig,
              fertilizer=arguments.fertilizer, co2=arguments.co2, soil=arguments.soil,
              source=getattr(arguments, "source", "power"),
              members=getattr(arguments, "members", 16),
              residual=not arguments.no_residual, offline=arguments.offline,
              top_k=arguments.top_k,
              depths=([float(x) for x in arguments.depths.split(",")]
                      if arguments.depths else None))
    keywords.update(over)
    return PlanRequest(points=points, **keywords)


def cmd_plan(arguments) -> int:
    result = plan(_request(arguments, [Point(latitude=arguments.latitude, longitude=arguments.longitude)]),
               workers=arguments.workers, device=arguments.device,
               compile=arguments.compile).results[0]
    if result.status != "ok":
        sys.stderr.write(f"{result.detail}\n")
        return 1
    json.dump(result.model_dump(
        include={"plans_evaluated", "scenarios", "pareto", "top"}),
        sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _scenarios(specs: list[str] | None) -> dict[str, dict[str, float]]:
    """`name:k=v,...` specs → ordered {name: overrides}; bare name → {}."""
    output: dict[str, dict[str, float]] = {"baseline": {}}
    for specification in specs or ():
        name, _, rest = specification.partition(":")
        kv = dict((k, float(v)) for k, _, v in
                  (p.partition("=") for p in rest.split(",") if p))
        bad = set(kv) - SCENARIO_KEYS
        if bad:
            raise SystemExit(f"scenario {name!r}: keys {sorted(bad)} "
                             f"∉ {sorted(SCENARIO_KEYS)}")
        output[name] = kv
    return output


def cmd_export(arguments) -> int:
    from pathlib import Path

    from nasa_farm import __version__

    source = (sys.stdin.read() if arguments.fields == "-"
           else Path(arguments.fields).read_text())
    raw = json.loads(source)
    items = raw if isinstance(raw, list) else raw.get(
        "fields", raw.get("points", []))
    points = [Point(id=str(f["id"]), latitude=f["lat"], longitude=f["lon"])
              for f in items]
    metadata = {str(f["id"]): {k: f[k] for k in ("name", "crop", "ha") if k in f}
            for f in items}

    scenarios = _scenarios(arguments.scenario)
    output = {"version": __version__,
           "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
           "request": {"horizon": arguments.horizon, "crops": arguments.crops.split(","),
                       "years": arguments.years, "irrig": arguments.irrig,
                       "fert": arguments.fertilizer, "co2": arguments.co2,
                       "offline": arguments.offline},
           "fields": metadata, "scenarios": {}}
    for name, over in scenarios.items():
        result = plan(_request(arguments, points, **over),
                   workers=arguments.workers, device=arguments.device,
                   compile=arguments.compile)
        fields = {}
        for i, r in enumerate(result.results):
            if r.point.id:
                fields[r.point.id] = r.model_dump(exclude={"point", "history"})
                if r.history and "history" not in output["fields"].get(r.point.id, {}):
                    output["fields"].setdefault(r.point.id, {})["history"] = r.history
            sys.stderr.write(
                f"{name} [{i + 1}/{len(result.results)}] "
                f"{r.point.id or r.point.latitude} {r.status}"
                f"{f' — {r.detail}' if r.detail else ''}\n")
        output["scenarios"][name] = {"params": over, "fields": fields}
    Path(arguments.output).write_text(json.dumps(output) + "\n")
    sys.stderr.write(f"wrote {arguments.output}\n")
    return 0


def cmd_crawl(arguments) -> int:
    from pathlib import Path

    from nasa_farm.acquire.crawl import crawl

    source = (sys.stdin.read() if arguments.fields == "-"
           else Path(arguments.fields).read_text())
    raw = json.loads(source)
    items = raw if isinstance(raw, list) else raw.get(
        "fields", raw.get("points", []))
    fields = [{"id": str(f["id"]), "lat": f["lat"], "lon": f["lon"],
               "name": f.get("name", "")} for f in items]
    last = dt.date.today().year - 1
    crawl(fields, arguments.output,
          list(range(last - arguments.years + 1, last + 1)),
          sources=tuple(arguments.sources.split(",")), workers=arguments.workers,
          log=lambda m: sys.stderr.write(f"{m}\n"))
    sys.stderr.write(f"wrote {arguments.output}\n")
    return 0


def cmd_serve(arguments) -> int:
    try:
        from granian import Granian
        from granian.constants import Interfaces
    except ImportError:
        sys.stderr.write("serve extra required: uv sync --extra serve\n")
        return 2
    from nasa_farm.server import tunnel
    process = tunnel(arguments.port) if arguments.tunnel else None
    try:
        Granian("nasa_farm.server:app", address=arguments.host, port=arguments.port,
                interface=Interfaces.ASGI, workers=arguments.workers).serve()
    finally:
        if process is not None:
            process.terminate()
    return 0


def _demo_actions(crops, days: int, crop: str = "maize"):
    """One sow event at the crop's default DOY — synthetic demo driver."""
    from nasa_farm.types import Actions
    a = Actions.idle(1, days)
    sow = a.sow.clone()
    sow[0, int(crops.get(crop)[INDEX["sow_doy"]]) - 1] = crops.index(crop)
    return Actions(sow, a.irrigation, a.n_fert)


def cmd_calibrate(arguments) -> int:
    """Identifiability demo of the exact pipeline real observations will use:
    truth = trait×scale → obs → gradient-fit the cell → recovery %."""
    crops = CropLibrary.default()
    i_c, i_p = crops.index(arguments.crop), INDEX[arguments.param]

    truth = WorldModel(crops, residual=False)
    tbl_t = truth.crop_table.detach().clone()
    tbl_t[i_c, i_p] *= arguments.scale
    truth.crop_table = tbl_t

    soil = SoilProfile.preset("silt_loam")
    field = Field(initial_state(1, soil), soil,
                  torch.tensor([arguments.latitude], dtype=DTYPE))
    forcing = tile_to_horizon(synthetic_year(arguments.latitude, seed=0), 1)
    actions = _demo_actions(crops, forcing.days, arguments.crop)
    observed = synthetic_observation(truth, field, forcing, actions, (arguments.channel,))

    candidate = WorldModel(crops, residual=False)
    base = candidate.crop_table.detach().clone()
    p = leaf(base[i_c, i_p])

    def loss():
        table = base.clone()
        table[i_c, i_p] = p
        candidate.crop_table = table
        return trajectory_loss(
            season_metrics(candidate, field, forcing, actions), observed)

    result = gradient(loss, {"v": p}, epochs=arguments.epochs, rate=arguments.rate)
    truth_v, start_v, fit_v = (float(tbl_t[i_c, i_p]),
                               float(base[i_c, i_p]), float(p.detach()))
    span = truth_v - start_v
    output = {
        "param": arguments.param, "crop": arguments.crop, "channel": arguments.channel,
        "start": start_v, "truth": truth_v, "fitted": fit_v,
        "recovery_pct": round(100 * (fit_v - start_v) / span, 1)
        if abs(span) > 1e-12 else 100.0,
        "best_loss": round(result.best_loss, 6), "epochs_run": result.epochs_run,
    }
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_nas(arguments) -> int:
    """NAS on the residual head, on a learnable synthetic signal: reference
    model carries a small non-zero residual — what real obs will replace."""
    crops = CropLibrary.default()
    soil = SoilProfile.preset("silt_loam")
    field = Field(initial_state(1, soil), soil,
                  torch.tensor([arguments.latitude], dtype=DTYPE))
    forcing = tile_to_horizon(synthetic_year(arguments.latitude, seed=0), 1)
    actions = _demo_actions(crops, forcing.days)

    reference = WorldModel(crops, residual=True)
    with torch.no_grad():
        reference.residual.head.weight.normal_(0, 0.1,
                                         generator=torch.Generator().manual_seed(123))
    observed = synthetic_observation(reference, field, forcing, actions,
                                ("yield_kg", "biomass_end"))

    def objective(net, budget):
        m = WorldModel(crops, residual=True)
        m.residual = net
        parameters = {f"p{i}": p for i, p in enumerate(net.parameters())}
        r = fit(lambda: trajectory_loss(
            season_metrics(m, field, forcing, actions), observed),
            parameters, epochs=int(budget), rate=0.02, patience=int(budget))
        return r.best_loss

    result = search(objective, n_candidates=arguments.candidates,
                     proxy_top_k=arguments.top_k,
                     rungs=tuple(int(x) for x in arguments.rungs.split(",")),
                     seed=arguments.seed)
    output = {
        "best_config": result.best_config,
        "best_loss": round(result.best_value, 5),
        "leaderboard": [
            {"cfg": c, "zero_cost": round(s, 3)} for c, s in result.leaderboard],
        "asha_rungs": [t.rung for t in result.trials],
    }
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_control(arguments) -> int:
    """Scenario-robust MPPI: optimize daily irrigation/N against a forcing
    ensemble; report the schedule and its yield uplift vs rainfed."""
    from nasa_farm.heads import METRICS, SeasonAccumulator

    crops = CropLibrary.default()
    soil = SoilProfile.preset("silt_loam")
    field = Field(initial_state(1, soil), soil,
                  torch.tensor([arguments.latitude], dtype=DTYPE))
    ensemble = augment(synthetic_year(arguments.latitude, seed=arguments.seed),
                  arguments.scenarios - 1, seed=arguments.seed)
    base = _demo_actions(crops, ensemble.days, arguments.crop)
    model = WorldModel(crops)
    weights = (None if arguments.weights is None else
               {k: float(v) for k, v in
                (w.split("=") for w in arguments.weights.split(","))})
    result = path_integral(model, field, ensemble, base, iters=arguments.iters,
               samples=arguments.samples, weights=weights, risk=arguments.risk,
               alpha=arguments.alpha,
               sow_shifts=tuple(int(x) for x in arguments.sow_shifts.split(",")),
               seed=arguments.seed)

    index = {n: i for i, n in enumerate(METRICS)}

    def _yield(acts) -> float:
        accumulate = SeasonAccumulator(ensemble.batch, ensemble.days // 365)
        with torch.inference_mode():
            model.rollout(field.repeat(ensemble.batch), ensemble,
                          acts.repeat(ensemble.batch), observer=accumulate)
        return float(accumulate.result()[..., index["yield_kg"]].mean())

    y_opt, y_rf = _yield(result.actions), _yield(base)
    output = {
        "crop": arguments.crop, "scenarios": ensemble.batch,
        "reward": round(result.reward, 2),
        "reward_history": [round(h, 2) for h in result.history],
        "irrig_total_mm": round(float(result.actions.irrigation.sum()), 1),
        "fert_total_kg": round(float(result.actions.n_fert.sum()), 1),
        "yield_kg_mean": round(y_opt, 0),
        "yield_vs_rainfed_pct": round(100 * (y_opt - y_rf)
                                      / max(abs(y_rf), 1e-9), 1),
    }
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_exp(arguments) -> int:
    perturb = {k: float(v) for k, v in
               (p.split(":") for p in arguments.perturb.split(",") if p)}
    specification = Specification(
        latitude=arguments.latitude, longitude=arguments.longitude, soil=arguments.soil, crop=arguments.crop,
        horizon=arguments.horizon,
        years=tuple(int(y) for y in arguments.years.split(",") if y),
        realizations=arguments.realizations,
        members=arguments.members, cadence=arguments.cadence,
        noise=arguments.noise, perturb=perturb,
        observed_channels=tuple(arguments.observed_channels.split(",")),
        channels=tuple(arguments.channels.split(",")),
        alphas=tuple(float(a) for a in arguments.alphas.split(",")),
        latent_dimension=arguments.latent_dimension,
        hierarchy=tuple(int(s) for s in arguments.hierarchy.split(",") if s),
        ema=arguments.ema,
        sig_weight=arguments.sig_weight,
        epochs=arguments.epochs, rate=arguments.rate, seed=arguments.seed)
    rep = exp_run(arguments.name, specification)
    output = {"experiment": rep.name, "seconds": rep.seconds,
           "spec": rep.specification, "meta": rep.metadata, "results": rep.results}
    if arguments.output:
        output["saved"] = str(rep.save(arguments.output))
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if arguments.latex:
        sys.stdout.write(rep.latex() + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = _arguments_with_config(list(sys.argv[1:] if argv is None else argv))
    arguments = _build_parser().parse_args(argv)
    match arguments.cmd:
        case "plan":
            return cmd_plan(arguments)
        case "export":
            return cmd_export(arguments)
        case "crawl":
            return cmd_crawl(arguments)
        case "serve":
            return cmd_serve(arguments)
        case "calibrate":
            return cmd_calibrate(arguments)
        case "nas":
            return cmd_nas(arguments)
        case "control":
            return cmd_control(arguments)
        case "exp":
            return cmd_exp(arguments)
        case _:
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
