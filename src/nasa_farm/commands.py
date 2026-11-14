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
from nasa_farm.acquire.power import Power, series_to_forcing
from nasa_farm.acquire.soilgrid import SoilGrids
from nasa_farm.constants import DTYPE
from nasa_farm.control import path_integral
from nasa_farm.experiments import Specification, experiments
from nasa_farm.experiments import run as exp_run
from nasa_farm.forcing import augment, synthetic_year, tile_to_horizon
from nasa_farm.optimize.hyperparameter import gradient
from nasa_farm.optimize.architecture import search
from nasa_farm.optimize.targets import (
    ObservedTrajectory, season_metrics, synthetic_observation, trajectory_loss,
)
from nasa_farm.optimize.trainer import fit, leaf
from nasa_farm.plan import expand_actions
from nasa_farm.service import SCENARIO_KEYS, PlanRequest, Point, plan
from nasa_farm.state.crops import INDEX
from nasa_farm.state.soil import SoilProfile
from nasa_farm.types import FORCING_VARS, Forcing

DEFAULT_CROPS_ARG = "maize,soybean,wheat,clover,fallow"


def _plan_flags(parser: argparse.ArgumentParser) -> None:
    """Shared compute flags for `plan` and `export` — one definition."""
    parser.add_argument("--horizon", type=int, default=3, help="rotation length, years")
    parser.add_argument("--crops", default=DEFAULT_CROPS_ARG)
    parser.add_argument("--years", type=int, default=5, help="climatology depth")
    parser.add_argument("--delta-t", type=float, default=0.0, help="+°C scenario shift")
    parser.add_argument("--precip-scale", type=float, default=1.0)
    parser.add_argument("--irrigation", type=float, default=0.0, help="mm/day post-sowing")
    parser.add_argument("--fertilizer", type=float, default=0.0, help="kg N/ha at sow+30d")
    parser.add_argument("--soil", default=None, help="texture preset override")
    parser.add_argument("--depths", default=None,
                    help="layer thicknesses m, comma-sep (e.g. 0.3,0.6,0.9)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--weights", default=None,
                    help="objective weights k=v,... e.g. yield_mean=2,soc_gain=1")
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
    parser = argparse.ArgumentParser(prog="nasa-farm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="rank rotations for a field")
    _plan_flags(plan)
    plan.add_argument("--latitude", type=float, required=True)
    plan.add_argument("--longitude", type=float, required=True)

    export = subparsers.add_parser("export", help="batch plan → static JSON for the app")
    _plan_flags(export)
    export.add_argument("--fields", required=True,
                     help="fields JSON path, or '-' for stdin")
    export.add_argument("--output", required=True, help="output JSON path")
    export.add_argument("--scenario", action="append", default=None,
                     help="name[:k=v,...] — repeatable; k ∈ "
                          + ",".join(sorted(SCENARIO_KEYS)))

    crawl = subparsers.add_parser("crawl", help="batch acquisition → data lake")
    crawl.add_argument("--fields", required=True,
                    help="fields JSON path, or '-' for stdin")
    crawl.add_argument("--output", default="lake", help="lake root dir")
    crawl.add_argument("--years", type=int, default=5)
    crawl.add_argument("--sources", default="power,soil",
                    help="power,soil,smap,ecostress-seb,ecostress-met")
    crawl.add_argument("--workers", type=int, default=8)

    serve = subparsers.add_parser("serve", help="HTTP API (granian + fastapi)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=1,
                    help="granian worker processes")
    serve.add_argument("--tunnel", action="store_true",
                    help="cloudflared quick tunnel → public URL")

    traits = subparsers.add_parser(
        "traits", help="calibrate crop traits to regional yield targets")
    traits.add_argument("--latitude", type=float, required=True)
    traits.add_argument("--longitude", type=float, required=True)
    traits.add_argument("--years", type=int, default=5)
    traits.add_argument("--target", required=True,
                    help="crop=kg/ha,... e.g. maize=9000,wheat=7400")
    traits.add_argument("--fert", default="",
                    help="per-crop N kg/ha, crop=kg,... — else --fertilizer")
    traits.add_argument("--fertilizer", type=float, default=0.0,
                    help="fallback N kg/ha for crops not in --fert")
    traits.add_argument("--params", default="rue=1.5:4.5",
                    help="joint bounded traits: name=lo:hi,... (sigmoid reparam)")
    traits.add_argument("--soil", default=None, help="texture preset")
    traits.add_argument("--offline", action="store_true",
                    help="synthetic forcing, no network")
    traits.add_argument("--epochs", type=int, default=160)
    traits.add_argument("--rate", type=float, default=0.05)
    traits.add_argument("--output", required=True)

    calibrate = subparsers.add_parser("calibrate", help="gradient HPO on a trait cell")
    calibrate.add_argument("--latitude", type=float, required=True)
    calibrate.add_argument("--longitude", type=float, required=True)
    calibrate.add_argument("--crop", default="maize")
    calibrate.add_argument("--parameter", default="rue", help="trait name to perturb+recover")
    calibrate.add_argument("--scale", type=float, default=0.8,
                   help="synthetic truth = trait × scale")
    calibrate.add_argument("--channel", default="yield_kg", help="METRICS channel")
    calibrate.add_argument("--epochs", type=int, default=120)
    calibrate.add_argument("--rate", type=float, default=0.05)

    search = subparsers.add_parser("nas", help="residual architecture search")
    search.add_argument("--latitude", type=float, default=42.0)
    search.add_argument("--longitude", type=float, default=-93.6)
    search.add_argument("--candidates", type=int, default=24)
    search.add_argument("--top-k", type=int, default=8)
    search.add_argument("--rungs", default="10,30,90", help="fit budgets, comma-sep")
    search.add_argument("--seed", type=int, default=0)

    control = subparsers.add_parser("control", help="MPPI irrigation/N scheduling")
    control.add_argument("--latitude", type=float, default=42.0)
    control.add_argument("--crop", default="maize")
    control.add_argument("--scenarios", type=int, default=8,
                    help="forcing ensemble size")
    control.add_argument("--samples", type=int, default=64)
    control.add_argument("--iterations", type=int, default=6)
    control.add_argument("--weights", default=None,
                    help="reward mix, e.g. yield_kg=1.0,n_leached=-0.5")
    control.add_argument("--risk", default="mean",
                    choices=("mean", "cvar", "conf"),
                    help="scenario risk aggregation")
    control.add_argument("--alpha", type=float, default=0.2,
                    help="CVaR tail level / conformal miscoverage")
    control.add_argument("--sow-shifts", default="0",
                    help="sow-day shifts searched, comma-sep (e.g. -14,0,14)")
    control.add_argument("--seed", type=int, default=0)

    experiment = subparsers.add_parser("exp", help="run a research experiment")
    experiment.add_argument("name", choices=experiments())
    experiment.add_argument("--latitude", type=float, default=42.0)
    experiment.add_argument("--longitude", type=float, default=-93.6)
    experiment.add_argument("--soil", default="silt_loam")
    experiment.add_argument("--crop", default="maize")
    experiment.add_argument("--horizon", type=int, default=1)
    experiment.add_argument("--years", default="",
                    help="POWER years, comma-sep; empty = synthetic forcing")
    experiment.add_argument("--realizations", type=int, default=1,
                    help="held-out truth realizations scored")
    experiment.add_argument("--members", type=int, default=16,
                    help="forcing-ensemble / DA ensemble size")
    experiment.add_argument("--cadence", type=int, default=10,
                    help="days between assimilated observations")
    experiment.add_argument("--noise", type=float, default=0.05,
                    help="relative observation noise")
    experiment.add_argument("--perturb", default="",
                    help="truth trait shifts, e.g. rue:0.8,gdd_mat:0.95")
    experiment.add_argument("--channels", default="yield_kg",
                    help="seasonal METRICS channels, comma-sep")
    experiment.add_argument("--obs-channels", dest="observed_channels", default="ndvi,sm1",
                    help="satellite channels assimilated, comma-sep")
    experiment.add_argument("--alphas", default="0.1",
                    help="interval levels for PICP/interval score")
    experiment.add_argument("--latent-dimension", type=int, default=8,
                    help="JEPA latent size")
    experiment.add_argument("--hierarchy", default="",
                    help="latent strides, comma-sep; empty = flat JEPA")
    experiment.add_argument("--ema", type=float, default=0.996,
                    help="JEPA target-encoder momentum")
    experiment.add_argument("--sig-weight", type=float, default=1.0,
                    help="SIGReg regularization strength")
    experiment.add_argument("--extra", default="",
                    help="truth-only extra processes (dynamics.EXTRAS), comma-sep")
    experiment.add_argument("--epochs", type=int, default=120)
    experiment.add_argument("--rate", type=float, default=0.05)
    experiment.add_argument("--seed", type=int, default=0)
    experiment.add_argument("--output", default=None, help="save report JSON here")
    experiment.add_argument("--latex", action="store_true",
                    help="also print booktabs rows of scalar results")
    experiment.add_argument("--config", default=None,
                    help="TOML file; CLI flags override config values")
    return parser


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
    for k, value in config.get(argv[0], config).items():
        key = "--" + k.replace("_", "-")
        match value:
            case bool() if value:  flags.append(key)
            case bool():       pass
            case _:            flags += [key, str(value)]
    return [argv[0], *flags, *argv[1:]]


def _request(arguments, points: list[Point], **over) -> PlanRequest:
    keywords = dict(horizon=arguments.horizon, crops=arguments.crops.split(","),
              years=arguments.years, delta_t=arguments.delta_t,
              precip_scale=arguments.precip_scale, irrig=arguments.irrigation,
              fertilizer=arguments.fertilizer, co2=arguments.co2, soil=arguments.soil,
              source=getattr(arguments, "source", "power"),
              members=getattr(arguments, "members", 16),
              residual=not arguments.no_residual, offline=arguments.offline,
              top_k=arguments.top_k,
              weights=({k: float(v) for k, v in
                        (w.split("=") for w in arguments.weights.split(","))}
                       if getattr(arguments, "weights", None) else None),
              depths=([float(x) for x in arguments.depths.split(",")]
                      if arguments.depths else None))
    keywords.update(over)
    return PlanRequest(points=points, **keywords)


def cmd_plan(arguments) -> int:
    response = plan(_request(arguments, [Point(latitude=arguments.latitude, longitude=arguments.longitude)]),
               workers=arguments.workers, device=arguments.device,
               compile=arguments.compile)
    site = response.results[0]
    if site.status != "ok":
        sys.stderr.write(f"{site.detail}\n")
        return 1

    def row(i: int) -> dict:
        return {"rotation": response.rotations[i], "score": site.scores[i],
                **dict(zip(response.columns, site.objectives[i]))}

    json.dump({"plans_evaluated": site.plans_evaluated,
               "scenarios": site.scenarios,
               "top": [row(i) for i in site.ranking],
               "pareto": [row(i) for i in site.pareto]},
              sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _scenarios(specs: list[str] | None) -> dict[str, dict[str, float]]:
    """`name:k=v,...` specs → ordered {name: overrides}; bare name → {}."""
    output: dict[str, dict[str, float]] = {"baseline": {}}
    for specification in specs or ():
        name, _, rest = specification.partition(":")
        kv = dict((k, float(value)) for k, _, value in
                  (part.partition("=") for part in rest.split(",") if part))
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
    points = [Point(id=str(f["id"]), latitude=f["latitude"],
                    longitude=f["longitude"]) for f in items]
    metadata = {str(f["id"]): {k: f[k] for k in ("name", "crop", "ha") if k in f}
            for f in items}

    scenarios = _scenarios(arguments.scenario)
    output = Path(arguments.output)
    shards_root = output.parent / "plan"
    shards_root.mkdir(parents=True, exist_ok=True)
    index = {"version": __version__,
           "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
           "request": {"horizon": arguments.horizon, "crops": arguments.crops.split(","),
                       "years": arguments.years, "irrig": arguments.irrigation,
                       "fert": arguments.fertilizer, "co2": arguments.co2,
                       "offline": arguments.offline},
           "columns": [], "directions": [], "rotations": [],
           "fields": metadata, "scenarios": {}}
    shards: dict[str, dict] = {}
    for name, over in scenarios.items():
        response = plan(_request(arguments, points, **over),
                   workers=arguments.workers, device=arguments.device,
                   compile=arguments.compile)
        index["columns"] = response.columns
        index["directions"] = response.directions
        index["rotations"] = response.rotations
        for i, r in enumerate(response.results):
            if r.point.id:
                shard = shards.setdefault(
                    r.point.id, {"history": r.history, "scenarios": {}})
                shard["scenarios"][name] = r.model_dump(
                    exclude={"point", "history"})
            sys.stderr.write(
                f"{name} [{i + 1}/{len(response.results)}] "
                f"{r.point.id or r.point.latitude} {r.status}"
                f"{f' — {r.detail}' if r.detail else ''}\n")
        index["scenarios"][name] = {"params": over}
    output.write_text(json.dumps(index) + "\n")
    written = set()
    for fid, shard in shards.items():
        target = shards_root / f"{fid}.json"
        target.write_text(json.dumps(shard) + "\n")
        written.add(target.name)
    for stale in shards_root.glob("*.json"):
        if stale.name not in written:
            stale.unlink()
    sys.stderr.write(f"wrote {output} + {len(written)} shards\n")
    return 0


def cmd_traits(arguments) -> int:
    """Fit bounded traits per crop against absolute yield targets —
    variational (gradient) calibration of the physics core on regional
    statistics, using sigmoid reparameterisation inside --params bounds.

    Forcing = the same real POWER years serving uses, rolled out as one
    continuous n-year trajectory: an autumn-sown crop harvests inside the
    next calendar year's metric cell, so year 0 is establishment and the
    loss scores years 1..n−1."""
    from pathlib import Path

    crops = CropLibrary.default()
    targets = {c: float(v) for c, v in
               (t.split("=") for t in arguments.target.split(","))}
    ferts = {c: float(v) for c, v in
             (t.split("=") for t in arguments.fert.split(",") if t)}
    bounds = {}
    for entry in arguments.params.split(","):
        name, rng = entry.split("=")
        bounds[name] = tuple(float(v) for v in rng.split(":"))
    unknown = [c for c in targets if c not in crops.names]
    if unknown:
        sys.stderr.write(f"unknown crops: {', '.join(unknown)}\n")
        return 2

    last = dt.date.today().year - 1
    years = list(range(last - arguments.years + 1, last + 1))
    if arguments.offline:
        series = [synthetic_year(arguments.latitude, seed=y) for y in years]
    else:
        raw = Power().data_years(arguments.latitude, arguments.longitude, years)
        series = [series_to_forcing(d, y) for d, y in zip(raw, years)]
    forcing = Forcing(*[torch.cat([getattr(f, v) for f in series], dim=-1)
                        for v in FORCING_VARS])
    soil = (SoilProfile.preset(arguments.soil)
            if arguments.soil else SoilGrids().profile(
                arguments.latitude, arguments.longitude))
    n = len(years)
    from nasa_farm.heads import METRICS
    channel = METRICS.index("yield_kg")
    field = Field(initial_state(1, soil), soil,
                  torch.full((1,), arguments.latitude, dtype=DTYPE))
    fitted, report = {}, {}
    for crop, target in targets.items():
        model = WorldModel(crops, residual=False)
        base = model.crop_table.detach().clone()
        row = crops.index(crop)
        fitted_params = {}
        for name in bounds:
            if name not in INDEX:
                sys.stderr.write(f"unknown trait: {name}\n")
                return 2
            col = INDEX[name]
            lo, hi = bounds[name]
            start = base[row, col].clamp(lo + 1e-4, hi - 1e-4)
            span = hi - lo
            parameter = leaf(torch.logit((start - lo) / span))
            fitted_params[name] = (col, parameter,
                                   lambda p=parameter, lo=lo, span=span:
                                   lo + span * torch.sigmoid(p))
        # One continuous n-year rollout — an autumn-sown crop harvests in
        # the next calendar year's metric cell, so year 0 is establishment
        # and stays unobserved; the loss scores years 1..n−1.
        values = torch.full((n, len(METRICS)), float("nan"), dtype=DTYPE)
        values[1:, channel] = target
        observed = ObservedTrajectory(values=values)
        actions = expand_actions(
            torch.full((1, n), row, dtype=torch.long), base,
            n_fert_kg=ferts.get(crop, arguments.fertilizer))

        def loss() -> torch.Tensor:
            table = base.clone()
            for col, _, value in fitted_params.values():
                table[row, col] = value()
            model.crop_table = table
            return trajectory_loss(
                season_metrics(model, field, forcing, actions), observed)

        result = gradient(
            loss, {name: p for name, (col, p, _) in fitted_params.items()},
            epochs=arguments.epochs, rate=arguments.rate)
        table = base.clone()
        for col, _, value in fitted_params.values():
            table[row, col] = value().detach()
        model.crop_table = table
        with torch.inference_mode():
            achieved = float(season_metrics(
                model, field, forcing, actions)[0, 1:, channel].mean())
        fitted[crop] = {name: round(float(value().detach()), 6)
                        for name, (_, _, value) in fitted_params.items()}
        report[crop] = {"target": target, "achieved": round(achieved, 1),
                        "fert": ferts.get(crop, arguments.fertilizer),
                        "loss": round(result.best_loss, 6)}
        sys.stderr.write(
            f"{crop}: {fitted[crop]} · yield "
            f"{achieved:.0f}/{target:.0f} kg/ha\n")

    output = Path(arguments.output)
    existing = (json.loads(output.read_text()).get("traits", {})
                if output.is_file() else {})
    payload = {"meta": {"latitude": arguments.latitude,
                        "longitude": arguments.longitude,
                        "years": years, "params": sorted(bounds),
                        "fertilizer": arguments.fertilizer,
                        "offline": arguments.offline},
               "traits": {**existing, **fitted}, "report": report}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    sys.stderr.write(f"wrote {output}\n")
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
    crop_index, parameter_index = crops.index(arguments.crop), INDEX[arguments.parameter]

    truth = WorldModel(crops, residual=False)
    tbl_t = truth.crop_table.detach().clone()
    tbl_t[crop_index, parameter_index] *= arguments.scale
    truth.crop_table = tbl_t

    soil = SoilProfile.preset("silt_loam")
    field = Field(initial_state(1, soil), soil,
                  torch.tensor([arguments.latitude], dtype=DTYPE))
    forcing = tile_to_horizon(synthetic_year(arguments.latitude, seed=0), 1)
    actions = _demo_actions(crops, forcing.days, arguments.crop)
    observed = synthetic_observation(truth, field, forcing, actions, (arguments.channel,))

    candidate = WorldModel(crops, residual=False)
    base = candidate.crop_table.detach().clone()
    parameter = leaf(base[crop_index, parameter_index])

    def loss():
        table = base.clone()
        table[crop_index, parameter_index] = parameter
        candidate.crop_table = table
        return trajectory_loss(
            season_metrics(candidate, field, forcing, actions), observed)

    result = gradient(loss, {"v": parameter}, epochs=arguments.epochs, rate=arguments.rate)
    truth_v, start_v, fit_v = (float(tbl_t[crop_index, parameter_index]),
                               float(base[crop_index, parameter_index]), float(parameter.detach()))
    span = truth_v - start_v
    output = {
        "param": arguments.parameter, "crop": arguments.crop, "channel": arguments.channel,
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
        parameters = {f"p{i}": parameter for i, parameter in enumerate(net.parameters())}
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
               {k: float(value) for k, value in
                (w.split("=") for w in arguments.weights.split(","))})
    result = path_integral(model, field, ensemble, base, iterations=arguments.iterations,
               samples=arguments.samples, weights=weights, risk=arguments.risk,
               alpha=arguments.alpha,
               sow_shifts=tuple(int(x) for x in arguments.sow_shifts.split(",")),
               seed=arguments.seed)

    index = {n: i for i, n in enumerate(METRICS)}

    def _yield(actions) -> float:
        accumulate = SeasonAccumulator(ensemble.batch, ensemble.days // 365)
        with torch.inference_mode():
            model.rollout(field.repeat(ensemble.batch), ensemble,
                          actions.repeat(ensemble.batch), observer=accumulate)
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
    perturb = {k: float(value) for k, value in
               (parser.split(":") for parser in arguments.perturb.split(",") if parser)}
    specification = Specification(
        latitude=arguments.latitude, longitude=arguments.longitude, soil=arguments.soil, crop=arguments.crop,
        horizon=arguments.horizon,
        years=tuple(int(y) for y in arguments.years.split(",") if y),
        realizations=arguments.realizations,
        members=arguments.members, cadence=arguments.cadence,
        noise=arguments.noise, perturb=perturb,
        extra=tuple(arguments.extra.split(",") if arguments.extra else ()),
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
    match arguments.command:
        case "plan":
            return cmd_plan(arguments)
        case "export":
            return cmd_export(arguments)
        case "traits":
            return cmd_traits(arguments)
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
