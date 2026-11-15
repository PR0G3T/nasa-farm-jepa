# FARM — prediction engine

*NASA Space Apps Challenge 2026 — [Field Shift: Adapting Farms with NASA Data](https://www.spaceappschallenge.org/2026/challenges/adapting-farms-with-nasa-data/)*

> *"Farmers around the world navigate shifting temperatures, changing rainfall
> patterns, water shortages, extreme weather, and declining soil health."*

This repository is the engine behind **FARM**: it acquires NASA Earth
observations and local soil/crop data, simulates how each candidate crop
rotation would unfold on a parcel, and ranks them on yield, water use, crop
stress and soil health — so a farmer or agronomist can compare strategies that
protect the field and conserve water under changing conditions. The companion
app that presents all of this is
[nasa-farm-app](https://github.com/PR0G3T/nasa-farm-app). Methodology and
results are detailed in the [research paper](https://github.com/PR0G3T/nasa-farm-jepa/blob/main/A%20Differentiable%20Physics%E2%80%93Residual%20World%20Model%20for%20Crop-Rotation%20Decision%20Support%20on%20NASA%20Earth%20Observations.pdf).

## The idea

A field is a dynamical system. Given its state today — soil water, nitrogen
and carbon pools, crop stage — plus a sequence of management actions (what to
sow, when, irrigation, fertilizer) and plausible weather for the coming
seasons, the engine rolls the future forward, season by season, for every
candidate rotation. Each rotation comes out scored on yield (mean and
downside risk), water consumed, nitrogen lost, days of crop stress, and
soil-carbon change — then ranked.

The model is deliberately two parts:

- a **physics core**: water balance, energy-driven evapotranspiration,
  nitrogen/carbon cycling, phenology and biomass accumulation — the agronomic
  mechanisms a domain expert can audit;
- a **learned residual** (`models/residual.pt`): a small network trained on
  real satellite and yield observations that corrects where the physics
  deviates from what is actually measured. At initialization it is exactly
  zero — the hybrid model *is* the physics until data says otherwise.

Crop traits are calibrated per region (`models/traits.json`) against public
yield references.

## What it serves

`nasa-farm serve` exposes the HTTP API the app consumes:

| Route | Returns |
|---|---|
| `POST /v1/plan` | every candidate rotation evaluated and ranked for a parcel — optionally under climate perturbations (+ΔT, ×precipitation) |
| `GET /v1/field/{id}` | parcel facts: soil profile, rotation history, recent weather and observations |
| `GET /v1/layers` | the observation catalog (dates, per-parcel stats, sector reference) |
| `GET /healthz`, `/v1/meta` | liveness and engine metadata |

## Where the data lives

Acquisition modules pull NASA POWER weather, Harmonized Landsat/Sentinel-2
imagery, ECOSTRESS, SMAP, GPM IMERG, GLDAS, GRACE-FO water storage and NISAR,
alongside SoilGrids profiles and USDA cropland boundaries/classification.
`nasa-farm ingest` stores recent observations, forcing history and field
context in PostgreSQL/PostGIS (`store.py`, compose `db` service) so repeated
requests reuse shared data instead of re-fetching; anything missing falls
back to the live sources. The full product list with links is in the
[app README](https://github.com/PR0G3T/nasa-farm-app#data).

## Research side

`experiments/` and `assimilation/` carry the JEPA-lineage work — latent state
embeddings, hierarchical variants, and data-assimilation benchmarks (open
loop vs EnKF vs hybrid) — with committed results under `results/`. The
optimization toolkit (`optimize/`) does trait calibration, hyperparameter
search and architecture search for the residual net. This is the layer the
paper documents.

## Run it

```bash
docker compose up          # api + PostGIS, the production shape
# or locally:
uv sync --extra serve
nasa-farm serve --port 8000
nasa-farm ingest --bbox ames            # populate Postgres for a region
nasa-farm plan --lat 42.03 --lon -93.63 --horizon 3   # rank rotations directly
```

Other commands: `export` (batch plans → static JSON for the app), `fields`
(boundaries), `crawl` (batch acquisition), `train` (residual fitting),
`traits` / `calibrate` / `nas` (calibration & search), `control`
(irrigation/N scheduling), `exp` (research experiments).

## Guarantees worth knowing

- **Conservation, not convention**: water and nutrient budgets close to
  machine precision — enforced by property tests, not assumed.
- **Pareto-exact**: no rotation is reported optimal if another beats it on
  every objective.
- **Hardware-adaptive**: the evaluator probes CPU vs GPU per workload and
  degrades gracefully under memory pressure; identical results either way.
- **Honest limits**: traits are literature-typical unless calibrated — the
  residual exists precisely to close that gap with observations.

## Links

- App: [nasa-farm-app](https://github.com/PR0G3T/nasa-farm-app)
- Research paper (methodology and results):
  [A Differentiable Physics–Residual World Model for Crop-Rotation Decision Support on NASA Earth Observations](https://github.com/PR0G3T/nasa-farm-jepa/blob/main/A%20Differentiable%20Physics%E2%80%93Residual%20World%20Model%20for%20Crop-Rotation%20Decision%20Support%20on%20NASA%20Earth%20Observations.pdf)
- Deployed API: [https://41kbfozcezrjypq1ig3vgwro.d3vex.fr](https://41kbfozcezrjypq1ig3vgwro.d3vex.fr)
