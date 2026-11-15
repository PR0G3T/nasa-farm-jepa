# nasa-farm

**Physics-informed latent world model for crop-rotation decision support.**
NASA Space Apps 2026 — *Field Shift: Adapting Farms with NASA Data*.
Runs on **CPython 3.15 free-threaded** (no GIL) — see *Performance*.

A field is a dynamical system. Given its state today, a sequence of management
actions (what to sow, when, how much water/N) and weather realizations, this
engine rolls the future forward and scores each candidate rotation on yield,
water, soil health and risk — every output tagged to the dataset it came from.

## Why this is a world model (and what kind)

Not a video model. Following the JEPA line of world models (V-JEPA 2, LeJEPA),
the state lives in a **latent-but-interpretable space** and planning happens by
rolling latent transitions — no pixels are ever rendered. The state vector
is the physics:

```
s_t = ( sw1, sw2,       # soil water, 2 layers (mm)
        n_min, n_org,   # mineral / organic N (kg/ha)
        soc, litter,    # carbon pools (kg/ha)
        gdd, lai,       # phenology, leaf area
        biomass, crop, growing )

T(s_t, a_t, f_t) → s_{t+1}     a_t = (sow, irrigation, n_fert)
                               f_t = (tmin, tmax, srad, precip, rh, u2, doy)
```

`T` = **differentiable physics core** (FAO-56 Penman–Monteith ET0, Ritchie
two-layer water balance, first-order N/C kinetics, RUE canopy) **+ a
zero-initialized neural residual** (`ResidualNet`). The residual is exactly
identity at init — the hybrid model *is* the physics until trained, and data
assimilation has a gradient path into it (proven by tests).

## Invariants (test-enforced, not asserted)

| Invariant | Where |
|---|---|
| `Δ(sw1+sw2) ≡ infil − runoff − Es − T1 − T2 − deep` (machine precision, 200 random cases) | `dynamics/water.py` |
| `air1 ≤ sw1 ≤ sat1`, `wp2 ≤ sw2 ≤ sat2` every day | `dynamics/water.py` |
| No flux from an empty reservoir | `dynamics/water.py` |
| N pools non-negative; fixation only legumes, down-regulated when N sufficient | `dynamics/biogeochemistry.py` |
| LAI = 0 before emergence, 0 at maturity; perennials never senesce | `dynamics/crop.py` |
| Fresh residual ⇒ hybrid rollout bitwise-identical to physics rollout | `dynamics/residual.py` |
| Pareto front exact (enumerated, O(R²) dominance) | `plan.py` |
| POWER/SG payloads → correct tensors, Feb 29 dropped, fills → NaN→default | `acquire/` |

## Layout (SRP: one concept per module)

```
src/nasa_farm/
├── constants.py            physical constants — the only magic numbers
├── types.py                Forcing / Actions (B,T) containers
├── state/
│   ├── soil.py             Saxton–Rawls pedotransfer → hydraulic storages
│   ├── crops.py            crop trait table (C,P) — crops are data, not code
│   └── field.py            FieldState, Field, initial_state
├── dynamics/
│   ├── atmosphere.py       Ra, FAO-56 PM ET0 (+ Hargreaves fallback), GDD, fT
│   ├── water.py            conservative 2-layer water balance
│   ├── biogeochemistry.py  N & C pools
│   ├── crop.py             phenology, LAI, RUE biomass, harvest
│   ├── residual.py         zero-init learnable correction (data assimilation)
│   └── model.py            WorldModel: step / rollout
├── forcing.py              scenario ensembles (history bootstrap, ΔT, ΔP)
├── heads.py                SeasonAccumulator — O(B·H) metrics, no trajectory
├── plan.py                 enumerate, evaluate, Pareto, scalarize, recommend
├── acquire/
│   ├── base.py             DataSource protocol, BBox/Temporal
│   ├── power.py            NASA POWER agro API (no key)
│   ├── earthdata.py        CMR via earthaccess (SMAP, IMERG, HLS, ECOSTRESS, GRACE)
│   └── soilgrids.py        ISRIC SoilGrids → SoilProfile
├── optimize/
│   ├── space.py            Search space + Sobol sampler (torch.quasirandom)
│   ├── targets.py          ObservedTrajectory + differentiable χ² loss
│   ├── trainer.py          fit() — Adam/clip/early-stop on rollout gradients
│   ├── hpo.py              gradient_hpo (variational) + blackbox (Sobol+ASHA,
│   │                       Optuna TPE/NSGA-II via `hpo` extra)
│   └── nas.py              zero-cost proxies (SynFlow/Jacobian-cov) + ASHA NAS
└── cli.py                  `nasa-farm plan|calibrate|nas`
```

## Usage

```bash
uv sync --all-extras

# Offline (synthetic climatology, deterministic):
uv run nasa-farm plan --lat 42.0 --lon -93.6 --horizon 3 --offline

# Live (POWER history 5y × all rotations, SoilGrids soil):
uv run nasa-farm plan --lat 42.03 --lon -93.63 --horizon 3 --years 5 \
    --crops maize,soybean,wheat,clover,fallow \
    --delta-t 1.5 --precip-scale 0.85 --irrig 4 --fert 120

# Repeatable runs: TOML defaults + selective CLI overrides
uv run nasa-farm plan --lat 42.0 --lon -93.6 --config plan.toml --fert 200
# plan.toml → [plan] horizon = 3, crops = "...", years = 5, offline = true …
```

Output: exact **Pareto front** of rotations + priority-weighted ranking, each
rotation scored on yield mean/10 %-risk quantile, water used, N leached,
stress days, ΔSOC, end-of-horizon mineral N.

Programmatic:

```python
from nasa_farm import CropLibrary, Field, WorldModel, initial_state
from nasa_farm.plan import enumerate_rotations, evaluate, recommend

model = WorldModel(CropLibrary.default())
ev = evaluate(model, field, plans, forcing_ensemble)   # batched: plans×scenarios
rec = recommend(ev, weights={"yield_mean": 2.0, "soc_gain": 1.0})
```

## Data provenance

| Product | Role | Access |
|---|---|---|
| NASA POWER agro (`T2M_MIN/MAX, PRECTOTCORR, ALLSKY_SFC_SW_DWN, RH2M, WS10M`) | daily forcing → PM-ET0, GDD | API, no key |
| SoilGrids v2 (`clay, sand, soc`, 0–30 cm) | pedotransfer → soil storages | REST, no key |
| SMAP `SPL3SMP_E`, GPM `GPM_3IMERGDF`, HLS `HLSL30`, ECOSTRESS `ECO_L3T_JET`, GRACE-FO mascons | observation streams for assimilation into `s_t` / residual training | `earthaccess`, Earthdata login |

## Performance — adaptive hardware engine

`evaluate()` auto-exploits everything the machine offers, with graceful
degradation instead of crashes:

- **Device probe** (`device="auto"`): a one-time ~1 s benchmark times the
  real rollout on CPU vs GPU (cached per workload bucket) and keeps the
  winner. Measured on an RTX 4060: CPU+threads wins below ~5k trajectory
  rows (launch-bound kernels); CUDA wins beyond (≈1.9× at 13k rows).
  Small runs skip the probe entirely. Universal PyPI torch wheel serves
  both paths — no separate build to install.
- **Threads** (`workers=0`): physical cores on no-GIL CPython (SMT siblings
  excluded — measured, not `os.cpu_count()`), 1 on GIL builds and on GPU.
  Shard results are bit-identical to serial (test-enforced).
- **Memory budget**: plan blocks are sized against *currently free*
  RAM/VRAM (`mem_budget` of it), so giant sweeps stream instead of OOMing.
- **OOM ladder** (tested): block ÷ 2 → threads ÷ 2 → CUDA → CPU → raise.
  Non-OOM errors always propagate — a bug is never masked by lower settings.
- **`compile="auto"`**: `torch.compile` engages past ~30 M field-days
  (dynamic shapes, ~2×). Note: inductor pulls triton, which currently
  re-enables the GIL — compiled path therefore runs serially, where its
  kernel fusion wins anyway.

~1.2 M field-days/s single-core float64 · ~1.7× with auto threads on 8
physical cores · ~1.9× further with CUDA on large sweeps. Rollouts stay
O(1) extra memory — seasonal metrics accumulate online.

## HPO & NAS (`optimize/`)

Because the dynamics core is differentiable, continuous parameters optimize
**through the rollout** — variational data assimilation, strictly stronger
than black-box evaluation per unit of compute:

- `gradient_hpo` + `trait_leaf(model)`: traits/soil/residual calibrated by
  direct `d(loss)/d(param)` over the full trajectory. Proven on an
  identifiability test: perturbing `rue` ×0.8 → fit recovers ≈99.9 %.
- `blackbox_hpo`: Sobol coverage + ASHA for non-differentiable knobs;
  Optuna TPE / NSGA-II kicks in automatically with `uv sync --extra hpo`.
- `nas_search` (`nasa-farm nas`): two-stage search over residual
  architectures — zero-cost proxies (SynFlow signal propagation, Jacobian
  covariance, grad-norm) rank candidates before any fitting, then
  successive halving spends fit budget only on contenders.
- `ObservedTrajectory` is the loss contract real SMAP/HLS/yield series will
  satisfy: same channels as `heads.METRICS`, NaN = unobserved, per-channel
  χ² scaling. `synthetic_observation` fills it from a rollout meanwhile —
  the whole pipeline is testable *today*, uncalibrated claims included.

```bash
uv run nasa-farm calibrate --lat 42 --lon -93.6 --param rue --scale 0.8
uv run nasa-farm nas --candidates 24 --top-k 8 --rungs 10,30,90
```

## Calibration notes (honest limits)

- Crop traits are literature-typical, not regionally calibrated; the residual
  net exists precisely to close this gap once trained on observations.
- Rainfed yields trend conservative (multiplicative stress limiters).
- 365-day years; Feb 29 is dropped in forcing preparation.
- LAI follows phenology (GDD); drought cuts biomass, not canopy expansion.

## Tests

`uv run pytest` — 148 tests: conservation laws (Hypothesis property tests),
bounds, lifecycle events, legume economics, Pareto exactness, API parsing,
differentiability, thread/streamed bit-parity, compiled-vs-eager parity,
OOM-ladder recovery, device probing, HPO identifiability, space sampling,
ASHA pruning, NAS proxies + search, and full offline pipeline + CLI tests.

## 1 km training grid (`nasa_farm.grid`)

One sample file per training product (`claude/download-access.md`, 53
products) downloaded and put on a common grid: **EPSG:5070 (CONUS Albers,
equal-area), 1 km cells snapped to multiples of 1000 m**, so grids of
different AOIs tile exactly.

```bash
uv sync --extra grid
uv run python -m nasa_farm.grid.run --list                 # products + credential status
uv run python -m nasa_farm.grid.run --aoi ames --date 2024-07-15
uv run python -m nasa_farm.grid.run --only smap-l4,cdl,3dep --jobs 8
```

Output in `data/grid1km/<aoi>_<date>/`: one CF NetCDF per product,
`stack.nc` (all products, vars `<product>__<var>`), `report.md` (ok /
skipped / failed, source granule, method, coverage). Raw files are cached
in `data/raw/<product>/`.

| Variable kind | Source finer than 1 km | Source coarser than 1 km |
|---|---|---|
| continuous | area-weighted average of QA-valid pixels (+ `valid_frac`) | bilinear |
| categorical (CDL, NLCD, SCL…) | class area fractions (`*_frac`, last = other) + mode | nearest |
| footprints / gauges (GEDI, USGS) | per-cell mean + count | — |
| polygons (CSB, STATSGO2) | area-weighted on a 100 m sub-grid | — |

Credentials (only those you have are used; the rest are reported as
skipped): Earthdata `~/.netrc` · `OPENET_API_KEY` · `~/.cdsapirc` · AWS
(requester-pays NLCD / Landsat L3) · `LFPS_EMAIL` (LANDFIRE).
