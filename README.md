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
└── cli.py                  `nasa-farm plan` — fetch→evaluate→rank in one shot
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

## Performance

- **Free-threaded CPython 3.15** (`.python-version` → `3.15+freethreaded`, GIL
  off): `evaluate(workers=N)` shards the plan axis across threads — `step` is
  pure, so scaling is real, not fake-async. `workers=0` auto-detects: CPU
  count on no-GIL builds, 1 elsewhere. Rollouts are dispatch-bound, so
  intra-op OMP is parked at 1 thread during parallel blocks.
- **`model.enable_compile()`** fuses the daily step via `torch.compile`
  (~2× on repeated rollouts; ~10 s graph-build on first call — opt-in via
  CLI `--compile`, meant for sweeps, not one-shots). Compiled-vs-eager
  parity is test-enforced.
- ~1.2 M field-days/s single-core float64; POWER year fetches are
  parallelized (I/O). Rollouts stay O(1) extra memory — seasonal metrics
  accumulate online; nothing per-day is stored unless an observer asks.

## Calibration notes (honest limits)

- Crop traits are literature-typical, not regionally calibrated; the residual
  net exists precisely to close this gap once trained on observations.
- Rainfed yields trend conservative (multiplicative stress limiters).
- 365-day years; Feb 29 is dropped in forcing preparation.
- LAI follows phenology (GDD); drought cuts biomass, not canopy expansion.

## Tests

`uv run pytest` — 125 tests: conservation laws (Hypothesis property tests),
bounds, lifecycle events, legume economics, Pareto exactness, API parsing,
differentiability, thread-sharded-vs-serial bit-parity, compiled-vs-eager
parity, and a full offline pipeline + CLI integration test.
