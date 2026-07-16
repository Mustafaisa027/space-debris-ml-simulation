# Space Debris Conjunction Simulation

Open-source LEO conjunction-analysis and machine-learning risk-classification
framework accompanying IAC 2026 paper **114764**, *"Machine Learning-Based
Simulation Approach for Assessing Space Debris Collision Risk in Low Earth
Orbit."*

The pipeline:

- reads Two-Line Element (TLE) data;
- propagates every object with the SGP4 model (via Skyfield);
- for each LEO pair computes the minimum approach distance, Time of Closest
  Approach (TCA), relative velocity at TCA, and LEO altitude features;
- screens for candidate conjunctions and writes `identified_conjunctions.csv`;
- builds a labelled dataset and compares supervised classifiers
  (Logistic Regression, Random Forest, SVM, XGBoost) against a classical
  fixed-distance threshold baseline.

## What is (and isn't) claimed

This framework computes geometric/kinematic conjunction metrics and provides a
reproducible ML comparison harness. It does **not** compute a true probability
of collision (Pc): that requires covariance data, object size, and an error
ellipsoid, which are outside the scope of the public-TLE, open-data setting the
paper targets. `risk_score` is a transparent ranking aid, not a Pc.

## Installation

```bash
python -m pip install -r requirements.txt
# or, as an editable package:
python -m pip install -e .
```

## Quick start (offline demo)

`data/sample_tles.txt` holds five real objects whose epochs are now years old;
propagating stale TLEs gives physically unreliable geometry, and five objects
almost never yield a genuine close approach. So for an offline, self-contained
demo the repo ships a deterministic **synthetic** catalogue generator:

```bash
python src/make_demo_tles.py --count 60 --output data/demo_tles.txt
python src/run_pipeline.py \
    --tle data/demo_tles.txt \
    --horizon-minutes 150 --step-minutes 3 \
    --candidate-threshold-km 500 \
    --fixed-threshold-km 50 \
    --label-threshold-km 200 \
    --label-relative-velocity-km-s 5
```

The `DEMO-SAT-xx` objects are clearly named so they are never mistaken for a
real catalogue. They exist only to exercise the code path, plots, and model
comparison without network access.

## Scientific runs (real data)

For results that mean something, fetch current elements and run against them:

```bash
# a handful of named objects
python src/fetch_tles.py --catnr 25544 25338 20580 25994 27424
python src/run_pipeline.py --tle data/live_tles_YYYYMMDD_HHMMSS.txt

# or a whole CelesTrak group (denser -> more real conjunctions)
python src/fetch_tles.py --group STARLINK --max-objects 120 --output data/starlink.txt
python src/run_pipeline.py --tle data/starlink.txt
```

Default thresholds are now physically interpretable screening scales (tens of
km), not the earlier 3000-10000 km placeholders. If any TLE epoch is older than
`--max-tle-age-hours` (default 14 days) the pipeline prints a data-quality
warning and the geometry should be treated as illustrative only.

## Building a proper dataset (60-day collection)

A single snapshot yields very few conjunctions, so no classifier can reliably
beat the fixed-distance baseline on it — that is a statistical fact, not a code
bug. The paper's comparison is meant to run on **accumulated observations**.
CelesTrak GP data should not be polled more often than every 2 hours; the
collector enforces this as a floor.

```bash
# one cycle
python src/collect_observations.py --once

# 60-day background collection (see scripts/ for Windows Task Scheduler helpers)
python src/collect_observations.py --days 60 --interval-hours 2

# rebuild corrected-TCA history from immutable snapshots
python src/resimulate_snapshots.py --config config/experiment_60_days.json

# pair-held-out, time-ordered evaluation over corrected-TCA history
python src/train_from_history.py
```

The training command defaults to the corrected frozen-cohort
`conjunction_observations_resimulated_iac26_75_v1.csv`, not the live collector
history or the older mixed 5/43/75-object archive.
It writes the canonical full-feature comparison and a separate
`*_without_label_rule_features.csv` ablation that removes minimum distance,
relative velocity, and the three RIC position components whose norm reconstructs
minimum distance. The ablation uses the identical held-out pair/time split and
exposes how much performance comes from directly recovering the transparent
proxy-label rule.

If an older checkout already accumulated history, rebuild the schema-safe
version from immutable run outputs before training:

```bash
python src/rebuild_history.py
```

TLE input is fail-closed: malformed/checksum-invalid responses and catalogue
coverage below 90% are rejected, writes are atomic, and `--provider auto` can
use authenticated Space-Track GP as a fallback when
`SPACETRACK_IDENTITY`/`SPACETRACK_PASSWORD` are configured. CelesTrak remains
the default and needs no credentials.

For computer-independent, no-cloud-account collection, the repository includes
a scheduled GitHub Actions workflow that commits immutable, hash-manifested
bundles to a separate `data-collection` branch. Setup instructions are in
[`docs/GITHUB_DATA_COLLECTION.md`](docs/GITHUB_DATA_COLLECTION.md).

`config/experiment_60_days.json` is the authoritative source for collection,
simulation thresholds, the `iac26-leo-mixed-75-v1` cohort, and report paths.
CLI flags can override it
for explicit one-off experiments, and those values are recorded in provenance.

GitHub runs scheduled workflows only from the repository's default branch.
Therefore `.github/workflows/collect_observations.yml` must be merged into
`main` before the two-hour collector can start automatically. After merging,
enable Actions if necessary, trigger one manual smoke run, and verify that the
`data-collection` branch receives a new immutable `collections/github-run-*`
bundle. Keeping the workflow only on a feature branch does not start the
60-day experiment.

`train_from_history.py` assigns canonical object pairs deterministically with
SHA-256, then trains only on train-pair observations before a complete snapshot
cutoff and tests only on held-out-pair observations after it. Cross-quadrant
rows are excluded and counted. This simultaneously prevents future leakage,
pair memorization, and splitting one snapshot block across train and test.
The configured publication gate also requires minimum positive support across
rows, independent catalogue pairs, and snapshots; insufficient data produces a
`not_enough_data` report instead of an unstable model claim.

Before exact TCA refinement, the maintained pipeline applies a conservative
coarse screen from `space_debris.encounters`. On a separate vectorized 30-second grid,
the screen uses twice Earth-surface escape speed (about 22.36 km/s relative),
a one-kilometre numerical guard, and the actual final interval duration. Exact
TCA refinement still searches every five-minute interval. A pair
is rejected only when a 200 km encounter is impossible under this stated bound;
invalid/unbound TLE assumptions fail open to exact refinement. Screening
statistics are stored in every collection/resimulation provenance record.

## Outputs

Written to `outputs/pipeline/` (or the `--outputs` directory):

- `conjunction_dataset.csv` — every conservative-screen-retained pair after
  exact TCA refinement (screened-out count remains in provenance)
- `identified_conjunctions.csv` — exact candidates within the configured
  distance threshold (ML history uses only these)
- `model_comparison.csv` — precision/recall/F1/accuracy per model
- `top_pair_distance_timeseries.csv`, `top_pair_distance.png`
- `risk_feature_space.png`, `tca_distance_scatter.png`,
  `altitude_distance_geometry.png`, `risk_ranking.png`, `model_metrics.png`
- `risk_density_heatmap.png` (only when >= 30 candidate rows exist)
- `top_pair_eci_trajectory.png`, `top_pair_altitude_evolution.png`,
  `top_pair_encounter_plane.png`, `top_pair_relative_ric.png`

Regenerate plots from existing CSVs:

```bash
python src/generate_plots.py --outputs outputs/pipeline
```

## Tests

```bash
python -m pytest -q
```

The suite covers the distance/altitude math, the physics-based label logic, the
no-target-leakage guarantee on the ML feature set, and a propagation smoke test.

## Methodology notes for reviewers

- **Label definition.** `risk_label` is a function of the minimum approach
  distance *and* the relative velocity at TCA. It couples geometry with
  kinematics, so a distance-only fixed threshold necessarily mislabels slow
  close passes (false alarms) and fast, slightly-more-distant passes (misses).
  This is the gap the ML models are asked to close.
- **No target leakage.** `risk_score` (a monotone transform of the same
  quantities that define the label) is deliberately excluded from the model
  feature set. A test enforces this. This is why the earlier version reported a
  perfect fixed-threshold model — the task had been trivially self-referential.
- **Screening before classification.** Per the paper, models are trained on the
  *identified conjunctions*, not on all pairs, so distance alone no longer
  separates the classes within the candidate set.

## Project layout

```
src/space_debris/    maintained package (core, ml, plots)
src/run_pipeline.py  one-shot pipeline
src/make_demo_tles.py deterministic synthetic demo catalogue
src/fetch_tles.py    validated CelesTrak fetch + optional Space-Track fallback
src/collect_observations.py  repeated-snapshot collector
src/rebuild_history.py schema-safe history reconstruction
src/resimulate_snapshots.py corrected-TCA historical resimulation
src/train_from_history.py    pair-held-out chronological evaluation
src/generate_plots.py        re-render plots from CSVs
src/legacy/          original step1..step10 prototypes (reference only)
tests/               unit tests
config/              experiment configuration
docs/                gap analysis and plotting blueprint
```
