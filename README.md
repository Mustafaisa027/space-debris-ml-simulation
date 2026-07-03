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

# time-ordered train/test evaluation over the accumulated history
python src/train_from_history.py
```

`train_from_history.py` uses a chronological split (train on earlier snapshots,
test on later ones), which is the honest way to evaluate a forecasting-style
classifier and avoids the optimism of a random split on correlated rows.

## Outputs

Written to `outputs/pipeline/` (or the `--outputs` directory):

- `conjunction_dataset.csv` — every simulated pair
- `identified_conjunctions.csv` — screened candidates (ML trains on these)
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
src/fetch_tles.py    CelesTrak TLE fetch
src/collect_observations.py  repeated-snapshot collector
src/train_from_history.py    time-split evaluation
src/generate_plots.py        re-render plots from CSVs
src/legacy/          original step1..step10 prototypes (reference only)
tests/               unit tests
config/              experiment configuration
docs/                gap analysis and plotting blueprint
```
