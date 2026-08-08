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

`risk_label` (the ML target) is itself a deterministic function of
`min_distance_km` and `relative_velocity_km_s` (see `space_debris/core.py`), not
an independent ground truth. Two features sets in `space_debris/ml.py` make
this explicit rather than accidental:

- `snapshot_only` (**the predictive-power claim**, and the default everywhere
  in this repo): predictors available before the TCA search runs. This is the
  headline "ML vs. fixed-distance baseline" comparison.
- `full_rule_recovery`: additionally gives the model the two features that
  *define* `risk_label`. This measures how well a model recovers a known,
  transparent rule from raw physical inputs — a sanity/diagnostic control, not
  a predictive-power result. `train_from_history.py` always runs and reports
  this separately (suffixed `_full_rule_recovery.csv`), never blended into the
  primary claim.

`run_pipeline.py` defaults to `snapshot_only` for this reason; pass
`--feature-set full_rule_recovery` only if you intend to reproduce the
rule-recovery diagnostic, not the predictive-power result.

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

These commands run with `--feature-set snapshot_only` by default — the
predictive-power comparison described above. A single ad hoc run does **not**
reproduce the active 10-day frozen-cohort protocol or its chronological
train/test split; use these commands to exercise the pipeline against live
data, not to reproduce the paper's headline result.

## Active claim-eligible experiment (15-day v4)

The prospective frozen protocol is `iac26-15d-v4`, running from
2026-08-10T00:17:00Z through 2026-08-25T00:17:00Z. It contains 120 half-open
three-hour slots, requires 90% coverage, and stores schema-3 bundles under
`experiments/iac26-15d-v4/collections/`. Exact experiment/config/catalogue and
simulation bindings are verified before import. The scientific slot width is
separate from the unchanged two-hour provider-request floor, adding scheduler
slack without increasing request frequency or lowering any quality gate. v1,
v2 and v3 remain separate immutable pilot/audit archives and are never pooled
into v4. See
[`docs/EXPERIMENT_15D_V4.md`](docs/EXPERIMENT_15D_V4.md).

A single snapshot yields very few conjunctions, so no classifier can reliably
beat the fixed-distance baseline on it — that is a statistical fact, not a code
bug. The paper's comparison is meant to run on **accumulated observations**.
CelesTrak GP data should not be polled more often than every 2 hours; the
collector enforces this as a floor.

```bash
# one cycle
python src/collect_observations.py --once

# one active-v4 cycle (normally GitHub Actions performs this)
python src/collect_observations.py --once --config config/experiment_15_days_v4.json

# verify/import the GitHub archive, then rebuild corrected-TCA history
git fetch origin data-collection
git worktree add ../space-debris-data origin/data-collection
python src/import_collection_archive.py ../space-debris-data --config config/experiment_15_days_v4.json
python src/resimulate_snapshots.py --config config/experiment_15_days_v4.json --workers 4

# pair-held-out, time-ordered evaluation over corrected-TCA history
python src/train_from_history.py --config config/experiment_15_days_v4.json
```

After the frozen window closes, the same fail-closed sequence is available as
one resumable PowerShell command:

```powershell
.\scripts\finalize_iac_experiment.ps1 -ArchiveRoot ..\space-debris-data -Workers 4
```

The finalizer refuses to run before the configured end time, requires a clean
committed archive through the importer, reuses fingerprint-verified completed
resimulations, and produces no publication claim unless every downstream
quality and evidence gate passes.

The active training command uses the corrected frozen-cohort
`conjunction_observations_resimulated_iac26_75_v4.csv`, not the live collector
history or the older mixed 5/43/75-object archive. The claim-eligible primary
experiment is pre-specified XGBoost using only six quantities available at the
observation snapshot: current distance, altitude difference, maximum TLE age,
radial velocity, tangential velocity and approach angle. TCA, minimum-distance,
TCA-relative-velocity and TCA-RIC quantities are excluded from that primary
model because they define or reconstruct the future proxy target.

The run also writes snapshot geometry/kinematics ablations, a snapshot-state
constant-velocity CPA baseline, an exact proxy-rule oracle, a deterministic
train-label-permutation negative control, and distance-only, transparent
proxy-rule and full 13-feature rule-recovery arms on the identical held-out pair/time split. These are
non-claim-eligible controls: in particular, strong performance by the full arm
shows how easily the deterministic proxy rule can be reconstructed, while
unexpectedly strong permutation performance warns of leakage or evaluation
error.

The same run writes a SHA-256-bound split manifest, row/pair-identifiable
held-out prediction CSV, machine-readable feature importances and 2,000-draw
separate paired pair-cluster and UTC-day-block bootstrap sensitivity intervals. Model operating thresholds are selected
only inside the outer training partition at the fixed baseline's validation
recall. The same inner-validation partition also freezes a stronger
distance-only comparator; held-out labels never tune either threshold. A lower
proxy false-alarm claim remains fail-closed unless it beats both the untouched
25 km alarm and this calibrated distance comparator and its 95%
interval excludes zero while recall is non-inferior within the configured 0.05
margin under both resampling units. At least five held-out UTC day blocks are
required. Publication plots consume these frozen artifacts and do not refit a
second copy of the models.

For each learner, the evidence table also reports model-minus-linear-CPA
metric differences and pair/day bootstrap intervals. This same-snapshot-
information comparison is a sensitivity analysis, not a second primary claim.
The pair and day resamples are deliberately reported as two conditional,
marginal checks rather than a multiway population-confidence interval; they do
not eliminate shared-object graph dependence.

The separate statistical adaptability artifact compares frozen snapshot-only
XGBoost with the calibrated continuous-distance comparator on non-overlapping
future blocks. It uses a dyadic catalogue-object x time-block bootstrap and
requires ten eligible blocks, 30 objects, 30 pairs, a positive one-sided 95%
lower bound, p < 0.05, and positive AP advantage in every block. Expanding-time
CV remains descriptive; insufficient support keeps the abstract claim closed.
Blocks start one polling interval after the outer cutoff. Rows in that initial
interval remain valid held-out predictions but are excluded as a pre-registered
embargo; rows before the cutoff are rejected. Partial final cycles are excluded.
Publication creation recomputes the inference after recursively verifying the
canonical history/config/split/prediction/evidence graph for the main run and
all six ablations, including canonical split digests and source hashes embedded
in supporting CSV/PNG artifacts.

If an older checkout already accumulated history, rebuild the schema-safe
version from immutable run outputs before training:

```bash
python src/rebuild_history.py
```

TLE input is fail-closed: malformed/checksum-invalid responses are always
rejected and writes are atomic. The 90% coverage floor applies to `--catnr`
fetches, where the requested catalogue IDs are known in advance so a
received/requested ratio is meaningful; a bare `--group` fetch has no fixed
target to compare against; and instead is only guaranteed to reject a
zero-object result (see `fetch_tle_blocks`/`write_tle_file` in
`src/fetch_tles.py`). The frozen IAC protocol always fetches via the `--catnr`
`leo_mixed` preset, so the 90% floor applies to every claim-eligible run.
`--provider auto` can use authenticated Space-Track GP as a fallback when
`SPACETRACK_IDENTITY`/`SPACETRACK_PASSWORD` are configured. CelesTrak remains
the default and needs no credentials.

For computer-independent, no-cloud-account collection, the repository includes
a scheduled GitHub Actions workflow that commits immutable, hash-manifested
bundles to a separate `data-collection` branch. Setup instructions are in
[`docs/GITHUB_DATA_COLLECTION.md`](docs/GITHUB_DATA_COLLECTION.md).

The companion cadence-health workflow verifies the newest schema-3
bundle manifest every three hours and fails after the frozen six-hour gap limit.
This is an operational alert only; publication coverage continues to use
snapshot timestamps and the frozen 120-slot definition below.

The final collection interval is frozen in the experiment config and anchored
to its pre-specified `00:17Z` start. GitHub is asked to run at `:07`, `:17`,
`:37` and `:47` every hour, providing twelve delivery opportunities inside
each three-hour scientific slot. The archive-backed slot guard permits at most
one actual CelesTrak poll per slot and retains a separate two-hour provider
request floor. Coverage is measured over 120 half-open three-hour bins using
each bundle's recorded `snapshot_utc`; retries
cannot inflate it, at least 90% of slots must be present, the actual timestamp
gap (including window endpoints) may not exceed six hours, at least 30% of
occupied bins must have distinct TLE hashes, and an identical-hash run may not
exceed six bins. The diversity threshold was frozen from the v1 pilot before
v2 collection and is not outcome-tuned.

`config/experiment_15_days_v4.json` is authoritative for active collection,
simulation thresholds, the `iac26-leo-mixed-75-v2` cohort, and report paths.
`config/experiment_10_days_v3.json`, `config/experiment_10_days_v2.json`
and `experiment_60_days.json` remain immutable for v3/v2/v1 audit replay.
Exploratory tools may accept alternate configs, but the claim-eligible
`train_from_history.py` rejects every unregistered config path. Changing
the frozen protocol requires a new versioned experiment and collection window;
it cannot be overridden after outcomes are observed.
for explicit one-off experiments, and those values are recorded in provenance.

GitHub runs scheduled workflows only from the repository's default branch.
Therefore `.github/workflows/collect_observations.yml` must be merged into
`main` before the v4 collector can start automatically. After merging,
enable Actions if necessary, trigger one manual smoke run, and verify that the
`data-collection` branch receives a new immutable
`experiments/iac26-15d-v4/collections/github-run-*` bundle. Keeping the
workflow only on a feature branch does not start the experiment.

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

- **Label definition.** `risk_label` is a deterministic future-propagation
  proxy based on minimum approach distance and relative velocity at TCA. Its
  disagreement with a 25 km distance alarm is expected by construction; that
  disagreement alone is not evidence of adaptability or operational safety.
- **Target-definition protection.** The claim-eligible primary feature set
  excludes `risk_score`, both label-defining variables, TCA/time-to-TCA fields,
  the three TCA-RIC coordinates that reconstruct miss distance, and the current
  implementation of relative inclination because it is evaluated at TCA. A
  full-feature rule-recovery model remains only as a non-claim-eligible positive
  control.
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
