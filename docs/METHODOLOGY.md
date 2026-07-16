# Methodology

Supporting documentation for IAC 2026 paper **114764**, *"Machine
Learning-Based Simulation Approach for Assessing Space Debris Collision Risk
in Low Earth Orbit."* This describes the data, label definition, modeling,
and evaluation choices actually implemented in this repository, and states
the limitations explicitly rather than leaving them implicit.

## 1. Data

**Source.** Two-Line Element (TLE) sets are fetched by `src/fetch_tles.py`.
CelesTrak's GP API (`celestrak.org/NORAD/elements/gp.php`) is the primary
provider. The authenticated Space-Track `gp` endpoint is an optional fallback
for explicit/preset NORAD catalogue lists; credentials are read only from
`SPACETRACK_IDENTITY` and `SPACETRACK_PASSWORD` environment variables.

**Acquisition strategy.** CelesTrak's `GROUP` endpoint (e.g.
`GROUP=STARLINK`) was observed to return HTTP 403 even for small requests,
while the per-object `CATNR` endpoint remained reachable. Two changes follow
from that observation:

- A curated multi-orbit catalog, **`leo_mixed`**
  (`fetch_tles.LEO_MIXED_CATALOG`), is fetched object-by-object via `CATNR`.
  It contains 75 independent NORAD IDs, including selected Iridium 33 and
  Cosmos 2251 collision fragments. Display-name duplicates remain distinct
  through catalogue-ID pair keys.
  It spans distinct LEO inclinations (ISS-like ~51.6 deg, sun-synchronous
  ~98-99 deg, ~28.5 deg) so the orbital planes actually cross — unlike a
  single constellation, whose satellites share near-identical planes and
  altitudes and therefore rarely produce genuine close approaches.
- The final experiment freezes these IDs as catalogue cohort
  `iac26-leo-mixed-75-v1`. Snapshot sidecars and history rows carry that value;
  they also carry the SHA-256 of the sorted 75-ID set. Collection rejects
  explicit/group overrides, a different object cap, or even one missing ID;
  rebuild and training verify both version and hash and fail closed on a
  different, partial, or mixed cohort. Earlier
  5/43-object snapshots are retained only as an engineering/audit archive.
- `fetch_gp_with_retry()` retries transient transport failures and HTTP
  403/408/425/429/5xx responses with capped attempts, exponential backoff,
  and jitter; if a `GROUP` query is blocked, `fetch_group_blocks()` falls back
  to the curated `CATNR` catalog registered for that group name
  (`GROUP_FALLBACK_CATALOGS`).

**Validation and commit policy.** Every accepted TLE pair must have line
numbers 1/2, identical catalogue identifiers, exactly 69 characters per line,
and valid modulo-10 checksums. HTML, oversized responses, malformed blocks,
and explicit-catalogue coverage below 90% are rejected. Snapshot and history
files are written to temporary files, flushed, and atomically replaced only
after validation. Thus a provider or pipeline failure can skip a collection
cycle, but cannot overwrite the last valid file or append shifted columns.

**Collection cadence.** `src/collect_observations.py` polls CelesTrak no
more often than every 2 hours (CelesTrak's own stated GP refresh interval)
and defaults to the `leo_mixed` preset. A single fetch or simulation failure
is logged and the collector proceeds to the next cycle rather than aborting
a multi-day collection run. In `--once` mode it instead returns a non-zero
exit status, which prevents GitHub Actions from publishing an incomplete run.

**Provenance.** Every TLE snapshot is paired with a sidecar
`<snapshot>.json` recording the source (preset/group/explicit catnrs),
fetch timestamp, object count, and git commit. Every derived CSV
(`conjunction_dataset.csv`, `identified_conjunctions.csv`,
`model_comparison*.csv`) carries a `#`-prefixed header with
`generated_utc`, `git_commit`, `source`, and `config` before the data rows
(`space_debris.provenance`); readers pass `comment="#"` to skip it. Every
PNG figure embeds the same commit hash and generation timestamp as PNG
metadata. This lets any table or figure in the paper be traced back to the
exact code version and configuration that produced it.

**Schema versioning.** The current accumulated file is
`conjunction_observations_iac26_75_v1.csv`. `src/rebuild_history.py` reconstructs it
from immutable per-run datasets and includes only the current 27-column pair
schema; legacy schemas are preserved as raw runs and listed in
`history_rebuild_report.json`, never silently coerced into shifted columns.
Historical rows involving station-attached/docked modules and vehicles are
also excluded and counted in that report; their near-zero separation from the
parent station is not an independent conjunction.

**Conservative candidate screening.** Exact all-interval TCA refinement is
performed only after a lossless-under-assumptions coarse screen. For relative
distance `d(t)` and a valid relative-speed bound `V`, the Lipschitz inequality
implies that an encounter inside an interval of duration `dt` must have at
least one endpoint within `candidate_threshold_km + V*dt/2`. The implementation
uses twice Earth-surface escape speed (about 22.36 km/s) plus a 1 km numerical
guard, includes the exact horizon remainder, and retains equality. Screening
uses a separate, vectorized 30 second grid while exact TCA refinement retains
the five-minute interval partition; at 200 km the universal screening cutoff
is therefore about 536 km. TLEs that do not satisfy the bound-Earth-orbit/perigee assumptions,
or produce non-finite samples, fail open to exact refinement. This screen
reduces cost; it never substitutes its coarse distance for the reported TCA.

**Historical resimulation.** `src/resimulate_snapshots.py` replays immutable
TLE snapshots at their original `fetched_utc` with the current experiment
config (including its deterministic `max_objects` cap) and corrected TCA
implementation. Each run is written to a temporary
directory and atomically published with input SHA-256, source provenance,
code commit, screening counts and exact-candidate counts. Completed matching
hashes are skipped, allowing interruption-safe continuation. Original run
artifacts are never overwritten.

## 2. Label Definition

**What is *not* claimed.** This framework does not compute a true
Probability of Collision (Pc). Pc requires covariance/error-ellipsoid data
and object size, neither of which is available from public TLEs. `risk_score`
is a transparent ranking aid, not a Pc, and is explicitly excluded from the
model feature set (see leakage note below).

**The proxy label.** `risk_label` (`space_debris.core._risk_label`) marks a
pair "risky" (1) only when **both**:

- the minimum approach distance is within a physical screening radius
  (`label_threshold_km`), **and**
- the relative velocity at TCA meets or exceeds a kinematic floor
  (`label_relative_velocity_km_s`).

Coupling geometry with kinematics is deliberate: a classical fixed-distance
threshold sees only distance, so it necessarily misclassifies slow, close
passes (false alarms) and fast, slightly-more-distant passes (misses). That
gap is exactly what the ML comparison is designed to expose.

**Threshold calibration.** `src/analyze_distributions.py` analyzes the
accumulated `min_distance_km` / `relative_velocity_km_s` distributions and
searches a percentile grid for thresholds that keep the positive rate in a
target range (default 2-15%). This is a **mechanism**, not a one-time
constant: it is rerun as collection history grows, and its output
(`config/threshold_calibration.json`) is a disposable, timestamped report
that a human reads before hand-editing `config/experiment_60_days.json` —
it never patches that config automatically. The distance search is capped at
the 50 km candidate-screening limit, so calibration cannot manufacture
positive labels by redefining "close approach" as hundreds or thousands of
kilometres. A report is `reliable: true` only when there are at least 200 rows
and a physically bounded threshold reaches the requested positive-rate range.
Otherwise the honest response is to expand/continue collection, not to lower
the reliability bar.

## 3. Features

`space_debris.ml.FEATURES` (13 total):

- Simulation kinematics: `time_to_tca_min`, `current_distance_km`,
  `min_distance_km`, `relative_velocity_km_s`, `altitude_difference_km`,
  `max_tle_age_hours`.
- Encounter geometry (derived, see `core._encounter_geometry_features`):
  `relative_radial_km`, `relative_intrack_km`, `relative_crosstrack_km`
  (RIC decomposition of the relative position at TCA, in the first
  satellite's local orbital frame); `relative_inclination_deg` (angle
  between the two orbits' angular-momentum directions — pure orbital-plane
  geometry, independent of how close/fast this particular encounter is);
  `radial_velocity_km_s` / `tangential_velocity_km_s` (closing-rate vs.
  perpendicular split of the relative velocity, evaluated at **snapshot**
  time, since at TCA the closing rate is ~0 by construction);
  `approach_angle_deg` (angle between the current relative-position and
  relative-velocity vectors).

**No target leakage.** `risk_score`, `risk_label`, and `fixed_threshold_alarm`
are excluded from `FEATURES` (enforced by
`tests/test_core.py::test_no_target_leakage_in_feature_set`). `risk_score` is
a monotone transform of the same distance/velocity quantities used to build
the label and would let a learner trivially recover it. The RIC/orbital-plane
features are geometric decompositions of the position/velocity *vectors* —
they describe orientation, not magnitude, and several are evaluated at a
different time (snapshot vs. TCA) than the label-defining scalars — so none
of them is a duplicate or monotone transform of `min_distance_km` /
`relative_velocity_km_s`.

## 4. Models

`space_debris.ml._build_models()` compares six classifiers against a fixed
distance threshold. Logistic Regression and SVM (RBF kernel) are retained as
weak baselines. A single Decision Tree provides the direct reference needed to
justify the variance reduction from Random Forest. Random Forest, XGBoost, and
LightGBM are the primary candidates; either boosting implementation is reported
as unavailable rather than aborting a run if its optional native package cannot
be imported. All learners are imbalance-aware:

- LogReg / Decision Tree / RF / SVM / LightGBM use
  `class_weight="balanced"`.
- XGBoost uses `scale_pos_weight = n_negative / n_positive`, computed from
  the **training** split only.

This ordering follows the adviser-requested comparison: the single tree tests
whether ensembling is worthwhile, while the two boosting families provide
independent high-capacity alternatives to Random Forest. No oversampling is
performed at this stage; SMOTE remains deferred until enough genuine positive
observations exist to synthesize from.

The training entry point also writes a separate
`*_without_label_rule_features.csv` ablation on the exact same pair/time split.
That supporting arm removes `min_distance_km` and
`relative_velocity_km_s`—the two quantities used to construct the proxy
label—and all three TCA RIC position components, whose joint norm would
otherwise reconstruct minimum distance. It leaves the canonical real-data
result unchanged. The performance gap between the two reports quantifies how
much of the apparent skill is direct proxy-rule recovery.

This matters because real conjunction events are rare (see Limitations):
without reweighting, a classifier can achieve near-perfect accuracy by
always predicting "not risky."

## 5. Evaluation

**Splits.**

- `compare_models()` assigns canonical NORAD-catalog pairs (`A|B == B|A`) to stable
  train/test groups with a salted SHA-256 mapping. A cutoff is placed only
  between complete snapshot blocks. Training is `train pair + past`; testing
  is `held-out pair + future`; the other two quadrants are excluded and
  counted. This guarantees pair disjointness and strict time ordering.
  Human-readable names remain reporting labels; catalogue IDs prevent many
  debris fragments sharing one display name from collapsing into one pair.
- `time_series_cv_report()` applies the same fixed pair assignment in
  expanding, whole-snapshot time folds and reports mean +/- std together with
  total and metrically valid fold counts. A single-class test fold cannot
  contribute PR-AUC/ROC-AUC and is never used to claim model superiority.

**Publication quality gate.** The authoritative experiment config requires at
least 30 real positive training rows and 20 real positive test rows, spanning
at least 10/5 independent positive catalogue pairs and 5/3 positive snapshots
in train/test respectively. It also requires an observation span of at least
`duration_days - poll_interval_hours / 24` (59.9167 days for the 60-day,
two-hour cadence), so a high-yield catalogue cannot produce an early paper
result. These support minimums are frozen before the final
60-day held-out evaluation; they are not a formal external preregistration or a
guarantee of statistical significance. If any minimum fails,
`train_from_history.py` writes a `not_enough_data` report and produces no model
or publication claim. The current short archive is expected to fail this gate;
the 60-day collection must continue.

**Metrics.** PR-AUC (`average_precision_score`) and ROC-AUC lead every
report; accuracy is retained for context but is **not** the headline metric.
Under severe class imbalance, a model that always predicts "not risky" still
scores >99% accuracy while missing every real conjunction — this was
observed directly in this project (all models reporting
precision = recall = F1 = 0.0 while accuracy read ~99.99%) before PR-AUC/
ROC-AUC were added. For the distance-only baseline, the configured 25 km alarm
is the binary operating point used for precision/recall/F1 and confusion
counts, while continuous `-min_distance_km` is used for PR-AUC/ROC-AUC. This
avoids treating a two-level alarm as if it were a full ranking.
`compare_to_baseline_pr_auc()` explicitly reports each model's PR-AUC delta
against that continuous distance ranking. The report also records false alarm
rate as `FP / (FP + TN)` for every model and the fixed 25 km operating point.
Any claim of fewer false alarms must be read together with recall; a lower
false alarm rate obtained by detecting fewer positives is not an improvement.

**Publication figures** (`space_debris.plots`, 300 DPI): precision-recall
curves per model, false-alarm-rate versus recall curves over all score
thresholds, confusion matrices per model, tree-based feature
importance (Random Forest / XGBoost), and a threshold-sensitivity sweep
showing how the fixed-distance baseline's precision/recall/F1 change as its
single operating point is varied — illustrating that the chosen
`fixed_threshold_km` is one point on a curve, not a uniquely correct value.

## 6. Reproducibility

- **Tests** (`python -m pytest -q`) cover distance/altitude math, the
  physics-based label logic, the no-leakage guarantee, encounter-geometry
  correctness (hand-verified vectors: RIC decomposition, perpendicular-plane
  inclination, radial/tangential sign conventions), retry/fallback behavior
  for TLE fetches, collector resilience, distribution-calibration logic, and
  the imbalance-aware modeling pipeline.
- **CI** (`.github/workflows/tests.yml`) runs the full test suite on every
  push across two Python versions.
- **Provenance** is embedded end-to-end as described in Section 1: every
  output CSV/figure traces to a git commit, generation timestamp, data
  source, and configuration summary.

## 7. Limitations

- **The present proxy task is rule recovery, not physical collision
  prediction.** `risk_label` is deterministically constructed from miss
  distance and relative velocity, and those physical quantities are available
  to the classifiers. Results therefore measure whether models reproduce this
  transparent screening rule better than a distance-only baseline; they do not
  establish real-world collision probability or causal predictive skill.
- **`risk_label` is a proxy, not a validated ground truth.** No public-TLE
  source provides confirmed historical close-approach outcomes at this
  granularity; the label is a physically-motivated geometric/kinematic rule,
  not an operational conjunction-assessment product.
- **No Pc, no covariance.** Object size and covariance are unavailable from
  public TLEs, so this framework cannot and does not claim a true
  Probability of Collision.
- **SGP4 accuracy degrades with TLE age.** `max_tle_age_hours` triggers a
  data-quality warning (default 14 days); stale elements make TCA and
  minimum-distance figures illustrative rather than operational.
- **Classic TLE catalogue-number ceiling.** The fixed-width TLE representation
  cannot represent all newly assigned catalogue numbers above 99999. The
  present paper catalogue remains below that boundary; expanding beyond it
  requires an OMM ingestion path (preferably XML) rather than truncating or
  inventing identifiers.
- **Real conjunctions are rare.** A single mixed-orbit snapshot yields very
  few candidate conjunctions; no classifier can reliably beat the
  fixed-distance baseline on it. The paper's comparison is meant to run on
  accumulated multi-day observations, not a single snapshot.
- **Threshold calibration needs useful encounters, not merely row volume.**
  `analyze_distributions.py` flags its output `reliable: false` below 200 rows
  or when no threshold at or below 50 km reaches the requested positive-rate
  range. Early-stage `leo_mixed` results should not be read as calibrated
  merely because they contain many distant pairs.
- **Synthetic demo data is clearly marked.** `make_demo_tles.py` generates a
  deterministic synthetic catalogue (`DEMO-SAT-xx`) so the code path, plots,
  and model comparison can be exercised offline. It is never used as a
  stand-in for real-data results in the paper.
