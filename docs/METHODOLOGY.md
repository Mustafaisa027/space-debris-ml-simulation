# Methodology

Supporting documentation for IAC 2026 paper **114764**, *"Machine
Learning-Based Simulation Approach for Assessing Space Debris Collision Risk
in Low Earth Orbit."* This describes the data, label definition, modeling,
and evaluation choices actually implemented in this repository, and states
the limitations explicitly rather than leaving them implicit.

## Active experiment version

The claim-eligible acquisition protocol is `iac26-10d-v2`, frozen in
`config/experiment_10_days_v2.json` for 2026-07-24T00:17:00Z through
2026-08-03T00:17:00Z. Schema-3 bundles bind the experiment ID, canonical
configuration hash, embedded configuration, catalogue hash, simulation
parameters and runtime. They are isolated under
`experiments/iac26-10d-v2/collections/`.

References below to the 60-day `iac26-leo-mixed-75-v1` protocol document the
preserved pilot/audit design. V1 cannot satisfy its frozen 90% gate under the
hardened legacy-bundle policy and is not pooled into v2. The model, label,
feature, split and fail-closed inference methods remain unchanged. The
version-specific acquisition rationale is in `docs/EXPERIMENT_10D_V2.md`.

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

Coupling geometry with kinematics is deliberate, but it also means a learner
given both label-defining quantities can recover the rule by construction. The
fixed-distance reference sees only distance and is therefore expected to
disagree with this proxy. That disagreement is a transparent reference, not by
itself evidence of general collision-risk adaptability.

**Frozen label protocol.** `src/analyze_distributions.py` analyzes the
accumulated `min_distance_km` / `relative_velocity_km_s` distributions and
searches a percentile grid for thresholds that keep the positive rate in a
target range (default 2-15%) for separately versioned exploratory follow-up
experiments. The canonical 50 km / 10 km/s definition was present before the
frozen collection began and must not change after outcomes are inspected. A
different label definition requires a new experiment/config version and a new
frozen window. Calibration output
(`config/threshold_calibration.json`) is a disposable, timestamped report
is never allowed to patch the canonical `config/experiment_60_days.json`
automatically. The distance search is capped at
the 50 km proxy-label physical upper bound (the conservative candidate screen
itself is 200 km), so calibration cannot manufacture
positive labels by redefining "close approach" as hundreds or thousands of
kilometres. A report is `reliable: true` only when there are at least 200 rows
and a physically bounded threshold reaches the requested positive-rate range.
Otherwise the honest response is to expand/continue collection, not to lower
the reliability bar.

## 3. Features

The claim-eligible primary set is
`space_debris.ml.SNAPSHOT_ONLY_FEATURES`: `current_distance_km`,
`altitude_difference_km`, `max_tle_age_hours`, `radial_velocity_km_s`,
`tangential_velocity_km_s`, and `approach_angle_deg`. All six are available at
the observation snapshot before the forward TCA result is evaluated.

`space_debris.ml.FEATURES` is the 13-feature future-propagation/rule-recovery
upper-bound set:

- Simulation kinematics: `time_to_tca_min`, `current_distance_km`,
  `min_distance_km`, `relative_velocity_km_s`, `altitude_difference_km`,
  `max_tle_age_hours`.
- Encounter geometry (derived, see `core._encounter_geometry_features`):
  `relative_radial_km`, `relative_intrack_km`, `relative_crosstrack_km`
  (RIC decomposition of the relative position at TCA, in the first
  satellite's local orbital frame); `relative_inclination_deg` (currently
  evaluated from TCA position/velocity vectors and therefore not admitted to
  the snapshot-only primary set);
  `radial_velocity_km_s` / `tangential_velocity_km_s` (closing-rate vs.
  perpendicular split of the relative velocity, evaluated at **snapshot**
  time, since at TCA the closing rate is ~0 by construction);
  `approach_angle_deg` (angle between the current relative-position and
  relative-velocity vectors).

**Target-definition protection.** `risk_score`, `risk_label`, and
`fixed_threshold_alarm` are excluded from every model set. More importantly,
the claim-eligible snapshot-only set excludes `min_distance_km`, TCA relative
velocity, `time_to_tca_min`, all TCA RIC components, and the currently
TCA-evaluated inclination. `risk_score` is
a monotone transform of the same distance/velocity quantities used to build
the label and would let a learner trivially recover it. The RIC/orbital-plane
features are geometric decompositions of the position/velocity *vectors*.
The joint norm of the three TCA RIC position components exactly reconstructs
`min_distance_km`; this is why the full set is an explicitly non-claim-eligible
positive control rather than the publication primary.

## 4. Models

`space_debris.ml._build_models()` compares six classifiers against a fixed
distance threshold. Logistic Regression and SVM (RBF kernel) are retained as
weak baselines. A single Decision Tree provides the direct reference needed to
justify the variance reduction from Random Forest. Random Forest, XGBoost, and
LightGBM are compared, while the single pre-specified claim-eligible primary is
XGBoost trained on the six snapshot-only features. Either boosting
implementation is reported as unavailable rather than aborting a run if its optional native package cannot
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

The training entry point writes a `*_full_rule_recovery.csv` positive-control
arm containing all 13 future-propagation features, plus a separate
`*_without_label_rule_features.csv` ablation on the exact same pair/time split.
That supporting arm removes `min_distance_km` and
`relative_velocity_km_s`—the two quantities used to construct the proxy
label—and all three TCA RIC position components, whose joint norm would
otherwise reconstruct minimum distance. These arms quantify direct proxy-rule
recovery but cannot support the abstract claim.

Four additional staged arms use the identical frozen split: snapshot geometry
only, snapshot kinematics only, distance only
(`min_distance_km`) and the complete transparent proxy rule
(`min_distance_km + relative_velocity_km_s`). Together with the strict
geometry/kinematics ablation and full model, these distinguish distance
ranking, direct rule recovery, residual geometric information and the full
feature result.
Every supporting arm uses the same nested train/validation/test protocol and
writes its own hash-bound predictions and uncertainty artefacts. They remain
`claim_eligible=false`: their ranking and calibrated operating-point metrics
are sensitivity analyses. Only the pre-specified snapshot-only XGBoost arm is
eligible for the conditional proxy false-alarm comparison.

The canonical evidence artifact also includes four non-claim-eligible diagnostic
controls. `inner_calibrated_distance_threshold` freezes, using only the
pair-held-out inner-validation partition, the strongest continuous-distance
threshold that reaches the pre-specified 25 km alarm's validation recall. It
then applies that frozen threshold once to the outer test. This deliberately
privileged comparator directly uses the propagated minimum-distance component
of the deterministic proxy label; it is not independent ground truth. Its
purpose is to test whether threshold tuning alone explains an apparent
false-alarm advantage. A publication false-alarm claim must pass against both
this comparator and the untouched 25 km reference.

`constant_velocity_cpa` analytically projects the snapshot relative
position/velocity under linear motion and applies the same 25 km scale; it is a
same-information physics baseline and never consumes SGP4 TCA fields.
Every learner is also compared directly with this baseline using the same
paired pair-cluster and UTC-day-block bootstrap draws. These model-minus-CPA
intervals are reported as a same-information sensitivity analysis and do not
replace the pre-specified fixed-threshold primary comparator.
`label_permutation_control` deterministically shuffles only inner-training
labels before fitting a clone of the primary estimator. Unexpectedly strong
performance from that negative control is a leakage/evaluation warning, not a
result to interpret. `proxy_rule_oracle` applies the exact deterministic label
definition and must reproduce every label; its expected perfect score exposes
target construction and is not a model-performance result.

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

The fold-level rows are also preserved. Descriptive temporal consistency is
pre-defined as the primary XGBoost model's PR-AUC advantage over the continuous
distance baseline remaining positive in every one of five future test blocks,
each of which must pass the configured positive-support minimums. The blocks
reuse the same pair cohort and overlapping expanding training data, so no
independence-based sign-flip p-value or statistical-significance claim is made.
Supporting models remain exploratory.

**Frozen collection window and publication quality gate.** The authoritative
experiment is the half-open interval `2026-07-16T16:17:00Z` through
`2026-09-14T16:17:00Z`, anchored to the pre-specified `17 */2` GitHub cron
rather than the delayed first fetch. Its 60 days at two-hour cadence define 720
half-open snapshot-time bins. A delayed run remains in the bin where the
observation occurred; retries falling in one bin count once. At least 90% of bins must be
occupied and no leading, internal or trailing gap between actual snapshot
timestamps and the window endpoints may exceed six hours. At least 50% of
occupied bins must contain distinct TLE-file hashes and no identical-hash run
may exceed 12 bins. Rows outside the interval are excluded before the split. This prevents
a sparse archive, duplicate retries or post-window observations from making a
nominal 60-day dataset appear complete.

The authoritative experiment config also requires at
least 30 real positive training rows and 20 real positive test rows, spanning
at least 10/5 independent positive catalogue pairs and 5/3 positive snapshots
in train/test respectively. These support and cadence minimums are frozen
before the final
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

**Comparable-recall operating point and uncertainty.** The outer held-out test
labels never select a model threshold. The outer training partition is split
again into earlier inner-train pairs and future, pair-held-out inner-validation
pairs. The fixed 25 km alarm's inner-validation recall becomes the frozen
target; for each learner, the highest score threshold reaching that recall
with the lowest attainable validation false-alarm rate is selected. The exact
same rule calibrates a distance-only threshold from inner-validation
`-min_distance_km`; its held-out threshold is never optimized on the outer
test. The exact
same fitted inner-train estimator is then evaluated once on the untouched outer
test; it is not refit after threshold selection, because doing so could shift
SVM/boosting score scales and invalidate the frozen threshold.

The exact per-row results are written to `held_out_predictions.csv` with stable
row IDs, canonical pair IDs, scores, frozen thresholds and the split hash.
Conditional sensitivity is computed by 2,000 paired bootstrap draws under two
separate resampling units: complete canonical pair clusters and UTC calendar-day blocks. The report
includes 95% percentile intervals for model-minus-baseline precision, recall,
F1, PR-AUC and proxy false-alarm rate. A reduced false-alarm claim is permitted
only when at least five held-out day blocks exist and, under both resampling
units, at least 90% of replicates are valid, the false-alarm delta's upper bound
is below zero, and the recall delta's lower bound is at least -0.05. The primary
model must satisfy these pair/day gates against both the frozen 25 km alarm and
the inner-calibrated distance-only comparator. "False alarm" throughout this
comparison means disagreement with the deterministic proxy label, not an
operational conjunction warning or physical Pc.
These are two marginal sensitivity analyses whose gates are combined; they are
not a multiway bootstrap. Day-block resampling assumes exchangeable held-out
days, and neither analysis fully captures shared-object graph dependence.

**Statistical adaptability protocol.** The five expanding-time CV folds remain
descriptive because their training sets overlap. Statistical adaptability is
tested separately on the frozen outer future predictions. The 60-day window is
pre-split 50/50 so that the future half can contain at least ten non-overlapping
48-hour blocks separated by a two-hour embargo. The block grid is anchored to
the frozen outer cutoff plus one polling interval, not to the first available
candidate row. Held-out rows from the cutoff up to that anchor are retained in
the prediction artifact but excluded as the pre-registered initial embargo;
only rows before the cutoff are rejected as split violations. Partial final cycles are excluded and every complete planned
block must meet the support gate. The estimand is the paired
average-precision difference between snapshot-only XGBoost and the
inner-calibrated continuous-distance comparator. Each block must contain both
classes, at least five positive canonical pairs, and positives on at least two
UTC days; the full analysis additionally requires ten valid blocks, 30 objects
and 30 pairs. A 10,000-draw dyadic object x future-block bootstrap resamples
catalogue endpoints and time blocks jointly with identical row weights for
both methods. The one-sided claim gate requires at least 90% valid replicates,
a 95% lower bound above zero, a centered-bootstrap p-value below 0.05, and a
positive AP difference in every valid future block. Otherwise the hash-bound
adaptability artifact reports `not_enough_data` or `not_supported` and the
overall abstract claim remains false. Publication generation revalidates the
canonical-history/config/split/prediction/evidence hash graph, recomputes the
split digest from its canonical payload, recursively verifies all six ablation
bundles and checks embedded source hashes in supporting CSV/PNG artifacts, then
deterministically recomputes this inference before accepting its result.

**Publication figures** (`space_debris.plots`, 300 DPI): precision-recall
curves per model, false-alarm-rate versus recall curves over all score
thresholds, confusion matrices per model, tree-based feature
importance from the same fitted inner-train models, and a threshold-sensitivity sweep
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
- **Numerical environment** is pinned in `requirements.txt` and
  `requirements-collector.txt`. Every scheduled bundle additionally records
  the exact Python/platform/package inventory in `environment/runtime.json`,
  which is hash-bound by manifest schema 2.
- **Provenance** is embedded end-to-end as described in Section 1: every
  output CSV/figure traces to a git commit, generation timestamp, data
  source, and configuration summary. `evaluation_split_manifest.json` freezes
  train/test/excluded row and pair identities plus dataset/config hashes;
  `evaluation_manifest.json` binds those hashes to exact model parameters,
  package versions, thresholds, predictions, evidence table and feature
  importance. It also records a validated one-to-one archive-import to
  resimulation binding by collection ID and TLE SHA-256. Publication plots consume these frozen machine-readable
  artifacts rather than fitting a second model.

## 7. Limitations

- **The primary task is candidate-conditioned proxy-risk triage, not physical
  collision prediction.** `risk_label` is deterministically constructed from
  future-propagated miss distance and relative velocity. The claim-eligible
  model sees only snapshot-state predictors; the full rule-recovery arm is a
  non-claim-eligible positive control. Neither establishes real-world collision
  probability or causal predictive skill.
- **Selection scope.** Metrics are conditional on pairs that passed the
  conservative SGP4 candidate screen (`min_distance_km <= 200 km`). They are not
  catalogue-wide encounter-detection recall or an operational alarm rate.
- **Conditional uncertainty.** Pair-cluster and UTC-day-block bootstrap
  intervals condition on the observed catalogue and snapshot process. They
  address repeated pairs and shared-day state, but do not fully model graph
  dependence created when different pairs share the same orbital object, nor
  uncertainty from choosing a different
  catalogue. They must not be read as population-wide confidence bounds.
- **Adaptability has a narrower inferential meaning.** It denotes maintenance
  of a paired ranking advantage across frozen future blocks, not autonomous
  online learning, operational adaptation, or population-wide collision-risk
  superiority. The dyadic/time bootstrap remains conditional on this frozen
  75-object catalogue and observed collection window.
- **`risk_label` is a proxy, not a validated ground truth.** No public-TLE
  source provides confirmed historical close-approach outcomes at this
  granularity; the label is a physically-motivated geometric/kinematic rule,
  not an operational conjunction-assessment product.
- **No Pc, no covariance.** Object size and covariance are unavailable from
  public TLEs, so this framework cannot and does not claim a true
  Probability of Collision.
- **SGP4 accuracy degrades with TLE age.** Collection still emits an immediate
  warning, and publication evaluation additionally fails closed if any retained
  row exceeds the frozen 14-day `max_tle_age_hours` bound. Snapshot quality also
  reports the number and fraction of unique TLE-input hashes; repeated bytes in
  different time bins are not presented as independent element-set updates.
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
