# Methodology

Supporting documentation for IAC 2026 paper **114764**, *"Machine
Learning-Based Simulation Approach for Assessing Space Debris Collision Risk
in Low Earth Orbit."* This describes the data, label definition, modeling,
and evaluation choices actually implemented in this repository, and states
the limitations explicitly rather than leaving them implicit.

## 1. Data

**Source.** Two-Line Element (TLE) sets from the CelesTrak GP API
(`celestrak.org/NORAD/elements/gp.php`), fetched by `src/fetch_tles.py`.

**Acquisition strategy.** CelesTrak's `GROUP` endpoint (e.g.
`GROUP=STARLINK`) was observed to return HTTP 403 even for small requests,
while the per-object `CATNR` endpoint remained reachable. Two changes follow
from that observation:

- A curated multi-orbit catalog, **`leo_mixed`**
  (`fetch_tles.LEO_MIXED_CATALOG`), is fetched object-by-object via `CATNR`.
  It spans distinct LEO inclinations (ISS-like ~51.6 deg, sun-synchronous
  ~98-99 deg, ~28.5 deg) so the orbital planes actually cross — unlike a
  single constellation, whose satellites share near-identical planes and
  altitudes and therefore rarely produce genuine close approaches.
- `fetch_gp_with_retry()` retries 403/429 responses with exponential
  backoff; if a `GROUP` query is blocked, `fetch_group_blocks()` falls back
  to the curated `CATNR` catalog registered for that group name
  (`GROUP_FALLBACK_CATALOGS`).

**Collection cadence.** `src/collect_observations.py` polls CelesTrak no
more often than every 2 hours (CelesTrak's own stated GP refresh interval)
and defaults to the `leo_mixed` preset. A single fetch or simulation failure
is logged and the collector proceeds to the next cycle rather than aborting
a multi-day collection run.

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
it never patches that config automatically. With only a few hours of
`leo_mixed` collection, the calibration report is correctly flagged
`reliable: false` (fewer than ~200 accumulated pairs); this is expected
behavior, not a bug, and the honest response is to wait for more data, not
to lower the reliability bar.

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

`space_debris.ml._build_models()` compares five classifiers against a fixed
distance threshold: Logistic Regression, Random Forest, SVM (RBF kernel), and
XGBoost (skipped only if the package is absent), all imbalance-aware:

- LogReg / RF / SVM use `class_weight="balanced"`.
- XGBoost uses `scale_pos_weight = n_negative / n_positive`, computed from
  the **training** split only.

This matters because real conjunction events are rare (see Limitations):
without reweighting, a classifier can achieve near-perfect accuracy by
always predicting "not risky."

## 5. Evaluation

**Splits.**

- `compare_models()` uses a single chronological 75/25 split when a time
  column is available (earlier snapshots train, later ones test) — the
  honest way to evaluate a forecasting-style classifier, avoiding the
  optimism of a random split on correlated rows.
- `time_series_cv_report()` offers k-fold `TimeSeriesSplit` cross-validation
  as an **additional option**, reporting mean +/- std per model/metric. More
  folds give a less noisy read of stability, at the cost of smaller/earlier
  training folds.

**Metrics.** PR-AUC (`average_precision_score`) and ROC-AUC lead every
report; accuracy is retained for context but is **not** the headline metric.
Under severe class imbalance, a model that always predicts "not risky" still
scores >99% accuracy while missing every real conjunction — this was
observed directly in this project (all models reporting
precision = recall = F1 = 0.0 while accuracy read ~99.99%) before PR-AUC/
ROC-AUC were added. `compare_to_baseline_pr_auc()` explicitly reports each
model's PR-AUC delta against the fixed-distance baseline.

**Publication figures** (`space_debris.plots`, 300 DPI): precision-recall
curves per model, confusion matrices per model, tree-based feature
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
- **Real conjunctions are rare.** A single mixed-orbit snapshot yields very
  few candidate conjunctions; no classifier can reliably beat the
  fixed-distance baseline on it. The paper's comparison is meant to run on
  accumulated multi-day observations, not a single snapshot.
- **Threshold calibration needs volume.** `analyze_distributions.py` flags
  its own output `reliable: false` below ~200 accumulated pairs. Early-stage
  results in this repository reflect only a few hours of `leo_mixed`
  collection and should not be read as calibrated thresholds.
- **Synthetic demo data is clearly marked.** `make_demo_tles.py` generates a
  deterministic synthetic catalogue (`DEMO-SAT-xx`) so the code path, plots,
  and model comparison can be exercised offline. It is never used as a
  stand-in for real-data results in the paper.
