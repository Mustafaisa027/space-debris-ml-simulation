# IAC26 10-day v2 frozen protocol

## Status and scientific boundary

`iac26-10d-v2` is the active claim-eligible experiment. Its half-open
collection window is 2026-07-24T00:17:00Z to 2026-08-03T00:17:00Z, with one
nominal observation every two hours (120 slots). The accepted IAC abstract
specifies TLE/SGP4 conjunction analysis, the four classifiers, and
precision/recall/F1 comparison; it does not prescribe a 60-day duration.

The previous `iac26-leo-mixed-75-v1` run is retained as pilot/audit data.
Only three exact legacy bundles are eligible under the hardened importer, so
even perfect future collection could reach at most 638/720 slots (88.61%),
below its frozen 90% requirement. Lowering that gate after observing the data
would be outcome-dependent. V1 is therefore not repaired, pooled, or presented
as confirmatory evidence.

## Frozen acquisition contract

- Catalogue: the same frozen 75 NORAD IDs, versioned as
  `iac26-leo-mixed-75-v2`; exact sorted-ID SHA-256 is required.
- Source: CelesTrak CATNR requests, with credentialed Space-Track fallback
  when secrets are available. Partial catalogue responses fail closed.
- Cadence: GitHub delivery attempts at `:17` and `:47` every UTC hour, giving
  four opportunities inside each frozen two-hour scientific slot. A
  deterministic archive-backed slot guard makes every attempt after the first
  successful poll in that slot a no-op and requires two elapsed hours from the
  latest archived fetch before another provider request. This prevents adjacent
  slot-boundary requests from violating CelesTrak's one-download-per-update
  policy. This operational redundancy was
  frozen before collection after preflight runs showed GitHub cron delivery
  delays exceeding 60 minutes; it does not change slot assignment or sample
  weighting.
- Archive: schema-3 immutable bundles under
  `experiments/iac26-10d-v2/collections/`.
- Binding: each bundle records and verifies the experiment ID, canonical
  config SHA-256, embedded config, catalogue hash, runtime, and all simulation
  parameters. Duplicate physical snapshot slots are rejected.
- Line endings: the data branch records archive paths as `-text`, preserving
  byte-level manifest hashes on Windows and Linux.

## Frozen quality and evaluation contract

- At least 108/120 slots (90%) and no endpoint-inclusive gap above six hours.
- At least 30% unique TLE hashes across occupied slots and no identical-hash
  run above six bins. The 30% floor was chosen before v2 from the v1 pilot,
  where 25/59 snapshots were distinct (42.4%); it detects a materially stale
  feed without demanding a new element set on every two-hour poll.
- Proxy label remains minimum distance <= 50 km and TCA relative velocity
  >= 10 km/s. It is not Probability of Collision.
- Primary learner remains snapshot-only XGBoost; Logistic Regression, Random
  Forest, SVM, and XGBoost run on the same deterministic pair-held-out,
  chronological split.
- The 25 km alarm and train-only calibrated distance comparator remain the
  baselines. Reduced proxy false alarms require recall non-inferiority and
  both pre-specified uncertainty gates.
- Adaptability remains fail-closed unless all configured support, block,
  bootstrap, confidence-bound, p-value, and every-block direction gates pass.
- The v2 outer split is frozen at 40% training time so the remaining future
  interval can contain exactly ten complete 12-hour analysis blocks separated
  by two-hour embargoes. Each eligible block requires both classes, at least
  five positive pairs and positive support on its UTC day. The registered
  bootstrap seed is 214764. The validator is experiment-versioned so legacy
  v1's 50%/48-hour/114764 contract cannot silently override v2.
- If class or temporal support is insufficient after ten days, the valid
  result is `not_enough_data`; thresholds and support gates are not relaxed.

No numerical abstract claim is released until archive verification,
corrected-TCA resimulation, model evaluation, and publication validation all
pass. If evidence contradicts the accepted wording, the wording must change.
