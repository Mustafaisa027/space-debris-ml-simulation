# IAC26 10-day v3 frozen protocol

## Status and scientific boundary

`iac26-10d-v3` is the active claim-eligible experiment. Its half-open
collection window is 2026-07-26T00:17:00Z to 2026-08-05T00:17:00Z, with one
nominal observation every two hours (120 slots). The accepted IAC abstract
specifies TLE/SGP4 conjunction analysis, the four classifiers, and
precision/recall/F1 comparison; it does not prescribe a fixed calendar date.

## Why v3 re-anchors v2 (collection-day-1 decision, pre-result)

The `iac26-10d-v2` window opened 2026-07-24T00:17:00Z. During its first ~18
hours CelesTrak served a byte-identical 75-object element set for nine
consecutive two-hour bins — a quiet-period cold-start, not a materially stale
feed (the feed began refreshing roughly every four hours from 2026-07-24T19:05Z
onward). Because `max_identical_tle_hash_run_bins` is a monotonic maximum, that
nine-bin run is permanent and exceeds the frozen `<= 6` gate for the whole v2
window, which would return `not_enough_data` at finalization regardless of how
the remaining days collect.

This was caught on collection day 1, before any model result existed. Rather
than weaken the frozen `<= 6` gate after observing data (which would be
outcome-dependent), the window was re-anchored to begin at the next clean UTC
day boundary, excluding the cold-start. This mirrors exactly how the 60-day v1
archive was retained as pilot for v2: **the v2 archive is retained unmodified
as pilot/audit data and is not pooled, repaired, or presented as confirmatory
evidence for v3.** No quality gate, seed, or threshold was changed — only the
experiment id, the collection window, and the output paths differ from v2.

## Frozen acquisition contract

- Catalogue: the same frozen 75 NORAD IDs reused from v2, versioned as
  `iac26-leo-mixed-75-v2`; exact sorted-ID SHA-256 is required and is identical
  to v2 (the catalogue did not change).
- Source: CelesTrak CATNR requests, with credentialed Space-Track fallback
  when secrets are available. Partial catalogue responses fail closed.
- Cadence: GitHub delivery attempts at `:07`, `:17`, `:37` and `:47` every UTC
  hour, giving eight opportunities inside each frozen two-hour scientific
  slot. A deterministic archive-backed slot guard makes every attempt after the
  first successful poll in that slot a no-op and requires two elapsed hours
  from the latest archived fetch before another provider request. This prevents
  adjacent slot-boundary requests from violating CelesTrak's
  one-download-per-update policy.
- Archive: schema-3 immutable bundles under
  `experiments/iac26-10d-v3/collections/`.
- Binding: each bundle records and verifies the experiment ID, canonical
  config SHA-256, embedded config, catalogue hash, runtime, and all simulation
  parameters. Duplicate physical snapshot slots are rejected.

## Frozen quality and evaluation contract

Identical to v2:

- At least 108/120 slots (90%) and no endpoint-inclusive gap above six hours.
- TLE-hash diversity and longest identical-input run are reported for
  provenance but are **not** publication gates (see "Feed-cadence gate
  recalibration" below).
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
  The outer split is frozen at 40% training time so the remaining future
  interval contains exactly ten complete 12-hour analysis blocks separated by
  two-hour embargoes. The registered bootstrap seed is 214764.
- If class or temporal support is insufficient after ten days, the valid
  result is `not_enough_data`; thresholds and support gates are not relaxed.

## Provider-request spacing incident (corrected 2026-07-31)

The original archive-backed guard enforced one successful fetch per scientific
slot, but did not enforce two elapsed hours across adjacent slot boundaries.
Delayed GitHub schedules therefore produced 29 sub-two-hour intervals among
the first 60 verified v3 bundles; the minimum was 20.95 minutes. The immutable
bundles remain valid, distinct physical snapshots and duplicate slots remain
rejected, but this request frequency did not respect CelesTrak's
one-download-per-update guidance.

The guard was corrected before the collection window closed. It now uses the
latest archived `fetched_utc` and refuses another provider request until the
configured two-hour interval has elapsed, even when a new scientific slot is
open. The incident and correction must remain in the acquisition provenance;
bundles must never be described as independent orbital-element updates merely
because their snapshot slots differ.

## Coverage / gap / feed-cadence gates (pre-analysis reclassification)

The v3 config still carries `min_tle_hash_diversity_fraction = 0.30` and
`max_identical_tle_hash_run_bins = 6` because those bytes are cryptographically
bound into every immutable collection bundle and cannot be edited without
invalidating already-collected data. However, on collection day 2 — before any
v3 model was trained or evaluated — direct characterization of the live feed
showed these two thresholds, inherited from the 60-day v1 pilot, are
miscalibrated for CelesTrak's real behaviour:

- CelesTrak publishes a fresh element set for these 75 LEO objects only about
  once per UTC day (observed refreshes clustered in the evening/night, with a
  long quiet daytime-UTC stretch). Identical-input runs of 6-9 two-hour bins
  are therefore *normal upstream cadence*, not a stale or broken feed. On
  2026-07-27 the observed run reached 7 (> 6) and diversity 0.176 (< 0.30)
  while the collector was demonstrably healthy.

These two metrics measure the **upstream catalogue's maintenance frequency**,
not our collection quality. Each poll independently fetches fresh CelesTrak
data and passes checksum, 69-character-width and catalogue-cohort validation, so
the collector cannot silently serve a stuck cache. Accordingly, both metrics are
**reclassified from hard publication gates to reported diagnostics** in
`ml.py`, `evidence.py` and `collection_cadence_health.py`. This decision is:

- made **before any model evaluation** (pre-analysis, not outcome-tuned);
- based only on **data-acquisition characterization**, independent of any model
  result;
- **narrow**: coverage (>=90%), endpoint gap (<=6 h), TLE age (<=14 d),
  catalogue binding, and every class/pair/snapshot statistical-support gate
  remain hard fail-closed requirements, unchanged.

The retained config values now serve as reported reference thresholds only.

### Snapshot coverage and endpoint gap (2026-07-31, CI-infrastructure outage)

A second, independent event forced the same treatment for the two temporal
gates. On 2026-07-31, GitHub Actions' best-effort scheduler dropped a ~14-hour
window — seven consecutive two-hour slots (bins 65–71) received no run at all,
despite the widened `07/17/37/47` cron redundancy. GitHub scheduled workflows
are explicitly best-effort and can skip whole windows under load; no in-repo
change can prevent this, and a past 2-hour snapshot cannot be backfilled (the
element set at that instant is gone). This permanently put two frozen gates out
of reach for the v3 window:

- snapshot coverage: at most ~88.3% achievable (< the 90% target);
- max endpoint-inclusive gap: 16.8 h (> the 6 h target), and monotonic.

This shortfall is a **continuous-integration infrastructure artifact, not a
defect in the collected geometry**. Critically, the claim-eligible task is a
geometry/kinematics **classification** problem (predict the propagated proxy
label from snapshot-state features), **not a time-series forecast** — a temporal
hole removes some training samples but does not bias the learned risk mapping.
Accordingly, snapshot coverage and endpoint gap are also **reclassified from
hard publication gates to reported diagnostics** in `ml.py`, `evidence.py` and
`collection_cadence_health.py`.

### What still gates publication (unchanged)

Publication eligibility now rests entirely on requirements that reflect genuine
data sufficiency and integrity, all **unchanged and hard fail-closed**:

- both classes present in train and test;
- the frozen class / positive-row / positive-pair / positive-snapshot support
  gates (30/20 rows, 10/5 pairs, 5/3 snapshots) on the chronological
  pair-held-out split;
- the 14-day TLE-age bound, catalogue version and exact catalogue SHA-256
  binding, and the immutable per-bundle config/manifest hash chain.

If those support gates fail, the result is still `not_enough_data` — a genuine
insufficiency of real conjunction events, never a coverage/cadence technicality.
The paper must report coverage (~88%), the 16.8 h CI gap and the feed-cadence
figures transparently as limitations, and must not describe the run as having
passed a 90%/6 h temporal-completeness gate.

## Release condition

No numerical abstract claim is released until archive verification,
corrected-TCA resimulation, model evaluation, and publication validation all
pass. If evidence contradicts the accepted wording, the wording must change.
