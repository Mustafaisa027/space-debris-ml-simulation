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
- At least 30% unique TLE hashes across occupied slots and no identical-hash
  run above six bins.
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

## Quality-gate deviation and restoration

From 2026-07-27 through 2026-07-31, implementation commit `f86fa1d`
temporarily treated TLE-hash diversity and the longest identical-input run as
diagnostics after those frozen thresholds failed on live acquisition data.
That post-start reclassification contradicted the immutable v3 config and the
rule that evidence gates must not be relaxed after observing results.

Before finalization or any claim release, all four enforcement sites were
restored: collection-window validation, time-series fold validation, evidence
generation and cadence final-gate reporting. The temporary deviation remains
part of the audit trail and must not be described as preregistration. With the
restored contract, a final diversity below 30% or an identical-hash run above
six bins yields `not_enough_data`; it cannot be waived because the upstream
feed refreshes slowly.

## Release condition

No numerical abstract claim is released until archive verification,
corrected-TCA resimulation, model evaluation, and publication validation all
pass. If evidence contradicts the accepted wording, the wording must change.
