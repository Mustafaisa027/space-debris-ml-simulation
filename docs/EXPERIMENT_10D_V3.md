# IAC26 10-day v3 frozen protocol

## Status and scientific boundary

`iac26-10d-v3` is a completed, immutable failed pilot/audit experiment. Its half-open
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
- TLE-hash diversity and longest identical-input run are binding publication
  gates. Their temporary post-start reclassification is recorded below as a
  protocol deviation; it does not alter the frozen final result.
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
Delayed GitHub schedules therefore produced 29 sub-two-hour intervals before
2026-07-31. One final request already admitted by the original guard brought
the immutable total to 30 among the first 61 verified v3 bundles; the minimum
was 20.95 minutes. The bundles remain valid, distinct physical snapshots and
duplicate slots remain rejected, but this request frequency did not respect
CelesTrak's one-download-per-update guidance.

The guard was corrected before the collection window closed. It now uses the
latest archived `fetched_utc` and refuses another provider request until the
configured two-hour interval has elapsed, even when a new scientific slot is
open. The incident and correction must remain in the acquisition provenance;
bundles must never be described as independent orbital-element updates merely
because their snapshot slots differ.

## Quality-gate deviations and restoration

The immutable v3 config binds four cadence-quality publication gates:
snapshot coverage >= 90%, endpoint-inclusive maximum gap <= 6 hours,
TLE-hash diversity >= 30%, and longest identical-hash run <= 6 bins. A failure
of any gate yields `not_enough_data`; an operational explanation does not turn a
failed frozen threshold into a passing result.

### Deviation chronology

- On 2026-07-27, after live v3 acquisition had begun and feed-cadence outcomes
  were visible, commit `f86fa1d` changed TLE-hash diversity and identical-run
  from gates to diagnostics.
- On 2026-08-02, after the GitHub Actions outage had made the temporal targets
  unreachable, PR #15 / commit `5b25444` changed coverage and endpoint gap from
  gates to diagnostics.

Both changes were post-start and outcome-dependent protocol deviations. They
must remain in the audit history and must not be described as preregistration or
as a pre-analysis amendment that preserves the original final gate.

### Final observed gate result

The immutable v3 archive contains 97 unique schema-3 bundles for 97/120 slots
(80.8333% coverage). Its endpoint-inclusive maximum gap is 16.843611 hours,
there are 30 unique TLE hashes among 97 snapshots (30.9278% diversity), and the
longest identical-hash run is 8 bins. Therefore coverage fails, endpoint gap
fails, diversity passes, and identical-hash run fails. The valid frozen-protocol
result is `not_enough_data`; no numerical abstract claim is eligible.

### Restoration

Before corrected-TCA resimulation, model fitting, or evidence generation, all
four enforcement sites are restored to fail-closed behavior: collection final
status, model-window validation, partition/fold validation, and evidence
generation. The config, catalogue, labels, thresholds, model definitions,
archive bundles, and overall architecture remain unchanged. The incident and
the two deviation commits remain part of the transparent provenance record.

## Release condition

No numerical abstract claim is released until archive verification,
corrected-TCA resimulation, model evaluation, and publication validation all
pass. If evidence contradicts the accepted wording, the wording must change.
