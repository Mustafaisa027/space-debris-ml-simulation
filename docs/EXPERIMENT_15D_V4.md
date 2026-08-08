# IAC26 15-day v4 preregistered confirmation protocol

## Status and scientific boundary

`iac26-15d-v4` is the planned claim-eligible confirmation experiment. Its
half-open collection window is frozen as 2026-08-10T00:17:00Z through
2026-08-25T00:17:00Z. No v4 acquisition may occur before this protocol and its
workflow are merged to `main`. Any change after the first accepted v4 bundle
requires a new experiment ID and a new future window.

The accepted IAC 114764 abstract binds the scientific method: open-source LEO
conjunction analysis from TLE data propagated with SGP4; minimum approach
distance, TCA and relative velocity in the dataset; Logistic Regression,
Random Forest, SVM and XGBoost; precision, recall and F1; and comparison with a
classical fixed-distance method. It does not bind a calendar date, collection
duration or acquisition cadence. v4 changes only the prospective acquisition
design needed to test that method reliably.

## Separation from prior experiments

- `iac26-10d-v2` remains an immutable failed pilot at 17/108 required slots.
- `iac26-10d-v3` remains an immutable failed pilot/audit experiment at 97/120
  occupied slots, 16.843611 hours endpoint-inclusive maximum gap, 30/97 unique
  TLE hashes and an eight-bin identical-hash run. Under its restored frozen
  gates the only valid outcome is `not_enough_data`.
- v2 and v3 bundles, rows and derived outputs are never pooled, backfilled,
  relabelled or reused as v4 observations. They inform only this prospective
  acquisition design and remain available for audit.

This is a pilot-informed preregistration, not a claim that the v2/v3 decisions
were preregistered. Their deviations and failures remain documented in
`docs/EXPERIMENT_10D_V3.md`.

## Frozen acquisition contract

- Experiment ID: `iac26-15d-v4`.
- Window: `[2026-08-10T00:17:00Z, 2026-08-25T00:17:00Z)`.
- Scientific cadence: one half-open three-hour slot, yielding 120 total slots
  over 15 days.
- Provider request floor: two elapsed hours between real provider fetches.
  This is separate from scientific slot width. The separation prevents a late
  fetch from forcing deterministic phase drift into later scientific slots.
- Delivery: GitHub Actions attempts at `:07`, `:17`, `:37` and `:47` each UTC
  hour. The archive-backed guard permits at most one accepted fetch per
  scientific slot and never requests the provider inside the two-hour floor.
- Catalogue: the unchanged 75 NORAD-ID cohort
  `iac26-leo-mixed-75-v2`, SHA-256
  `64c3d2329281da0e246226cb552aafe3b961c553c9487d7cfd9c96374aaa1f4c`.
- Archive: immutable manifest-schema-3 bundles only, beneath
  `experiments/iac26-15d-v4/collections/` on `data-collection`.
- Duplicate slots, mixed experiment IDs, partial catalogues, wrong hashes,
  malformed TLEs and out-of-window observations fail closed.

## Frozen simulation, label and model contract

The v3 values remain unchanged: 160-2000 km LEO bounds, 12-hour propagation
horizon, five-minute exact-TCA intervals, 30-second conservative screening,
200 km candidate threshold, 25 km fixed alarm, proxy-positive label at minimum
distance <= 50 km and TCA relative velocity >= 10 km/s, and maximum TLE age of
14 days. The proxy label is not Probability of Collision.

Snapshot-only XGBoost remains primary. Logistic Regression, Random Forest,
SVM and XGBoost use the same deterministic chronological pair-held-out design,
seeds, support requirements, bootstrap settings and adaptability gates as v3.
The 25 km fixed-distance comparator and train-only calibrated-distance
comparator remain unchanged.

## Frozen acquisition-quality gates

All four gates are binding and fail closed:

- occupied coverage >= 108/120 slots (90%);
- endpoint-inclusive maximum actual snapshot gap <= 6 hours;
- unique TLE-input hashes / occupied slots >= 30%;
- longest consecutive identical-hash run <= 6 occupied bins.

The 15-day/three-hour design preserves the same 120 target observations and
all four numerical thresholds. It adds acquisition time and one hour of
scheduler slack between the two-hour provider floor and each scientific slot;
it does not lower a failed v3 threshold.

## Monitoring and intervention policy

Cadence health reports operational liveness separately from scientific slot
quality. During an unoccupied active slot, scheduled attempts receive a
reasonable completion interval. A single manual `workflow_dispatch` fallback
is allowed only when no collector is successful or in progress, the slot is
approaching expiry, and two real hours have elapsed since the last archived
`fetched_utc`. A fallback cannot create or backdate a missed observation.

Collector or provider failures may be repaired only operationally. The config,
catalogue, labels, simulation, models, seeds, thresholds and evaluation flow
cannot change after collection begins.

## Release condition

After 2026-08-25T00:17:00Z, finalization runs strictly in this order:
window/cadence verification, clean archive and schema/hash import, corrected-TCA
resimulation, model evaluation, and publication evidence validation. Every
binding gate must pass. Otherwise the result is `not_enough_data`, no threshold
is relaxed, and the accepted abstract's numerical superiority language is not
asserted as a repository result.
