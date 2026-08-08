# IAC Abstract Alignment Review

This document treats IAC 2026 paper 114764's accepted abstract as the
scientific contract for the repository. Implementation coverage and empirical
evidence are deliberately separated: having code for a method is not evidence
that its result claim is true.

The exact accepted one-page PDF is bound in
`config/iac_114764_abstract_contract.json` by SHA-256. Regression tests require
its TLE/SGP4, dataset-field, four-model, precision/recall/F1, fixed-threshold
and fail-closed claim-gate requirements to remain connected to the active v4
pipeline.

## Requirement-to-evidence matrix

| Abstract requirement | Implementation | Evidence state |
| --- | --- | --- |
| Open-source LEO conjunction framework | Python pipeline and reproducible experiment config; MIT `LICENSE` at repository root | Implemented |
| TLE input and SGP4 propagation | CelesTrak/Space-Track providers, immutable snapshots, Skyfield SGP4 | Implemented |
| Potential encounter identification | Conservative coarse screening plus exact TCA refinement over every coarse interval | Implemented and regression-tested |
| Minimum distance, TCA, relative velocity dataset | Versioned candidate history with provenance and catalogue IDs | Implemented |
| Logistic Regression, Random Forest, XGBoost, SVM | Pair-held-out chronological evaluation; Decision Tree and LightGBM are supporting comparisons | Implemented; final evidence awaits sufficient data |
| Precision, recall and F1 | Per-model report and publication figures from one frozen prediction artifact | Implemented; final values await sufficient data |
| Comparison with fixed-distance thresholds | Binary 25 km point plus inner-validation-calibrated continuous-distance comparator; claim must beat both | Implemented |
| Reduced false alarm rate | Separate paired pair-cluster and UTC-day-block conditional sensitivity CIs plus recall non-inferiority gate | **Not yet established**; both marginal gates must pass on final real evidence and do not constitute a multiway/object-network CI |
| Higher adaptability | Separate paired AP inference on non-overlap future blocks with dyadic object x time-block bootstrap | Implemented fail-closed; final claim requires ten valid blocks, CI/p-value gates and real data |

## Current empirical gate

The active canonical experiment is the prospective, pilot-informed 15-day v4
protocol with three-hour scientific slots and a separate two-hour provider
request floor. v1, v2 and v3 are retained unmodified as pilot/audit data and
are never pooled into v4. v3 completed at 97/120 slots and failed its frozen
coverage, endpoint-gap and identical-hash-run gates; its result remains
`not_enough_data`. A final
model claim is permitted only after the corrected historical snapshots have
been re-simulated and the strict pair-held-out future split contains both
classes in train and test. Row count alone is not enough; positive events must
span multiple snapshots and independent catalogue pairs. The configured gate
requires 30/20 positive train/test rows, 10/5 positive train/test pairs, and
5/3 positive train/test snapshots. The frozen half-open v4 window contains 120
three-hour snapshot-time bins; at least 90% must be occupied and no gap may exceed
six hours. The gap is measured from actual snapshot timestamps, including the
window endpoints, rather than nominal bin indices. Publication evaluation also
rejects retained rows beyond the frozen 14-day TLE-age bound. TLE-hash
diversity must be at least 30%, and the longest identical-input run must not
exceed six bins. All four remain hard fail-closed requirements. The wider v4
scientific slot supplies delivery slack while retaining the same 120 target
observations and numerical gates (see `docs/EXPERIMENT_15D_V4.md`). Only the frozen
`iac26-leo-mixed-75-v2` catalogue cohort is
eligible; older mixed-size snapshots remain an audit archive.

Until that gate passes, the abstract sentence saying that results show
"significantly higher adaptability and a reduced false alarm rate" is a
hypothesis to test, not a repository result. If the completed experiment does
not support it, the paper/abstract wording must be revised rather than forcing
the analysis to match the claim.

## Scientific interpretation

- `risk_label` is a transparent geometry/kinematics proxy, not confirmed
  collision ground truth.
- Public TLEs do not provide the covariance, hard-body radius, or validated
  outcome data required for physical Probability of Collision (Pc).
- The claim-eligible supervised task predicts the future-propagated proxy among
  candidate-screened pairs from six snapshot-state quantities. The full
  13-feature rule-recovery arm is a non-claim-eligible positive control. Neither
  task may be described as operational collision-probability prediction.
- An exact proxy-rule oracle is retained as a non-claim-eligible positive
  control. Its perfect result documents deterministic target construction; it
  is not evidence of model performance. A non-perfect oracle fails the
  publication sanity gate.
- The fixed 25 km alarm supplies confusion counts and false alarm rate.
  Continuous `-min_distance_km` supplies the propagated distance-only reference
  ranking for PR-AUC/ROC-AUC. A second distance-only operating point is frozen
  exclusively on inner validation at comparable recall; the primary claim must
  beat both distance references. Because distance is itself one component of
  the proxy-label rule, this is a stringent proxy-component control rather than
  independent collision ground truth. The linear constant-velocity CPA control supplies
  a same-snapshot-information physics baseline; model-minus-CPA point
  differences and pair/day bootstrap intervals are reported as sensitivity
  evidence rather than promoted to the primary claim.
- False alarm comparisons are meaningful only alongside recall. A method that
  alarms less because it misses positives has not improved the system.

## Remaining work before submission

1. Complete the 15-day v4 immutable snapshot collection and corrected-TCA
   historical re-simulation.
2. Import the GitHub archive through manifest/hash verification, re-simulate,
   and publish the generated frozen pair/time split manifest.
3. Run LR, RF, XGBoost and SVM on the same real test partition; keep optional
   Decision Tree/LightGBM results in a supporting table.
4. Report precision, recall, F1, PR-AUC, confusion matrices and false alarm
   rate for the fixed-distance baseline and every model.
5. Run expanding-time pair-grouped folds and report mean, standard deviation,
   valid-fold count and class support.
6. Report snapshot-only XGBoost as the pre-specified primary experiment. Report
   snapshot geometry/kinematics ablations, constant-velocity CPA, deterministic
   label permutation, exact proxy-rule oracle, full rule-recovery, distance-only and transparent-rule
   arms only as non-claim-eligible controls on the identical frozen split.
7. Apply the pre-specified train-only comparable-recall operating point and
   2,000-draw paired pair-cluster and UTC-day-block bootstrap. Make a "reduced
   proxy false alarm" statement only if both resampling analyses pass their
   false-alarm and recall non-inferiority gates; require at least five held-out
   day blocks and state that uncertainty remains conditional on the observed catalogue.
8. Keep five-fold expanding-time CV descriptive. Run the separate frozen-future
   v4 adaptability protocol: ten eligible 12-hour blocks with two-hour embargo,
   30 objects, 30 pairs, 10,000 dyadic object x time-block bootstrap draws,
   one-sided 95% lower bound above zero, p < 0.05 and positive AP delta in every
   block. Revise the abstract unless this fail-closed inference passes.
9. Migrate to OMM before using catalogue numbers that legacy fixed-width TLE
   cannot represent safely.
10. ~~Publish the repository under a user-selected open-source license before
    describing the released implementation as open source.~~ Done: MIT
    `LICENSE` added at the repository root, referenced from `pyproject.toml`.

## Synthetic-data boundary

Synthetic augmentation is not part of the canonical result. It may be added
later as a separately named supporting experiment only after the real-data
split is frozen, only on training rows, and only if there are enough genuine
positive snapshots and pairs to support the sampler. The test set must remain
entirely real TLE-derived and byte-identical across experiment arms. Ordinary
SMOTE is feature-space statistical resampling, not a physically simulated
conjunction.
