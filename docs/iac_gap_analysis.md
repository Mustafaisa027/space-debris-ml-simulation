# IAC Abstract Alignment Review

This document treats IAC 2026 paper 114764's accepted abstract as the
scientific contract for the repository. Implementation coverage and empirical
evidence are deliberately separated: having code for a method is not evidence
that its result claim is true.

## Requirement-to-evidence matrix

| Abstract requirement | Implementation | Evidence state |
| --- | --- | --- |
| Open-source LEO conjunction framework | Public Python pipeline and reproducible experiment config | Implemented |
| TLE input and SGP4 propagation | CelesTrak/Space-Track providers, immutable snapshots, Skyfield SGP4 | Implemented |
| Potential encounter identification | Conservative coarse screening plus exact TCA refinement over every coarse interval | Implemented and regression-tested |
| Minimum distance, TCA, relative velocity dataset | Versioned candidate history with provenance and catalogue IDs | Implemented |
| Logistic Regression, Random Forest, XGBoost, SVM | Pair-held-out chronological evaluation; Decision Tree and LightGBM are supporting comparisons | Implemented; final evidence awaits sufficient data |
| Precision, recall and F1 | Per-model report and publication figures | Implemented; final values await sufficient data |
| Comparison with fixed-distance thresholds | Binary 25 km operating point plus continuous distance ranking | Implemented |
| Reduced false alarm rate | Per-model `FP / (FP + TN)` and false-alarm-versus-recall curves are reported | **Not yet established**; final real held-out curves must show the reduction at comparable recall |

## Current empirical gate

The canonical experiment is 60 days at a two-hour collection cadence. A final
model claim is permitted only after the corrected historical snapshots have
been re-simulated and the strict pair-held-out future split contains both
classes in train and test. Row count alone is not enough; positive events must
span multiple snapshots and independent catalogue pairs. The configured gate
requires 30/20 positive train/test rows, 10/5 positive train/test pairs, and
5/3 positive train/test snapshots, plus the full 60-day cadence-adjusted
observation span. Only the frozen `iac26-leo-mixed-75-v1` catalogue cohort is
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
- The present supervised task measures recovery of a proxy screening rule.
  It must not be described as operational collision-probability prediction.
- The fixed 25 km alarm supplies confusion counts and false alarm rate.
  Continuous `-min_distance_km` supplies the fair distance-only ranking for
  PR-AUC/ROC-AUC.
- False alarm comparisons are meaningful only alongside recall. A method that
  alarms less because it misses positives has not improved the system.

## Remaining work before submission

1. Complete the 60-day immutable snapshot collection and corrected-TCA
   historical re-simulation.
2. Freeze the real pair-held-out chronological split and publish its manifest.
3. Run LR, RF, XGBoost and SVM on the same real test partition; keep optional
   Decision Tree/LightGBM results in a supporting table.
4. Report precision, recall, F1, PR-AUC, confusion matrices and false alarm
   rate for the fixed-distance baseline and every model.
5. Run expanding-time pair-grouped folds and report mean, standard deviation,
   valid-fold count and class support.
6. Report the implemented strict feature ablation that removes label-defining
   minimum distance, relative velocity, and the RIC position trio whose norm
   reconstructs minimum distance.
7. Make any "significant" or "reduced false alarm" statement only if the real
   held-out evidence supports it at comparable recall.
8. Migrate to OMM before using catalogue numbers that legacy fixed-width TLE
   cannot represent safely.

## Synthetic-data boundary

Synthetic augmentation is not part of the canonical result. It may be added
later as a separately named supporting experiment only after the real-data
split is frozen, only on training rows, and only if there are enough genuine
positive snapshots and pairs to support the sampler. The test set must remain
entirely real TLE-derived and byte-identical across experiment arms. Ordinary
SMOTE is feature-space statistical resampling, not a physically simulated
conjunction.
