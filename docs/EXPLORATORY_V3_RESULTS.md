# IAC26 v3 exploratory model comparison

## Status

This document records a **non-claim-eligible exploratory analysis** of the
completed `iac26-10d-v3` archive. The frozen confirmatory protocol did not pass
its acquisition-quality gate: 97/120 scientific slots were occupied (80.8333%,
required 90%), the endpoint-inclusive maximum gap was 16.843611 hours (allowed
6 hours), and the longest identical-TLE-hash run was 8 bins (allowed 6). The
canonical result therefore remains `not_enough_data`. Nothing in this analysis
changes, relaxes, or retrospectively reclassifies those gates.

## Data and evaluation

All 97 schema-3 bundles were verified against their manifests, catalogue hash,
embedded experiment configuration, runtime provenance, and source-commit
ancestry. Every snapshot was then re-simulated with the same corrected global
TCA algorithm before rebuilding the canonical history.

- Corrected-TCA candidate rows: 19,018
- Proxy-positive rows: 1,469 (7.724%)
- Unique catalogue pairs: 1,656
- Snapshot blocks: 97
- Outer training rows: 5,705
- Held-out future/test-pair rows: 2,795
- Held-out positives: 223 across 43 catalogue pairs and 58 snapshots
- Cutoff: 2026-07-29T15:13:52Z

The split preserves the v3 deterministic pair assignment and chronological
40% cutoff. Training and test catalogue pairs do not overlap, and test rows
occur after the cutoff. Models receive only the six quantities available at
the observation snapshot: current distance, altitude difference, maximum TLE
age, radial velocity, tangential velocity, and approach angle. TCA,
minimum-distance, TCA-relative-velocity, risk score, and other target-defining
or target-reconstructing quantities are excluded.

## Exploratory results

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | False-alarm rate |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 25 km distance alarm | **0.725** | **0.975** | **0.754** | 0.386 | **0.510** | 0.011 |
| Logistic Regression | 0.385 | 0.826 | 0.170 | 0.839 | 0.283 | 0.355 |
| Decision Tree | 0.137 | 0.608 | 0.289 | 0.274 | 0.281 | 0.058 |
| Random Forest | 0.412 | 0.877 | 0.620 | 0.197 | 0.299 | **0.010** |
| SVM | **0.418** | **0.904** | 0.265 | **0.874** | 0.406 | 0.211 |
| XGBoost | 0.367 | 0.850 | 0.360 | 0.439 | 0.396 | 0.068 |
| LightGBM | 0.408 | 0.876 | 0.338 | 0.623 | **0.438** | 0.106 |

Bold values among learned models identify the best learned-model result for a
metric; bold baseline values indicate that the fixed alarm exceeds every
learned model. The strongest learned-model PR-AUC was SVM at 0.418, 0.307 below
the fixed-distance ranking baseline. The strongest learned-model F1 was
LightGBM at 0.438, 0.072 below the fixed alarm. SVM recovered 87.4% of proxy
positives but produced 542 false positives; Random Forest matched the fixed
alarm's false-alarm rate but recovered only 19.7% of positives.

## Interpretation and paper boundary

The snapshot-only learners did not outperform the fixed-distance comparator on
this held-out pair/time split. This is scientifically informative rather than a
software failure. The proxy target is defined partly by future minimum
distance, while the primary learners are intentionally denied that
future-propagation quantity. The continuous minimum-distance score used to rank
the fixed comparator is consequently closely aligned with the proxy target.

These results may be described as pilot-informed model-development evidence or
as a negative exploratory result. They must not be presented as the frozen
v3 confirmatory result, as proof of operational collision-risk prediction, or
as evidence that any learner reduces false alarms at non-inferior recall. A new
prospectively registered experiment is required for such claims.

## Reproduction

```powershell
python src/import_collection_archive.py <data-collection-worktree> `
  --config config/experiment_10_days_v3.json
python src/resimulate_snapshots.py `
  --config config/experiment_10_days_v3.json --workers 6
python src/train_from_history.py `
  --config config/experiment_10_days_v3.json
python src/exploratory_compare.py `
  --config config/experiment_10_days_v3.json
```

The final command writes a provenance-headered metric table, PNG comparison,
and machine-readable manifest under `outputs/exploratory_iac26_10d_v3/`. The
manifest sets `claim_eligible=false`, `publication_use_permitted=false`, and
`cadence_publication_gates_enforced=false`.
