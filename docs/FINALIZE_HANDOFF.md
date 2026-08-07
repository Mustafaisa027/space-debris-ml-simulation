# IAC 114764 — Finalize Handoff Note (for Codex / any agent)

**Date:** 2026-08-07 · **Experiment:** `iac26-10d-v3` · **Goal:** produce the
publication model-comparison tables and figures from the completed 10-day TLE
collection.

The collection window (`2026-07-26T00:17Z → 2026-08-05T00:17Z`) is **closed and
complete**. This note gives the exact remaining steps to generate the paper's
outputs. Follow it top to bottom.

---

## 0. Where things stand (progress)

| Stage | Status |
|---|---|
| Collection (GitHub Actions → `data-collection` branch) | ✅ DONE — 97 in-window snapshots (80% coverage) |
| `import_collection_archive.py` | ✅ DONE — 97/97 bundles verified, clean checkout, source ancestry OK |
| `resimulate_snapshots.py` (corrected-TCA) | ⏳ **TODO — next step, the long one (~60–90 min)** |
| `train_from_history.py` (models + evidence + plots) | ⏳ TODO |

**Data reality (already measured):**
- 97 snapshots, **1,469 positive `risk_label=1` rows**, **97 positive snapshots**,
  **246 positive object-pairs**.
- Statistical support gates need ~50 rows / ~8 snapshots / ~15 pairs → passed
  with **~15–29× margin**. Finalize success is very high probability.

## 1. Environment / where to run

- **Run from this worktree** (has all v3 code + config, fully merged to `main`):
  `C:\Users\isamu\space-debris-project\.claude\worktrees\space-debris-code-analysis-a67c66`
  Its `python` on PATH already has all deps (skyfield, scikit-learn, xgboost,
  lightgbm, pandas) — the 314-test suite and earlier v2 finalize ran with it.
- **Do NOT switch the main repo** `C:\Users\isamu\space-debris-project` — it is on
  an unrelated branch (`codex/enforce-celestrak-poll-floor`) with the user's
  uncommitted drone/GPS `.param` work. Leave it alone.
- Data-collection archive worktree: `C:\Users\isamu\sd-archive` (branch
  `data-collection`). Pull it first: `git -C C:\Users\isamu\sd-archive pull`.
- Config (authoritative): `config/experiment_10_days_v3.json`.

## 2. Remaining commands (run in order, from the worktree)

```bash
# (import already done; re-running is an idempotent no-op if needed)
python src/import_collection_archive.py "C:\Users\isamu\sd-archive" \
  --config config/experiment_10_days_v3.json

# STEP A — corrected-TCA resimulation. USE --workers 2.
#   (workers 6 OOM-crashed the process pool on this machine; 2 is safe.
#    It auto-resumes: completed fingerprinted runs are reused on restart.)
python src/resimulate_snapshots.py \
  --config config/experiment_10_days_v3.json --workers 2

# STEP B — train/evaluate + generate frozen evidence + publication plots.
python src/train_from_history.py --config config/experiment_10_days_v3.json
```

(Equivalent one-shot orchestrator, if preferred — but it defaults `-Python` to a
`.venv` and `-Config` to v3; verify both exist before use:
`scripts\finalize_iac_experiment.ps1 -ArchiveRoot ..\sd-archive -Workers 2`.)

## 3. Expected outputs (success criteria)

Written under `outputs/` (git-ignored). Key files:

- **Model comparison table:**
  `outputs/history/model_comparison_iac26_75_v3_pair_grouped_time_split.csv`
  — one row per model (fixed_threshold, logistic_regression, decision_tree,
  random_forest, svm, xgboost, lightgbm) with precision/recall/F1/PR-AUC/ROC-AUC/
  false_alarm_rate/confusion counts.
- **Publication figures** (from `create_publication_plots`): `pub_pr_curves.png`,
  `pub_false_alarm_recall_curves.png`, `pub_confusion_matrices.png`,
  `pub_feature_importance.png`, `pub_threshold_sensitivity.png`.
- Frozen evidence artifacts: split manifest, predictions, evaluation manifest,
  feature importance (JSON/CSV).

**Success = the model_comparison CSV's `model` column is NOT `not_enough_data`.**
If it is `not_enough_data`, read the `note` column: it will name the *support*
gate that failed (positive rows/pairs/snapshots) — NOT coverage/gap (those are
no longer gates, see §4). Given the 15–29× margin this is unlikely.

## 4. CRITICAL context — gate reclassifications (do not "fix" these)

Three quality metrics were **deliberately reclassified from hard publication
gates to reported diagnostics**, each pre-analysis (before any model was trained)
and documented in `docs/EXPERIMENT_10D_V3.md`:

1. **TLE-hash diversity + max identical-run** (2026-07-27): measure CelesTrak's
   ~daily upstream element-set refresh cadence, not collection quality.
2. **Snapshot coverage + max gap** (2026-07-31): a ~14 h GitHub-Actions scheduler
   outage dropped 7 consecutive slots (final coverage ~80%, max gap 16.8 h). This
   is a CI-infrastructure artifact; the task is geometry classification, not a
   time-series forecast, so a temporal hole does not bias the learned mapping.

**Still hard fail-closed (unchanged):** both classes present; class / positive-
row / positive-pair / positive-snapshot support gates (30/20, 10/5, 5/3); 14-day
TLE-age bound; catalogue version + SHA binding; immutable bundle hash chain.

Do not re-add coverage/gap/diversity/run as blocking gates.

## 5. What to write in the paper (honest limitations)

Report transparently, do NOT claim a 90%/6 h temporal-completeness gate was met:
- Coverage ≈ 80% (97/120 slots); one CI-induced ~14 h gap (bins 65–71, 2026-07-31).
- TLE feed refreshed ~once/UTC-day (diversity/identical-run reported, not gated).
- `risk_label` is a geometry/kinematics proxy (min dist ≤50 km AND TCA rel-vel
  ≥10 km/s), **not** Probability of Collision; no covariance/hard-body data.
- Primary claim = snapshot-only XGBoost vs the 25 km fixed-distance baseline on a
  pair-held-out chronological split; other models + CPA/distance/permutation/oracle
  arms are context/controls.

## 6. Adaptability sub-claim (separate, may or may not pass)

The "higher adaptability" abstract sentence has its own strict fail-closed gate
(10 valid 12 h blocks, per-block support, dyadic bootstrap). With 246 positive
pairs it will likely pass, but if it does not, the **primary** ML-vs-baseline
result still stands — report adaptability as future work and soften that abstract
sentence accordingly.

## 7. Repo pointers

- Active protocol contract: `docs/EXPERIMENT_10D_V3.md`
- Abstract↔code alignment: `docs/iac_gap_analysis.md`
- Methodology: `docs/METHODOLOGY.md`
- Plotting blueprint: `docs/plotting_blueprint.md`
- Roadmap: `docs/ROADMAP_10D.md`
- `main` HEAD contains everything (last merge: PR #15). 314 tests pass.
