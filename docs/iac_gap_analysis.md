# IAC Abstract Alignment Review

## Abstract Requirements

The abstract promises an open-source LEO conjunction analysis framework with:

- TLE-driven state propagation using SGP4.
- Identification of potential conjunction events.
- A dataset containing minimum approach distance, TCA, and relative velocity.
- Supervised models: Logistic Regression, Random Forest, XGBoost, and SVM.
- Evaluation using precision, recall, F1-score, and comparison with fixed-distance thresholds.
- A repeatable infrastructure suitable for student-led SSA research.

## Current Implementation Status

Implemented:

- TLE ingestion from local files and CelesTrak GP API.
- SGP4 propagation through Skyfield.
- LEO altitude filtering.
- Pairwise conjunction screening.
- Dataset columns for TCA, minimum distance, relative velocity, altitude difference, TLE epoch age, and risk score.
- Fixed-threshold baseline.
- Logistic Regression, Random Forest, SVM, and optional XGBoost.
- Repeated collection workflow for a 60-day experiment.
- Time-based train/test evaluation from accumulated history.

Important limitation:

- `risk_label` is a transparent proxy label because public TLE data does not include covariance or ground-truth collision labels. This should be stated in the paper. The project should not claim physical Probability of Collision (Pc); it should claim relative risk classification under open-data constraints.

## Gaps To Close Before IAC Submission

1. Increase sample size.
   The current smoke test uses only a few objects. For publishable results, collect 60 days of snapshots and include at least one larger CelesTrak group such as `STATIONS`, `WEATHER`, or a curated LEO catalog.

2. Avoid polling abuse.
   CelesTrak states that GP data is checked every 2 hours, so the collector is set to a minimum 2-hour interval.

3. Add XGBoost to the environment.
   Code supports it and `requirements.txt` includes it for the final comparison table.

4. Report proxy-label methodology.
   Explain that labels are generated from conjunction geometry: severe distance, distance threshold, relative velocity, and TCA window.

5. Add ablation studies.
   Compare models with and without `relative_velocity_km_s`, `time_to_tca_min`, and `max_tle_age_hours`.

6. Add statistical confidence.
   Report mean and standard deviation across time splits or weekly folds.

7. Add reproducibility metadata.
   Save collection timestamp, TLE filename, object count, thresholds, and software versions with every run.

8. Plan OMM migration.
   TLE is sufficient for the current abstract and legacy SGP4 workflow, but CelesTrak recommends OMM-compatible formats for future catalog-number growth and interoperability.

## Strong Additions

- A dashboard for monitoring top-N conjunction candidates over time.
- Weekly drift analysis: whether model performance changes as TLE epochs and orbital geometry evolve.
- Calibration plots for predicted risk score reliability.
- Confusion-matrix figures for each model.
- Larger open-data object sets using CelesTrak `GROUP` queries.
- Optional OMM/CSV ingestion later, because CelesTrak is moving beyond legacy TLE limitations.
