# Machine Learning-Based Simulation for LEO Space-Debris Conjunction Risk: A Leakage-Resistant Exploratory Evaluation

**Paper ID:** IAC-26, 114764  
**Manuscript status:** Technical-paper draft based on the completed `iac26-10d-v3` pilot archive  
**Evidence status:** Exploratory; not eligible for the pre-registered confirmatory claims  

> Author names, affiliations, corresponding-author details, conference track,
> acknowledgements, and the final bibliography must be inserted before
> submission.

## Abstract

This study presents an open-source simulation and machine-learning framework
for screening conjunction risk among catalogued objects in low Earth orbit.
Public two-line element sets are propagated with SGP4, candidate encounters are
identified by conservative coarse screening followed by refined time-of-closest-
approach search, and each retained object pair is described by conjunction
geometry and kinematics. A deterministic proxy target identifies encounters
whose propagated minimum separation is at most 50 km and whose relative
velocity at closest approach is at least 10 km/s. To prevent temporal leakage
and pair memorisation, the evaluation combines a chronological cutoff with
deterministic held-out catalogue pairs. The primary learners receive only six
quantities available at the observation snapshot; future closest-approach and
target-defining variables are excluded. Logistic regression, random forest,
support vector machine, and XGBoost are compared with a fixed 25 km distance
alarm, with decision tree and LightGBM included as supporting models. The
completed archive contained 97 verified snapshots, 19,018 candidate-pair rows,
and 1,469 proxy-positive rows. The held-out evaluation contained 2,795 rows and
223 positives. Among learned models, SVM achieved the highest PR-AUC (0.418)
and recall (0.874), while LightGBM achieved the highest F1 score (0.438).
However, the fixed-distance alarm achieved a higher F1 score (0.510) and
PR-AUC (0.725) than every learned model. The results therefore do not support
claims of superior false-alarm reduction or adaptability. They instead expose
the difficulty of predicting a future-propagated distance-based proxy from
snapshot-only features under pair-held-out temporal evaluation. Because the
archive failed its pre-specified acquisition-cadence gates, all numerical
results are reported as transparent pilot evidence rather than confirmatory or
operational collision-risk performance.

**Keywords:** space debris; low Earth orbit; conjunction assessment; SGP4;
machine learning; time of closest approach; false alarms; temporal validation

## 1. Introduction

The growing number of active spacecraft and debris objects in low Earth orbit
increases the number of conjunctions that must be screened. Distance-threshold
alarms are transparent and inexpensive, but they may not represent the full
geometry and kinematics of an encounter. Machine-learning models can combine
multiple observed quantities, yet an apparently strong evaluation can be
misleading when the same catalogue pair appears in training and testing, when
future closest-approach variables leak into the predictors, or when random row
splits mix observations from the same orbital snapshot.

This work evaluates whether snapshot-state features can predict a transparent
future-propagation proxy on previously unseen catalogue pairs in a later time
period. The contribution is primarily methodological:

1. an auditable TLE/SGP4 acquisition and conjunction-simulation pipeline;
2. conservative encounter screening followed by refined closest-approach
   estimation;
3. a feature boundary that excludes target-defining and future information;
4. a combined pair-held-out and chronological evaluation; and
5. direct reporting of a negative result when learned models do not outperform
   a strong distance-based comparator.

The framework does not estimate physical probability of collision. Public TLE
data do not contain the covariance and hard-body-radius information required
for a reliable probability-of-collision calculation. The target used here is
therefore a simulation-derived proxy rather than operational ground truth.

## 2. Data and methods

### 2.1 Catalogue and observation archive

The frozen catalogue contains 75 LEO objects identified by NORAD catalogue
number (`iac26-leo-mixed-75-v2`). TLE snapshots were acquired from CelesTrak,
with a configured authenticated Space-Track fallback, and stored as immutable
schema-3 bundles. Each bundle binds the experiment identifier, embedded
configuration, catalogue hash, source data hash, runtime provenance, and
simulation parameters.

The completed `iac26-10d-v3` archive spans 26 July to 5 August 2026. Ninety-seven
of 120 nominal two-hour slots were occupied (80.833%). All 97 bundles were
verified before resimulation. The archive failed three frozen acquisition
gates: minimum coverage of 90%, endpoint-inclusive maximum gap of 6 h, and a
maximum identical-TLE-hash run of six bins. The observed maximum gap was
16.844 h and the longest identical-hash run was eight bins. Consequently, the
registered confirmatory result is `not_enough_data`; this manuscript reports a
separately labelled exploratory analysis.

### 2.2 Orbit propagation and conjunction identification

TLEs are propagated using the modernised SGP4 implementation documented by
Vallado et al. [1] over a 12 h horizon. A conservative 30 s coarse
screen excludes only pairs for which a 200 km encounter is physically
impossible under the configured upper relative-speed bound and numerical
guard. Remaining pairs are evaluated over every 5 min interval, with refined
time-of-closest-approach (TCA) search applied within each interval, following
the close-approach problem class described by Alfano and Negron [2]. This
procedure records minimum distance, TCA, and relative velocity at TCA for each
candidate pair.

The resimulated archive contains 19,018 candidate observations from 1,656
unique catalogue pairs across 97 snapshot blocks. Of these, 1,469 observations
(7.724%) satisfy the proxy-positive definition

\[
y = \mathbb{1}\left[d_{\min} \le 50\ \mathrm{km}\;\land\;
v_{\mathrm{rel,TCA}} \ge 10\ \mathrm{km\,s^{-1}}\right].
\]

This label represents propagated encounter severity under the simulation
assumptions; it is not a confirmed collision outcome or probability of
collision.

### 2.3 Features and leakage controls

Learned models receive six predictors available at the snapshot time:

- current pair distance (km);
- altitude difference (km);
- maximum TLE age (h);
- radial relative velocity (km/s);
- tangential relative velocity (km/s); and
- approach angle (degrees).

Risk score, minimum future distance, relative velocity at TCA, TCA and
time-to-TCA, TCA-frame coordinates, and other target-defining or
target-reconstructing variables are excluded. Feature preprocessing is fitted
only on the training partition.

### 2.4 Evaluation design

Catalogue pairs are assigned deterministically using SHA-256 and the frozen
seed `iac26-pair-split-v2`. The first 40% of complete snapshot times forms the
training-time region, while 25% of catalogue pairs are reserved for testing.
Only training-pair observations before the cutoff enter model fitting; only
held-out-pair observations after the cutoff enter testing. Cross-quadrant rows
are excluded. This design prevents both future-to-past leakage and direct pair
memorisation.

The cutoff is 29 July 2026 at 15:13:52 UTC. The resulting training set contains
5,705 rows from 883 pairs, including 470 positives. The test set contains 2,795
rows from 361 held-out pairs, including 223 positives across 43 positive pairs
and 58 snapshots. No catalogue pair overlaps between training and test.

The required classifiers are logistic regression (LR), random forest (RF) [5],
support vector machine (SVM) [6], and XGBoost [7]. A decision tree and LightGBM
[8] are reported as supporting comparisons. The classical comparator alarms when
propagated minimum distance is at most 25 km. Performance is reported using
precision, recall, F1, accuracy, false-alarm rate (FAR), area under the
precision-recall curve (PR-AUC), and area under the receiver-operating-
characteristic curve (ROC-AUC). Because positives constitute 7.98% of the
held-out set, PR-AUC and the precision-recall trade-off are emphasised [4].

## 3. Results

### 3.1 Held-out model comparison

**Table 1.** Pair-held-out chronological test performance. The fixed alarm is
the classical comparator; DT and LightGBM are supporting models.

| Method | PR-AUC | ROC-AUC | Precision | Recall | F1 | FAR | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed 25 km alarm | **0.725** | **0.975** | **0.754** | 0.386 | **0.510** | 0.011 | 86 | 28 | 137 | 2544 |
| Logistic regression | 0.385 | 0.826 | 0.170 | 0.839 | 0.283 | 0.355 | 187 | 912 | 36 | 1660 |
| Decision tree | 0.137 | 0.608 | 0.289 | 0.274 | 0.281 | 0.058 | 61 | 150 | 162 | 2422 |
| Random forest | 0.412 | 0.877 | 0.620 | 0.197 | 0.299 | **0.010** | 44 | 27 | 179 | 2545 |
| SVM | **0.418** | **0.904** | 0.265 | **0.874** | 0.406 | 0.211 | 195 | 542 | 28 | 2030 |
| XGBoost | 0.367 | 0.850 | 0.360 | 0.439 | 0.396 | 0.068 | 98 | 174 | 125 | 2398 |
| LightGBM | 0.408 | 0.876 | 0.338 | 0.623 | **0.438** | 0.106 | 139 | 272 | 84 | 2300 |

Bold learned-model entries identify the best learned result for that metric;
bold baseline entries indicate that the baseline exceeded all learned models.

SVM produced the strongest learned-model ranking performance (PR-AUC 0.418)
and recovered 87.4% of positives, but generated 542 false positives and a FAR
of 0.211. LightGBM produced the strongest learned-model F1 score (0.438), with
recall 0.623 and precision 0.338. Random forest approximately matched the fixed
alarm's FAR (0.010 versus 0.011) but detected only 19.7% of positives. Thus, its
lower alarm burden resulted from substantially lower sensitivity rather than a
superior operating trade-off.

The fixed 25 km alarm achieved F1 = 0.510, 0.072 above LightGBM, and PR-AUC =
0.725, 0.307 above SVM. None of the learned models therefore dominated the
fixed comparator on the held-out partition.

![Exploratory held-out model comparison](../outputs/exploratory_iac26_10d_v3/model_metrics_exploratory.png)

**Figure 1.** Exploratory held-out comparison generated from the provenance-
bound metric artifact. Numerical values are non-claim-eligible because the
source archive failed the registered acquisition-cadence gates.

### 3.2 Interpretation of the negative result

The outcome is consistent with the structure of the prediction task. The
proxy target directly contains propagated minimum distance, while the fixed
comparator uses that closely aligned future-propagation quantity. In contrast,
the learned models are intentionally restricted to snapshot-state variables.
The comparison is therefore stringent but scientifically useful: it tests
whether instantaneous geometry and kinematics generalise to future conjunction
severity on unseen object pairs. Under the present archive and split, they do
not outperform the distance reference.

These findings reject, for this pilot, the hypotheses that the learned model
reduces proxy false alarms at non-inferior recall or exhibits higher measured
adaptability. The result should not be reframed as a positive operational
claim.

## 4. Discussion

Random row-wise validation would allow repeated observations of the same pair
and adjacent snapshots to appear on both sides of the split. The lower scores
reported here are more conservative because catalogue identity and time are
both separated. This distinction is important for conjunction datasets, where
rows are not independent and repeated orbital geometry may otherwise be
memorised.

The fixed-distance result should also be interpreted carefully. Its advantage
does not establish that a 25 km alarm is operationally optimal; rather, it
shows that a comparator constructed from a component closely related to the
proxy target is difficult to beat. A more operational target would require
validated conjunction messages, covariance, hard-body radii, manoeuvre context,
and ideally resolved outcomes, consistent with the operational information
described in NASA's satellite-operator handbook [3]. Such information is
absent from the public TLE archive used here.

The different learned models expose clear operating trade-offs. SVM and LR
favour recall but issue many false alarms. RF favours specificity but misses
most positives. LightGBM offers the strongest learned F1 balance, while the
pre-specified XGBoost model lies between these extremes. Threshold calibration
can move these operating points, but it cannot be tuned on the held-out test
set without invalidating the evaluation.

## 5. Limitations

1. The archive missed 23 of 120 scheduled snapshot slots and failed the frozen
   coverage, gap, and repeated-input gates; numerical results are exploratory.
2. TLE/SGP4 uncertainty is not represented by covariance propagation.
3. The deterministic target is a geometry/velocity proxy, not collision
   probability or confirmed collision ground truth.
4. The 75-object cohort and approximately ten-day span limit catalogue and
   temporal generalisation.
5. Observations from shared objects and neighbouring times remain correlated;
   row-level metric estimates must not be interpreted as independent trials.
6. No confirmatory bootstrap confidence interval or adaptability inference is
   claimed for this failed-gate pilot.

## 6. Reproducibility and open-source artifacts

The implementation, frozen configuration, immutable input manifests,
resimulation reports, model-comparison table, and figure-generation code are
versioned in the project repository. The exact exploratory metric artifact is
`outputs/exploratory_iac26_10d_v3/model_comparison_exploratory.csv`; its source
history SHA-256 is
`bbbd5a845565cc767331ae7135c5ce2c6cfec47e6969dbcc5fa02911bdf8bdc9`.
The machine-readable manifest explicitly sets `claim_eligible=false` and binds
the output hashes to the evaluated history and code commit.

The result can be regenerated from the verified archive with:

```powershell
python src/import_collection_archive.py <data-collection-worktree> `
  --config config/experiment_10_days_v3.json
python src/resimulate_snapshots.py `
  --config config/experiment_10_days_v3.json --workers 6
python src/train_from_history.py `
  --config config/experiment_10_days_v3.json
python src/exploratory_compare.py `
  --config config/experiment_10_days_v3.json
python -m pytest -q
```

At the time of manuscript preparation, the test suite contains 320 passing
tests covering propagation, distance and altitude calculations, target and
feature boundaries, split integrity, archive validation, and evidence gates.

## 7. Conclusion

This work provides a reproducible, leakage-resistant framework for comparing
machine-learning classifiers in TLE/SGP4 conjunction screening. On a held-out
future partition of unseen catalogue pairs, SVM maximised learned-model recall
and PR-AUC, while LightGBM maximised learned-model F1. Neither surpassed the
fixed-distance comparator overall. The scientifically defensible conclusion is
therefore negative: snapshot-only machine-learning models did not demonstrate
superior proxy-risk discrimination or false-alarm performance in this pilot.
The evaluation design, provenance controls, and transparent failed-gate
reporting provide a foundation for future work using more complete data and
more operationally meaningful targets.

## Declarations

**Data and code availability.** Source code and auditable experiment artifacts
are available in the accompanying repository under the MIT License. Insert the
archived release DOI and public repository URL in the camera-ready version.

**Conflict of interest.** The authors must provide the final declaration.

**Funding.** The authors must provide the final declaration.

**Use of AI-assisted tools.** Complete according to the conference and
institutional disclosure policy.

## References

1. D. A. Vallado, P. Crawford, R. Hujsak, and T. S. Kelso, “Revisiting
   Spacetrack Report #3,” *AIAA/AAS Astrodynamics Specialist Conference*,
   AIAA 2006-6753, 2006. https://doi.org/10.2514/6.2006-6753
2. S. Alfano and J. L. Negron, “Determining satellite close approaches, Part
   II,” *Journal of the Astronautical Sciences*, vol. 42, no. 2, pp. 143–152,
   1994.
3. National Aeronautics and Space Administration, *Spaceflight Safety Handbook
   for Satellite Operators*, version 1.5, 2020.
   https://www.nasa.gov/wp-content/uploads/2020/03/spaceflight_safety_handbook_for_operators_v1.5_aug201.pdf
4. J. Davis and M. Goadrich, “The relationship between Precision-Recall and ROC
   curves,” *Proceedings of the 23rd International Conference on Machine
   Learning*, pp. 233–240, 2006. https://doi.org/10.1145/1143844.1143874
5. L. Breiman, “Random forests,” *Machine Learning*, vol. 45, pp. 5–32, 2001.
   https://doi.org/10.1023/A:1010933404324
6. C. Cortes and V. Vapnik, “Support-vector networks,” *Machine Learning*, vol.
   20, pp. 273–297, 1995. https://doi.org/10.1007/BF00994018
7. T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,”
   *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
   Discovery and Data Mining*, pp. 785–794, 2016.
   https://doi.org/10.1145/2939672.2939785
8. G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu,
   “LightGBM: A highly efficient gradient boosting decision tree,” *Advances in
   Neural Information Processing Systems*, vol. 30, pp. 3146–3154, 2017.
