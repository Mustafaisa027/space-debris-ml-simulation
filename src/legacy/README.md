# Legacy step-by-step prototypes

These `step1..step10` scripts are the original exploratory prototypes that were
written while building the project up one concept at a time (read one TLE,
propagate it, compute a pair distance, find the CPA, add a risk score, etc.).

**They are kept for reference only and are not part of the supported pipeline.**

Known limitations (do not fix these here — use the package instead):

- Hard-coded Windows-style paths (`r"outputs\pair_distance.csv"`) that create a
  single back-slash file name on Linux/macOS instead of a folder.
- Hard-coded, years-old TLEs embedded directly in the source.
- `step9` and `step10` are near-duplicates and depend on a `data/tle_*.txt`
  snapshot that the scripts themselves never create.
- Distance thresholds in the thousands of kilometres, which are not physically
  meaningful for conjunction screening.
- No empty-input guards (`step7` raises `IndexError` on an empty CSV).

The maintained, cross-platform, physically-calibrated implementation lives in
`src/space_debris/` and is driven by:

```bash
python src/run_pipeline.py --tle data/demo_tles.txt
```

If you want the incremental teaching narrative that these scripts provided,
prefer reading `src/space_debris/core.py`, which implements the same steps
(propagation -> closest approach -> relative velocity -> risk) in one place.
