# LEO Conjunction Lab

Offline-first simulation GUI for IAC-26-A6.IP.6.  The interface separates an
illustrative archived encounter replay from the manuscript's held-out aggregate
results.

## Start

From the repository root:

```powershell
.\gui\start_gui.ps1
```

Then open <http://localhost:8000>.  No internet connection or JavaScript build
step is required.

If port 8000 is already in use, choose another port:

```powershell
.\gui\start_gui.ps1 -Port 8001
```

If PowerShell script execution is disabled:

```powershell
python -m http.server 8000 --directory gui
```

## Controls

- **Play / pause:** use the button or press Space.
- **Step:** use the minus/plus controls or the left/right arrow keys.
- **Orbit / Encounter plane:** switch between the schematic orbit replay and
  the saved RIC geometry at TCA.
- **Results:** compare PR-AUC, F1 and recall from the final manuscript.
- **Quality:** inspect the frozen acquisition-gate failures.
- **Poster mode:** hides secondary controls and enlarges the simulation for
  conference display or screen recording.

## Refresh the local payload

The included payload uses a real-object validation snapshot with a saved
propagation series.  Regenerate it with:

```powershell
python src/export_gui_data.py `
  --conjunctions outputs/catalog_75_validation_pipeline/identified_conjunctions.csv `
  --distance-series outputs/catalog_75_validation_pipeline/top_pair_distance_timeseries.csv `
  --output gui/data/simulation.json
```

`--tle <path>` can replace `--distance-series` when the local Python environment
has the project's Skyfield dependencies; the exporter will recompute the first
selected pair's separation series from the TLE file.

## Scientific boundary

The centre animation is a schematic replay and is explicitly marked not to
scale.  Numeric encounter fields come from the selected local row; manuscript
model metrics and acquisition gates are a separate paper aggregate.  Neither
`risk_score` nor the `proxy_positive` label is a probability of collision:
public TLEs do not provide the covariance and hard-body information required
for operational Pc.
