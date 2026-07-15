$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

.\.venv\Scripts\python.exe src\collect_observations.py `
  --days 60 `
  --interval-hours 2 `
  --max-objects 75 `
  --history outputs\history\conjunction_observations_v2.csv
