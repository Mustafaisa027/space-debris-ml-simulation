$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

.\.venv\Scripts\python.exe src\collect_observations.py `
  --config config\experiment_60_days.json
