$ErrorActionPreference = "Stop"

# Re-registers the 60-day collector Task Scheduler job using the authoritative
# experiment JSON. Its leo_mixed preset is a curated CATNR catalog
# spanning multiple orbital planes, fetched one object at a time. This avoids
# the GROUP=STARLINK endpoint, which has been observed to return HTTP 403.
# The preset, thresholds, cadence and v3 history path stay synchronized through
# config/experiment_60_days.json.

$Root = Split-Path -Parent $PSScriptRoot
$TaskName = "SpaceDebrisCollector60Days"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "src\collect_observations.py"
$Start = Get-Date
$End = $Start.AddDays(60)

Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue | Unregister-ScheduledTask -Confirm:$false

$Action = New-ScheduledTaskAction `
  -Execute $Python `
  -Argument "`"$Script`" --once --config `"config\experiment_60_days.json`"" `
  -WorkingDirectory $Root

$Trigger = New-ScheduledTaskTrigger `
  -Once `
  -At $Start.AddMinutes(1) `
  -RepetitionInterval (New-TimeSpan -Hours 2) `
  -RepetitionDuration (New-TimeSpan -Days 60)

$Settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -MultipleInstances IgnoreNew

Register-ScheduledTask `
  -TaskName $TaskName `
  -Action $Action `
  -Trigger $Trigger `
  -Settings $Settings `
  -Description "Collect CelesTrak TLE snapshots (leo_mixed preset) and conjunction observations every 2 hours for 60 days." `
  -Force

Write-Host "Re-registered task $TaskName with the authoritative experiment config, from $Start to $End"
