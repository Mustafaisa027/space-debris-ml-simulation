$ErrorActionPreference = "Stop"

# Re-registers the 60-day collector Task Scheduler job so it explicitly uses
# the leo_mixed preset (ROADMAP_YOL1.md GOREV 2): a curated CATNR catalog
# spanning multiple orbital planes, fetched one object at a time. This avoids
# the GROUP=STARLINK endpoint, which has been observed to return HTTP 403.
# collect_observations.py now falls back to leo_mixed by default even without
# --preset, but it is passed explicitly here so the scheduled task definition
# stays self-documenting.

$Root = Split-Path -Parent $PSScriptRoot
$TaskName = "SpaceDebrisCollector60Days"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "src\collect_observations.py"
$Start = Get-Date
$End = $Start.AddDays(60)

Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue | Unregister-ScheduledTask -Confirm:$false

$Action = New-ScheduledTaskAction `
  -Execute $Python `
  -Argument "`"$Script`" --once --preset leo_mixed --history `"outputs\history\conjunction_observations.csv`"" `
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

Write-Host "Re-registered task $TaskName with --preset leo_mixed, from $Start to $End"
