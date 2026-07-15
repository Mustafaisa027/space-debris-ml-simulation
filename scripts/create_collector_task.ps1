$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$TaskName = "SpaceDebrisCollector60Days"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "src\collect_observations.py"
$Start = Get-Date
$End = $Start.AddDays(60)

$Action = New-ScheduledTaskAction `
  -Execute $Python `
  -Argument "`"$Script`" --once --history `"outputs\history\conjunction_observations_v2.csv`"" `
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
  -Description "Collect CelesTrak TLE snapshots and conjunction observations every 2 hours for 60 days." `
  -Force

Write-Host "Registered task $TaskName from $Start to $End"
