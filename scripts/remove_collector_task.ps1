$ErrorActionPreference = "Stop"
Unregister-ScheduledTask -TaskName "SpaceDebrisCollector60Days" -Confirm:$false
Write-Host "Removed task SpaceDebrisCollector60Days"
