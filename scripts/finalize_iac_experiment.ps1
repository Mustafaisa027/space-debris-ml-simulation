[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ArchiveRoot,

    [string]$Config = "config\experiment_60_days.json",

    [ValidateRange(1, 64)]
    [int]$Workers = 4,

    [string]$Python = ".venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$archivePath = (Resolve-Path -LiteralPath $ArchiveRoot).Path
$configPath = (Resolve-Path -LiteralPath (Join-Path $projectRoot $Config)).Path
$pythonPath = if ([System.IO.Path]::IsPathRooted($Python)) {
    (Resolve-Path -LiteralPath $Python).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $projectRoot $Python)).Path
}

Push-Location $projectRoot
try {
    & $pythonPath "src\collection_window_status.py" --config $configPath |
        Tee-Object -Variable windowOutput
    if ($LASTEXITCODE -ne 0) {
        throw "Collection-window check failed with exit code $LASTEXITCODE."
    }
    $window = $windowOutput | Select-Object -Last 1 | ConvertFrom-Json
    if ($window.status -ne "window_complete") {
        throw "Finalization is locked until the frozen collection window is complete (status=$($window.status))."
    }

    & $pythonPath "src\collection_cadence_health.py" $archivePath --config $configPath
    if ($LASTEXITCODE -ne 0) {
        throw "Cadence-health verification failed with exit code $LASTEXITCODE."
    }

    & $pythonPath "src\import_collection_archive.py" $archivePath --config $configPath
    if ($LASTEXITCODE -ne 0) {
        throw "Archive import failed with exit code $LASTEXITCODE."
    }

    & $pythonPath "src\resimulate_snapshots.py" --config $configPath --workers $Workers
    if ($LASTEXITCODE -ne 0) {
        throw "Corrected-TCA resimulation failed with exit code $LASTEXITCODE."
    }

    & $pythonPath "src\train_from_history.py" --config $configPath
    if ($LASTEXITCODE -ne 0) {
        throw "Training/publication validation failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
