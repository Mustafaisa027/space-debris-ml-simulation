[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ArchiveRoot,

    [string]$Config = "config\experiment_15_days_v5.json",

    [ValidateRange(1, 64)]
    [int]$Workers = 4,

    [string]$Python = ".venv\Scripts\python.exe",

    [string]$Checkpoint
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
$configData = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$checkpointPath = if ($Checkpoint) {
    if ([System.IO.Path]::IsPathRooted($Checkpoint)) {
        [System.IO.Path]::GetFullPath($Checkpoint)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Checkpoint))
    }
} else {
    Join-Path $projectRoot "outputs\history\finalization_checkpoint_$($configData.experiment_id).json"
}

function Resolve-ProjectOutput {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Path))
}

function Test-CheckpointStage {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string[]]$Outputs
    )
    $arguments = @(
        "src\finalization_checkpoint.py", "check",
        "--config", $configPath,
        "--checkpoint", $checkpointPath,
        "--stage", $Stage
    )
    foreach ($outputPath in $Outputs) {
        $arguments += @("--output", $outputPath)
    }
    $result = & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Checkpoint verification failed for stage '$Stage'."
    }
    return [bool](($result | Select-Object -Last 1 | ConvertFrom-Json).reusable)
}

function Set-CheckpointStage {
    param(
        [Parameter(Mandatory = $true)][string]$Stage,
        [Parameter(Mandatory = $true)][string[]]$Outputs
    )
    $arguments = @(
        "src\finalization_checkpoint.py", "mark",
        "--config", $configPath,
        "--checkpoint", $checkpointPath,
        "--stage", $Stage
    )
    foreach ($outputPath in $Outputs) {
        $arguments += @("--output", $outputPath)
    }
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot record checkpoint stage '$Stage'."
    }
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

    & $pythonPath "src\finalization_checkpoint.py" prepare `
        --config $configPath --checkpoint $checkpointPath --archive-root $archivePath
    if ($LASTEXITCODE -ne 0) {
        throw "Finalization checkpoint preparation failed with exit code $LASTEXITCODE."
    }

    $importOutputs = @(
        (Resolve-ProjectOutput $configData.outputs.snapshots),
        (Resolve-ProjectOutput $configData.outputs.runs),
        (Resolve-ProjectOutput $configData.outputs.history),
        (Resolve-ProjectOutput $configData.outputs.archive_import_report)
    )
    if (Test-CheckpointStage "import" $importOutputs) {
        Write-Host "Reusing fingerprint-verified archive import stage."
    } else {
        & $pythonPath "src\import_collection_archive.py" $archivePath --config $configPath
        if ($LASTEXITCODE -ne 0) {
            throw "Archive import failed with exit code $LASTEXITCODE."
        }
        Set-CheckpointStage "import" $importOutputs
    }

    $resimulationOutputs = @(
        (Resolve-ProjectOutput $configData.outputs.resimulated_runs),
        (Resolve-ProjectOutput $configData.outputs.resimulated_history),
        (Resolve-ProjectOutput $configData.outputs.resimulation_report)
    )
    if (Test-CheckpointStage "resimulation" $resimulationOutputs) {
        Write-Host "Reusing fingerprint-verified corrected-TCA resimulation stage."
    } else {
        & $pythonPath "src\resimulate_snapshots.py" --config $configPath --workers $Workers
        if ($LASTEXITCODE -ne 0) {
            throw "Corrected-TCA resimulation failed with exit code $LASTEXITCODE."
        }
        Set-CheckpointStage "resimulation" $resimulationOutputs
    }

    & $pythonPath "src\train_from_history.py" --config $configPath
    if ($LASTEXITCODE -ne 0) {
        throw "Training/publication validation failed with exit code $LASTEXITCODE."
    }
    Set-CheckpointStage "training" @(
        (Resolve-ProjectOutput $configData.outputs.time_split_report)
    )
} finally {
    Pop-Location
}
