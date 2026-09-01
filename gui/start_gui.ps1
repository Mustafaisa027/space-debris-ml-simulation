param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$guiRoot = $PSScriptRoot
$projectRoot = Split-Path -Parent $guiRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$codexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
$candidates = @(
    $pythonCommand.Source,
    $pyLauncher.Source,
    $venvPython,
    $codexPython
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

$python = $null
foreach ($candidate in $candidates) {
    try {
        & $candidate --version *> $null
        if ($LASTEXITCODE -eq 0) {
            $python = $candidate
            break
        }
    } catch {
        continue
    }
}

if (-not $python) {
    throw "Python was not found. Install Python 3 or run: python -m http.server 8000 -d gui"
}

Write-Host "LEO Conjunction Lab is starting..." -ForegroundColor Cyan
Write-Host "Open http://localhost:$Port in your browser." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the local server." -ForegroundColor DarkGray

& $python -m http.server $Port --directory $guiRoot
