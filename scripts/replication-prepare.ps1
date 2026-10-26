param(
    [ValidateSet('PlantDoc', 'PlantWild', 'PlantVillage', 'all')]
    [string]$Dataset = 'all'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv-e1/Scripts/python.exe'
$PrepareScript = Join-Path $PSScriptRoot 'prepare_acquired_dataset.py'
$env:TMP = Join-Path $ProjectRoot '.codex-tmp'
$env:TEMP = $env:TMP
$env:PYTHONIOENCODING = 'utf-8'
$Names = @('PlantDoc', 'PlantWild', 'PlantVillage')
if ($Dataset -ne 'all') { $Names = @($Dataset) }

foreach ($name in $Names) {
    Write-Output "START dataset=$name"
    & $PythonPath -u $PrepareScript $name
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset preparation failed for $name with exit code $LASTEXITCODE"
    }
    Write-Output "FINISH dataset=$name"
}
