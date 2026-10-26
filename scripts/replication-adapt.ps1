param(
    [ValidateSet('PlantDoc', 'PlantWild', 'PlantVillage', 'all')][string]$Dataset = 'all',
    [ValidateSet('vits16', 'vitb16', 'vitl16', 'all')][string]$Size = 'all',
    [switch]$Execute,
    [string]$Confirm
)

# Adaptation + epoch probe-gate is driven end to end by
# ssl_hdc_pilot.replication.adapt_and_gate, which is resumable (safe to
# re-run after a crash) and refuses to overwrite finished outputs. It picks
# the correct --model-id per encoder size itself; this launcher never
# bypasses that.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv-e1/Scripts/python.exe'
$Datasets = @('PlantDoc', 'PlantWild', 'PlantVillage'); if ($Dataset -ne 'all') { $Datasets = @($Dataset) }
$Sizes = @('vits16', 'vitb16', 'vitl16'); if ($Size -ne 'all') { $Sizes = @($Size) }
foreach ($d in $Datasets) { foreach ($s in $Sizes) {
    $root = Join-Path $ProjectRoot "ssl_hdc_pilot/replication/data/$d"
    "PREPARE dataset=$d size=$s manifest=$root/adaptation_train_manifest.csv epochs=1,2,3,4,5 gate=tolerance:0.005; train-only pixels; output=ssl_hdc_pilot/replication/adapted/$d/$s"
} }
if (-not $Execute) { Write-Output 'Prepare-only: adaptation and probe-gate extraction were not launched. Re-run with -Execute after Gate 1 approval.'; return }
if ($Confirm -cne 'ADAPT REPLICATION') { throw "Cancelled before adaptation: pass -Confirm 'ADAPT REPLICATION' (exact case) to proceed." }
foreach ($d in $Datasets) { foreach ($s in $Sizes) {
    & $PythonPath -m ssl_hdc_pilot.replication.adapt_and_gate --dataset $d --size $s --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Adaptation/probe-gate failed or is blocked for $d/$s; see ssl_hdc_pilot/replication/adapted/$d/$s/receipt.json" }
    Write-Output "Adaptation and probe gate complete for $d/$s."
} }
