param(
    [ValidateSet('PlantDoc', 'PlantWild', 'PlantVillage', 'all')][string]$Dataset = 'all',
    [ValidateSet('vits16', 'vitb16', 'vitl16', 'all')][string]$Size = 'all',
    [switch]$Execute
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv-e1/Scripts/python.exe'
$Datasets = @('PlantDoc', 'PlantWild', 'PlantVillage'); if ($Dataset -ne 'all') { $Datasets = @($Dataset) }
$Sizes = @('vits16', 'vitb16', 'vitl16'); if ($Size -ne 'all') { $Sizes = @($Size) }
$ModelIds = @{ vits16='facebook/dinov3-vits16-pretrain-lvd1689m'; vitb16='facebook/dinov3-vitb16-pretrain-lvd1689m'; vitl16='facebook/dinov3-vitl16-pretrain-lvd1689m' }
foreach ($d in $Datasets) { foreach ($s in $Sizes) {
    $root = Join-Path $ProjectRoot "ssl_hdc_pilot/replication/data/$d"
    $out = Join-Path $ProjectRoot "ssl_hdc_pilot/replication/embeddings/$d/frozen_$s.npz"
    "PREPARE dataset=$d size=$s model=$($ModelIds[$s]) input=$root/splits/all.csv output=$out"
} }
if (-not $Execute) { Write-Output 'Prepare-only: frozen extraction was not launched. Re-run with -Execute after Gate 1 approval.'; return }
if ((Read-Host 'Type EXTRACT REPLICATION') -cne 'EXTRACT REPLICATION') { throw 'Cancelled before extraction.' }
foreach ($d in $Datasets) { foreach ($s in $Sizes) {
    $root = Join-Path $ProjectRoot "ssl_hdc_pilot/replication/data/$d"
    $input = Join-Path $root 'splits/all.csv'; if (-not (Test-Path -LiteralPath $input)) { throw "Missing approved split manifest: $input" }
    $out = Join-Path $ProjectRoot "ssl_hdc_pilot/replication/embeddings/$d/frozen_$s.npz"
    if (Test-Path -LiteralPath $out) { throw "Refusing to overwrite embedding: $out" }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $out) | Out-Null
    & $PythonPath -m ssl_hdc_pilot.scripts.extract_embeddings --input-csv $input --output-npz $out --metadata-json ($out -replace '\\.npz$','.json') --model-id $ModelIds[$s] --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Extraction failed for $d/$s" }
} }
