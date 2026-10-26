# Runs the whole three-dataset replication after data prep: (a) all 9
# adaptation arms, (b) the 270 learner cells, (c) aggregation/comparison
# including the original dataset. Stops at the first failure; rerun this
# same script to resume -- every stage below is independently resumable
# (adapt-all skips finished arms/epochs; run_batch skips finished cells and
# quarantines/retries a crashed one up to three attempts; aggregate.py is
# idempotent and safe to rerun on partial results at any time).
param(
    [int]$Workers = 2
)
$ErrorActionPreference = 'Continue'  # PS 5.1 treats native stderr (progress bars) as errors under 'Stop'; failures are caught via $LASTEXITCODE
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = '.\.venv-e1\Scripts\python.exe'

function Banner([string]$Stage) {
    Write-Host "=== $Stage  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" -ForegroundColor Cyan
}

Banner 'Stage 1/3: adaptation (9 arms)'
& (Join-Path $PSScriptRoot 'replication-adapt-all.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Stage 1 (adaptation) failed. Rerun this script to resume.' }

Banner 'Stage 2/3: learner cells (270 cells, run_batch.py)'
& $py -m ssl_hdc_pilot.replication.run_batch --workers $Workers --require-adapted
if ($LASTEXITCODE -ne 0) { throw 'Stage 2 (run_batch) failed. Rerun this script to resume; finished cells are skipped.' }

Banner 'Stage 3/3: aggregation and comparison'
& $py -m ssl_hdc_pilot.replication.aggregate
if ($LASTEXITCODE -ne 0) { throw 'Stage 3 (aggregate) failed. Rerun this script to resume.' }

Banner 'ALL STAGES DONE'
