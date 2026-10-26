# Runs all 9 adaptation arms in order (resumable: finished arms/steps are skipped).
# Close other apps first. Stops at the first failure; just rerun to resume.
$ErrorActionPreference = 'Continue'  # PS 5.1 treats native stderr (progress bars) as errors under 'Stop'; failures are caught via $LASTEXITCODE
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = '.\.venv-e1\Scripts\python.exe'
New-Item -ItemType Directory -Force .codex-tmp | Out-Null
foreach ($d in 'PlantDoc','PlantWild','PlantVillage') {
  foreach ($s in 'vits16','vitb16','vitl16') {
    Write-Host "=== $d $s  $(Get-Date -Format T) ==="
    & $py -m ssl_hdc_pilot.replication.adapt_and_gate --dataset $d --size $s --device cuda 2>&1 | ForEach-Object { "$_" } | Tee-Object -Append ".codex-tmp\adapt_$d`_$s.log"
    if ($LASTEXITCODE -ne 0) { throw "FAILED $d $s (see .codex-tmp\adapt_$d`_$s.log). Rerun this script to resume." }
  }
}
Write-Host "ALL 9 ARMS DONE $(Get-Date -Format T)"
