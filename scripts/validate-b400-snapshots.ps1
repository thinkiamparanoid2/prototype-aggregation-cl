param([switch]$PlanOnly)

# One user-owned validation run, not training or a full grid.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv-e1/Scripts/python.exe'
$SourcePath = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/budget_longitudinal_development/2026-09-12_b_frozen_ncm_s0_b400_t010_attempt01'
$EmbeddingPath = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/arm_f_vitb16_trusted_all.npz'
$ValidationPath = Join-Path $ProjectRoot 'ssl_hdc_pilot/trusted_splits/val.csv'
$ResultParent = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/longitudinal_validation_development'
$OutputPath = Join-Path $ResultParent '2026-09-12_b_frozen_ncm_s0_b400_t010_attempt01'
$PlanArguments = @($PythonPath, '-u', '-m', 'ssl_hdc_pilot.scripts.evaluate_snapshots',
    '--source-dir', $SourcePath, '--embeddings', $EmbeddingPath, '--validation-split', $ValidationPath)

Push-Location -LiteralPath $ProjectRoot
try {
    if (-not $PlanOnly -and (Test-Path -LiteralPath $OutputPath)) {
        throw "Attempt already exists. Preserve it and tell me: $OutputPath"
    }
    & rtk proxy @PlanArguments
    if ($LASTEXITCODE -ne 0) { throw 'Snapshot validation input plan failed.' }
    if ($PlanOnly) { return }

    New-Item -ItemType Directory -Force -Path $ResultParent | Out-Null
    $RunLogPath = Join-Path $ResultParent ('preflight_run_' + [guid]::NewGuid().ToString('N') + '.log')
    $TeeParameters = @{ FilePath = $RunLogPath; Append = $true }
    function Invoke-LoggedCommand([string[]]$CommandArguments) {
        & rtk proxy @CommandArguments 2>&1 | Tee-Object @TeeParameters
        if ($LASTEXITCODE -ne 0) { throw "Command failed. Preserve the log: $RunLogPath" }
    }

    Invoke-LoggedCommand @('powershell', '-NoProfile', '-File', 'scripts/hdc.ps1', '-Action', 'env')
    Invoke-LoggedCommand @('nvidia-smi')
    Get-Process | Sort-Object ProcessName | Select-Object ProcessName, Id, CPU |
        Format-Table -AutoSize | Out-String -Width 220 | Tee-Object @TeeParameters
    Write-Host 'Review preflight. Close competing GPU workloads, games, browser/video and recording/overlay apps.'
    $Confirmation = Read-Host 'When the machine is clear, type RUN SNAPSHOT VAL'
    if ($Confirmation -cne 'RUN SNAPSHOT VAL') { throw 'Cancelled. No validation scoring launched.' }
    Write-Output ('User confirmed RUN SNAPSHOT VAL at ' + (Get-Date -Format o)) | Tee-Object @TeeParameters

    $RunArguments = $PlanArguments + @('--output-dir', $OutputPath, '--execute', '--preflight-confirmed')
    Invoke-LoggedCommand $RunArguments
    Write-Host "Report: $OutputPath/integrity_report.json"
    Write-Host "Metrics: $OutputPath/metrics.json"
    Write-Host "Log: $RunLogPath"
    Write-Host 'Done. Tell me done; I will inspect the saved files directly.'
} finally {
    Pop-Location
}
