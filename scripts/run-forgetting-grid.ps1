param(
    [switch]$PlanOnly,
    [Parameter(Mandatory = $true)][string]$Learner,
    [Parameter(Mandatory = $true)][ValidateSet('frozen', 'adapted')][string]$Arm,
    [Parameter(Mandatory = $true)][ValidateRange(0, 2)][int]$Seed
)

# One user-owned B400 capture plus validation-only snapshot scoring.  This is
# deliberately one cell per invocation: it never batches the comparative grid.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv-e1/Scripts/python.exe'
$TrainPath = Join-Path $ProjectRoot 'ssl_hdc_pilot/trusted_splits/train.csv'
$ValidationPath = Join-Path $ProjectRoot 'ssl_hdc_pilot/trusted_splits/val.csv'
$CaptureParent = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/budget_longitudinal_development'
$ValidationParent = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/longitudinal_validation_development'
$EmbeddingName = if ($Arm -eq 'frozen') { 'arm_f_vitb16_trusted_all.npz' } else { 'arm_d_vitb16_ep3_trusted_all.npz' }
$EmbeddingPath = Join-Path $ProjectRoot ('ssl_hdc_pilot/results/' + $EmbeddingName)
$CellName = ('2026-09-15_grid_b400_{0}_{1}_s{2}_t010_attempt01' -f $Arm, $Learner, $Seed)
$CapturePath = Join-Path $CaptureParent $CellName
$ValidationOutputPath = Join-Path $ValidationParent $CellName
$CompletedCapture = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/budget_longitudinal_development/2026-09-12_b_frozen_ncm_s0_b400_t010_attempt01'
$CompletedValidation = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/longitudinal_validation_development/2026-09-12_b_frozen_ncm_s0_b400_t010_attempt01'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'

function Invoke-LoggedCommand([string[]]$CommandArguments, [hashtable]$TeeParameters) {
    & rtk proxy @CommandArguments 2>&1 | Tee-Object @TeeParameters
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE). Preserve the log." }
}

Push-Location -LiteralPath $ProjectRoot
try {
    # Names are read from the installed project rather than copied from this launcher.
    $KnownLearners = @(& rtk proxy $PythonPath -c 'from ssl_hdc_pilot.scripts.stream_runner import LEARNER_NAMES; print(*LEARNER_NAMES, sep=chr(10))')
    if ($LASTEXITCODE -ne 0 -or $Learner -notin $KnownLearners) {
        throw "Unknown learner '$Learner'. Registered learners: $($KnownLearners -join ', ')"
    }
    if ((Test-Path -LiteralPath $CapturePath) -or (Test-Path -LiteralPath $ValidationOutputPath)) {
        throw "Target already exists. Preserve it; no overwrite, retry, or resume: $CellName"
    }

    $CapturePlan = @($PythonPath, '-u', '-m', 'ssl_hdc_pilot.scripts.budget_experiment',
        '--experiment', 'B', '--threshold', '0.10', '--budget', '400', '--learner', $Learner,
        '--seed', "$Seed", '--embeddings', $EmbeddingPath, '--trusted-split', $TrainPath)
    & rtk proxy @CapturePlan
    if ($LASTEXITCODE -ne 0) { throw 'Capture input plan failed; no learner was constructed.' }
    if ($PlanOnly) {
        Write-Host "PASS capture plan: $CellName"
        Write-Host 'Validation plan is intentionally deferred until this fresh capture exists.'
        return
    }

    $Drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($CaptureParent))
    if ($Drive.AvailableFreeSpace -lt 8GB) {
        throw "Less than the required 8 GiB prelaunch headroom remains on $($Drive.Name). No cell launched."
    }
    New-Item -ItemType Directory -Force -Path $CaptureParent, $ValidationParent | Out-Null
    $RunLogPath = Join-Path $CaptureParent ('preflight_run_' + [guid]::NewGuid().ToString('N') + '.log')
    $TeeParameters = @{ FilePath = $RunLogPath; Append = $true }
    Invoke-LoggedCommand @('powershell', '-NoProfile', '-File', 'scripts/hdc.ps1', '-Action', 'env') $TeeParameters
    Invoke-LoggedCommand @('nvidia-smi') $TeeParameters
    & rtk proxy $PythonPath -m ssl_hdc_pilot.scripts.benchmark_environment 2>&1 | Tee-Object @TeeParameters
    if ($LASTEXITCODE -ne 0) { throw "Workload preflight failed. Preserve the log: $RunLogPath" }
    Get-Process | Sort-Object ProcessName | Select-Object ProcessName, Id, CPU |
        Format-Table -AutoSize | Out-String -Width 220 | Tee-Object @TeeParameters
    $Confirmation = Read-Host 'After reviewing preflight, type RUN FORGETTING GRID'
    if ($Confirmation -cne 'RUN FORGETTING GRID') { throw 'Cancelled. No capture or validation scoring launched.' }
    Write-Output ('User confirmed RUN FORGETTING GRID at ' + (Get-Date -Format o)) | Tee-Object @TeeParameters

    $CaptureRun = $CapturePlan + @('--output-dir', $CapturePath, '--execute', '--preflight-confirmed')
    Invoke-LoggedCommand $CaptureRun $TeeParameters
    $ValidationPlan = @($PythonPath, '-u', '-m', 'ssl_hdc_pilot.scripts.evaluate_snapshots',
        '--source-dir', $CapturePath, '--embeddings', $EmbeddingPath, '--validation-split', $ValidationPath)
    & rtk proxy @ValidationPlan 2>&1 | Tee-Object @TeeParameters
    if ($LASTEXITCODE -ne 0) { throw "Snapshot-validation input plan failed. Preserve the capture and log: $RunLogPath" }
    $ValidationRun = $ValidationPlan + @('--output-dir', $ValidationOutputPath, '--execute', '--preflight-confirmed')
    Invoke-LoggedCommand $ValidationRun $TeeParameters
    Write-Host "Capture receipt: $CapturePath/integrity_report.json"
    Write-Host "Validation receipt: $ValidationOutputPath/integrity_report.json"
    Write-Host "Log: $RunLogPath"
} finally {
    Pop-Location
}
