param([switch]$PlanOnly, [ValidateRange(1, 9999)][int]$Attempt = 1)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv-e1/Scripts/python.exe'
$ResultParent = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/benchmark_development'
$OutputPath = Join-Path $ResultParent ('2026-09-14_b_frozen_s0_b400_t010_attempt{0:D2}' -f $Attempt)
$EmbeddingPath = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/arm_f_vitb16_trusted_all.npz'
$TrainPath = Join-Path $ProjectRoot 'ssl_hdc_pilot/trusted_splits/train.csv'
$PlanArguments = @($PythonPath, '-u', '-m', 'ssl_hdc_pilot.scripts.benchmark_study',
    '--experiment', 'B', '--budget', '400', '--threshold', '0.10', '--seed', '0',
    '--embeddings', $EmbeddingPath, '--trusted-split', $TrainPath)
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'

Push-Location -LiteralPath $ProjectRoot
try {
    if (-not $PlanOnly -and (Test-Path -LiteralPath $OutputPath)) {
        throw "Attempt already exists. Preserve it and tell me: $OutputPath"
    }
    & rtk proxy @PlanArguments
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark input plan failed.' }
    if ($PlanOnly) { return }
    $TargetDrive = New-Object System.IO.DriveInfo ([System.IO.Path]::GetPathRoot($ResultParent))
    if ($TargetDrive.AvailableFreeSpace -lt 8GB) { throw 'Need 8 GiB operational headroom before this pilot; no run launched.' }
    New-Item -ItemType Directory -Force -Path $ResultParent | Out-Null
    $RunLogPath = Join-Path $ResultParent ('preflight_run_' + [guid]::NewGuid().ToString('N') + '.log')
    $TeeParameters = @{ FilePath = $RunLogPath; Append = $true }
    function Invoke-LoggedCommand([string[]]$CommandArguments) {
        $PreviousErrorPreference = $ErrorActionPreference
        $NativeExitCode = $null
        try {
            # Drain native stderr before checking its exit code (Windows PowerShell).
            $ErrorActionPreference = 'Continue'
            $global:LASTEXITCODE = $null
            & rtk proxy @CommandArguments 2>&1 |
                ForEach-Object { $_.ToString() } -ErrorAction Stop |
                Tee-Object @TeeParameters -ErrorAction Stop
            $NativeExitCode = $global:LASTEXITCODE
        } finally { $ErrorActionPreference = $PreviousErrorPreference }
        if ($null -eq $NativeExitCode -or $NativeExitCode -ne 0) {
            throw "Command failed (exit $NativeExitCode). Preserve outputs and log: $RunLogPath"
        }
    }
    Invoke-LoggedCommand @('powershell', '-NoProfile', '-File', 'scripts/hdc.ps1', '-Action', 'env')
    Invoke-LoggedCommand @('nvidia-smi')
    Get-Process | Sort-Object ProcessName | Select-Object ProcessName, Id, CPU |
        Format-Table -AutoSize | Out-String -Width 220 | Tee-Object @TeeParameters
    Invoke-LoggedCommand @($PythonPath, '-u', '-m', 'ssl_hdc_pilot.scripts.benchmark_environment')
    Write-Host 'Six learners, ONE frozen-B/seed-0/Experiment-B/allowance400/threshold0.10 feasibility condition; not a grid.'
    Write-Host 'Includes new reference/warm-up/update passes, five timing blocks and separate memory/inventory passes.'
    Write-Host 'Up to 400 purchased stream labels per learner; actual consumption may differ by method.'
    Write-Host 'Avoid games, browser/video, recording/overlays and other training workloads.'
    Write-Host 'Background GPU entries may recur: unclassified apps produce logged warnings, not automatic aborts.'
    Write-Host 'Known workload exclusions still block. Warning-bearing timings are shared-desktop observations, not isolated-machine evidence.'
    Write-Host "Fresh attempt: $OutputPath"
    Write-Host 'No user-imposed runtime cap. Disk/preflight/integrity guards remain; no automatic retry or cleanup.'
    $Confirmation = Read-Host 'After reviewing preflight, type RUN BENCH B400'
    if ($Confirmation -cne 'RUN BENCH B400') { throw 'Cancelled. No benchmark launched.' }
    Write-Output ('User confirmed RUN BENCH B400 at ' + (Get-Date -Format o)) | Tee-Object @TeeParameters
    $RunArguments = $PlanArguments + @('--output-dir', $OutputPath, '--execute', '--preflight-confirmed')
    Invoke-LoggedCommand $RunArguments
    Write-Host "Report: $OutputPath/integrity_report.json"
    Write-Host "Summary: $OutputPath/summary.json"
    Write-Host "Log: $RunLogPath"
    Write-Host 'Done. Tell me done; I will inspect the saved files directly.'
} finally {
    Pop-Location
}
