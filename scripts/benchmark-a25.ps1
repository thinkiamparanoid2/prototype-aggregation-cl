param([switch]$PlanOnly, [ValidateRange(1, 9999)][int]$Attempt = 2)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv-e1/Scripts/python.exe'
$ResultParent = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/benchmark_development'
$OutputPath = Join-Path $ResultParent ('2026-09-13_a_frozen_s0_b25_attempt{0:D2}' -f $Attempt)
$EmbeddingPath = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/arm_f_vitb16_trusted_all.npz'
$TrainPath = Join-Path $ProjectRoot 'ssl_hdc_pilot/trusted_splits/train.csv'
$PlanArguments = @($PythonPath, '-u', '-m', 'ssl_hdc_pilot.scripts.benchmark_study',
    '--experiment', 'A', '--budget', '25', '--seed', '0', '--embeddings', $EmbeddingPath,
    '--trusted-split', $TrainPath)
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
            # Windows PowerShell turns native stderr into ErrorRecords. Drain
            # and stringify them before checking exit status; retain full traces.
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
    # Reject current blockers before RUN confirmation / attempt creation.
    Invoke-LoggedCommand @($PythonPath, '-u', '-m', 'ssl_hdc_pilot.scripts.benchmark_environment')
    Write-Host 'Six learners, ONE frozen-B/seed-0/A25 development condition; not a full grid.'
    Write-Host 'Includes new reference/warm-up/update passes, five timing blocks and separate memory/inventory passes.'
    Write-Host 'Avoid games, browsers, Overwolf, recording apps, NVIDIA overlay and other Python training workloads.'
    Write-Host 'Approved Windows desktop processes, including Settings, plus Terminal/VS Code are logged as a shared baseline.'
    Write-Host 'Unclassified background GPU entries produce logged warnings, not automatic aborts or isolated-machine evidence.'
    Write-Host "Fresh attempt: $OutputPath"
    Write-Host 'Known workload exclusions or integrity mismatches stop the attempt; nothing is deleted.'
    $Confirmation = Read-Host 'After reviewing preflight, type RUN BENCH A25'
    if ($Confirmation -cne 'RUN BENCH A25') { throw 'Cancelled. No benchmark launched.' }
    Write-Output ('User confirmed RUN BENCH A25 at ' + (Get-Date -Format o)) | Tee-Object @TeeParameters
    $RunArguments = $PlanArguments + @('--output-dir', $OutputPath, '--execute', '--preflight-confirmed')
    Invoke-LoggedCommand $RunArguments
    Write-Host "Report: $OutputPath/integrity_report.json"
    Write-Host "Summary: $OutputPath/summary.json"
    Write-Host "Log: $RunLogPath"
    Write-Host 'Done. Tell me done; I will inspect the saved files directly.'
} finally {
    Pop-Location
}
