param(
    [switch]$PlanOnly,
    [ValidateSet('1','2','3','all')][string]$Phase = 'all',
    [int]$StartAt = 1,
    [int]$MaxCells = 0
)

# User-owned unattended mechanism-ablation batch. Never delete or resume a partial attempt.
# Mirrors scripts/run-forgetting-grid-all.ps1 (36/36 cells, 72/72 PASS receipts), changed only
# where the ablation differs: three phases instead of one grid, a phase-filtered manifest, and
# an optional --hypervector-dim on Phase-3 capture commands.
#
# Run order is Phase 2 -> Phase 1 -> Phase 3 (see -Phase 'all' below). Phase 2 (Sign-NCM /
# Median-NCM, projection-free) is the decisive phase: it determines whether HDC's collapse
# resistance survives without any hyperdimensional projection, which gates
# CODEX_DATASET_REPLICATION_PROMPT.md and CODEX_OPTION_C_CORRUPTION_PROMPT.md. Running it first
# means that if the batch is interrupted, the decisive result is already in hand. See
# ssl_hdc_pilot/results/mechanism_ablation/ABLATION_PLAN.md for the full rationale and the
# 45/30/45 = 120 cell-count reconciliation.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv-e1/Scripts/python.exe'
$AblationParent = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/mechanism_ablation'
$CaptureParent = Join-Path $AblationParent 'capture'
$ValidationParent = Join-Path $AblationParent 'validation'
$ProgressPath = Join-Path $AblationParent 'batch_progress.jsonl'

# Five registered ablation arms, in a fixed (not hashtable-random) order; filenames match the
# single-cell prober scripts/run-mechanism-ablation.ps1.
$ArmOrder = @('s_frozen', 'b_frozen', 'l_frozen', 'b_adapted', 'l_adapted')
$ArmFiles = @{
    s_frozen  = 'arm_f_trusted_all.npz'
    b_frozen  = 'arm_f_vitb16_trusted_all.npz'
    l_frozen  = 'arm_f_vitl16_trusted_all.npz'
    b_adapted = 'arm_d_vitb16_ep3_trusted_all.npz'
    l_adapted = 'arm_d_vitl16_ep4_trusted_all.npz'
}
$Phase3Dims = @(1000, 2000, 10000)
# Working estimate from ABLATION_PLAN.md (observed on the archived grid); used only until this
# batch has its own measured per-cell timings in batch_progress.jsonl.
$DefaultSecondsPerCell = 44.0

function Get-PackageLearners {
    # Learner names come from the installed package, not a literal list in this file, so a
    # rename in ssl_hdc_pilot.scripts.stream_runner is caught here rather than silently drifting.
    $json = & $PythonPath -c "import json; from ssl_hdc_pilot.scripts import stream_runner as r; print(json.dumps({'canonical': list(r.LEARNER_NAMES), 'ablation': list(r.ABLATION_LEARNER_NAMES)}))"
    if ($LASTEXITCODE -ne 0) { throw 'Failed to read learner names from the installed package.' }
    return $json | ConvertFrom-Json
}

Push-Location -LiteralPath $ProjectRoot
try {
    $Learners = Get-PackageLearners
    # Phase composition is an ABLATION_PLAN.md decision (which subset of the installed
    # registries this batch exercises); the registries themselves are read from the package.
    $Phase1Learners = @('hdc', 'ncm', 'ranpac')
    foreach ($l in $Phase1Learners) {
        if ($Learners.canonical -notcontains $l) { throw "Phase-1 learner '$l' is missing from the installed canonical registry." }
    }
    $Phase2Learners = @($Learners.ablation)
    if (@($Phase2Learners).Count -eq 0) { throw 'Installed package reports no ablation learners.' }
    $Phase3Learner = 'hdc'
    if ($Learners.canonical -notcontains $Phase3Learner) { throw "Phase-3 learner '$Phase3Learner' is missing from the installed canonical registry." }

    function Get-Cells {
        # Global order is fixed: Phase 2 (30) -> Phase 1 (45) -> Phase 3 (45) = 120.
        $out = @()
        foreach ($learner in $Phase2Learners) { foreach ($arm in $ArmOrder) { foreach ($seed in 0..2) {
            $out += [pscustomobject]@{ Phase = 2; Arm = $arm; Learner = $learner; Seed = $seed; Dim = 0 }
        } } }
        foreach ($learner in $Phase1Learners) { foreach ($arm in $ArmOrder) { foreach ($seed in 0..2) {
            $out += [pscustomobject]@{ Phase = 1; Arm = $arm; Learner = $learner; Seed = $seed; Dim = 0 }
        } } }
        foreach ($dim in $Phase3Dims) { foreach ($arm in $ArmOrder) { foreach ($seed in 0..2) {
            $out += [pscustomobject]@{ Phase = 3; Arm = $arm; Learner = $Phase3Learner; Seed = $seed; Dim = $dim }
        } } }
        return $out
    }
    function BaseName($c) {
        if ($c.Dim -gt 0) {
            '2026-09-15_ablation_b400_p{0}_{1}_{2}_d{3}_s{4}_t010' -f $c.Phase, $c.Arm, $c.Learner, $c.Dim, $c.Seed
        } else {
            '2026-09-15_ablation_b400_p{0}_{1}_{2}_s{3}_t010' -f $c.Phase, $c.Arm, $c.Learner, $c.Seed
        }
    }
    function ReceiptComplete($dir) {
        $receipt = Join-Path $dir 'integrity_report.json'; if (-not (Test-Path -LiteralPath $receipt)) { return $false }
        try { $r = Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json } catch { return $false }
        if ($r.status -ne 'PASS' -or -not $r.artifacts) { return $false }
        foreach ($p in $r.artifacts.psobject.Properties) { if (-not (Test-Path -LiteralPath (Join-Path $dir $p.Name))) { return $false } }
        return $true
    }
    function NextAttempt($cell) {
        $base = BaseName $cell; $items = @(Get-ChildItem -LiteralPath $CaptureParent -Directory -Filter ($base + '_attempt*') -ErrorAction SilentlyContinue)
        foreach ($item in $items) { $evaluation = Join-Path $ValidationParent $item.Name; if ((ReceiptComplete $item.FullName) -and (ReceiptComplete $evaluation)) { return [int]($item.Name -replace '.*_attempt', '') } }
        foreach ($item in $items) { $evaluation = Join-Path $ValidationParent $item.Name; if ((ReceiptComplete $item.FullName) -and -not (Test-Path -LiteralPath $evaluation)) { return [int]($item.Name -replace '.*_attempt', '') } }
        if ($items.Count -ge 3) { throw "Attempt cap reached for ${base}: three retained incomplete attempts require inspection." }
        return $items.Count + 1
    }
    function Paths($cell, $attempt) { $n = (BaseName $cell) + '_attempt{0:D2}' -f $attempt; return @{ Capture = (Join-Path $CaptureParent $n); Evaluation = (Join-Path $ValidationParent $n); Name = $n } }
    function Add-Progress($record) { ($record | ConvertTo-Json -Compress) | Add-Content -LiteralPath $ProgressPath -Encoding utf8; }
    function Invoke-Logged($CommandArgs, $tee) {
        # $args is a PowerShell automatic variable and must not be used as a parameter name:
        # @args would splat the empty automatic one instead of the passed array.
        # $ErrorActionPreference is relaxed around the native call because in Windows PowerShell
        # 5.1 2>&1 wraps native stderr lines in NativeCommandError, which would terminate on
        # EAP=Stop even when the process exits 0. Exit code is checked explicitly instead.
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & rtk proxy @CommandArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object @tee
            $code = $LASTEXITCODE
        } finally { $ErrorActionPreference = $previous }
        if ($code -ne 0) { throw "Native command failed (exit $code): $($CommandArgs -join ' ')" }
    }

    $cells = Get-Cells
    $selectedAll = @($cells | Where-Object { $Phase -eq 'all' -or $_.Phase -eq [int]$Phase })
    $selected = @($selectedAll | Select-Object -Skip ($StartAt - 1)); if ($MaxCells -gt 0) { $selected = @($selected | Select-Object -First $MaxCells) }

    $complete = 0; $partial = 0
    foreach ($c in $cells) {
        $base = BaseName $c
        $dirs = @(Get-ChildItem -LiteralPath $CaptureParent -Directory -Filter ($base + '_attempt*') -ErrorAction SilentlyContinue)
        if (@($dirs | Where-Object { ReceiptComplete $_.FullName }).Count) { $complete++ }
        if ($dirs.Count -and -not @($dirs | Where-Object { ReceiptComplete $_.FullName }).Count) { $partial++ }
    }
    $priorElapsed = @()
    if (Test-Path -LiteralPath $ProgressPath) {
        foreach ($line in Get-Content -LiteralPath $ProgressPath) {
            try { $rec = $line | ConvertFrom-Json } catch { continue }
            if ($rec.status -eq 'COMPLETE' -and $rec.elapsed_seconds) { $priorElapsed += [double]$rec.elapsed_seconds }
        }
    }
    $avgSeconds = if ($priorElapsed.Count) { ($priorElapsed | Measure-Object -Average).Average } else { $DefaultSecondsPerCell }
    $remaining = $cells.Count - $complete
    $estimatedRemainingSeconds = [math]::Round($avgSeconds * $remaining, 0)
    $free = [System.IO.DriveInfo]::new('G').AvailableFreeSpace
    $existingLogs = @(Get-ChildItem -LiteralPath $AblationParent -Filter 'batch_*.log' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime)
    $priorLog = if ($existingLogs.Count) { $existingLogs[-1].Name } else { 'none' }
    $summary = "START complete=$complete remaining=$remaining superseded=$partial free=$free avg_seconds_per_cell=$([math]::Round($avgSeconds,1)) estimated_remaining_seconds=$estimatedRemainingSeconds phase=$Phase phase_order='2,1,3' prior_log=$priorLog"
    Write-Output $summary

    if ($PlanOnly) {
        # Read-only: no directory is created, no log or progress file is written, no learner is
        # constructed. NextAttempt/ReceiptComplete only stat the filesystem.
        $i = 0
        foreach ($c in $cells) {
            $i++
            if ($Phase -ne 'all' -and $c.Phase -ne [int]$Phase) { continue }
            $a = NextAttempt $c
            $p = Paths $c $a
            $status = if ((ReceiptComplete $p.Capture) -and (ReceiptComplete $p.Evaluation)) { 'SKIP' } else { 'RUN' }
            $dimText = if ($c.Dim -gt 0) { "dim=$($c.Dim) " } else { '' }
            "$i phase$($c.Phase) $($c.Arm)/$($c.Learner)/s$($c.Seed) ${dimText}attempt$a status=$status capture=$($p.Capture) evaluation=$($p.Evaluation)"
        }
        return
    }
    if ((Read-Host 'Type RUN ABLATION') -cne 'RUN ABLATION') { throw 'Cancelled before batch start.' }

    New-Item -ItemType Directory -Force -Path $AblationParent, $CaptureParent, $ValidationParent | Out-Null
    $log = Join-Path $AblationParent ('batch_' + [guid]::NewGuid().ToString('N') + '.log'); $tee = @{FilePath = $log; Append = $true}
    $summary | Tee-Object @tee

    $done = 0; $skipped = 0; $started = Get-Date
    foreach ($c in $selected) {
        $index = [array]::IndexOf($cells, $c) + 1
        $a = NextAttempt $c
        $p = Paths $c $a
        if ((ReceiptComplete $p.Capture) -and (ReceiptComplete $p.Evaluation)) { "$index SKIPPED (already complete)" | Tee-Object @tee; $skipped++; continue }
        if ([System.IO.DriveInfo]::new('G').AvailableFreeSpace -lt 8GB) { throw "FAILED cell ${index}: below 8 GiB guard" }
        $cellStart = Get-Date
        $dimText = if ($c.Dim -gt 0) { " dim$($c.Dim)" } else { '' }
        "$index RUN phase$($c.Phase) $($c.Arm)/$($c.Learner)/s$($c.Seed)$dimText attempt$a" | Tee-Object @tee
        try {
            Invoke-Logged @($PythonPath, '-m', 'ssl_hdc_pilot.scripts.benchmark_environment') $tee
            $emb = Join-Path $ProjectRoot ('ssl_hdc_pilot/results/' + $ArmFiles[$c.Arm])
            if (-not (Test-Path $p.Capture)) {
                $captureArgs = @($PythonPath, '-u', '-m', 'ssl_hdc_pilot.scripts.budget_experiment',
                    '--experiment', 'B', '--threshold', '0.10', '--budget', '400',
                    '--learner', $c.Learner, '--seed', "$($c.Seed)", '--embeddings', $emb,
                    '--trusted-split', (Join-Path $ProjectRoot 'ssl_hdc_pilot/trusted_splits/train.csv'),
                    '--output-dir', $p.Capture, '--ablation-mode', '--execute', '--preflight-confirmed')
                if ($c.Dim -gt 0) { $captureArgs += @('--hypervector-dim', "$($c.Dim)") }
                Invoke-Logged $captureArgs $tee
            }
            Invoke-Logged @($PythonPath, '-u', '-m', 'ssl_hdc_pilot.scripts.evaluate_snapshots',
                '--source-dir', $p.Capture, '--embeddings', $emb,
                '--validation-split', (Join-Path $ProjectRoot 'ssl_hdc_pilot/trusted_splits/val.csv'),
                '--output-dir', $p.Evaluation, '--execute', '--preflight-confirmed') $tee
            $elapsed = ((Get-Date) - $cellStart).TotalSeconds
            $done++
            Add-Progress @{index = $index; phase = $c.Phase; arm = $c.Arm; learner = $c.Learner; seed = $c.Seed; dim = $c.Dim; attempt = $a; status = 'COMPLETE'; timestamp = (Get-Date -Format o); elapsed_seconds = [math]::Round($elapsed, 1)}
            "$index COMPLETE elapsed=$((Get-Date)-$cellStart)" | Tee-Object @tee
        } catch {
            Add-Progress @{index = $index; phase = $c.Phase; arm = $c.Arm; learner = $c.Learner; seed = $c.Seed; dim = $c.Dim; attempt = $a; status = 'FAILED'; timestamp = (Get-Date -Format o); error = $_.Exception.Message}
            "FAILED cell $index command=$($_.InvocationInfo.Line) error=$($_.Exception.Message)" | Tee-Object @tee
            throw
        }
    }
    "SUMMARY completed=$done skipped=$skipped remaining=$($selected.Count-$done-$skipped) elapsed=$((Get-Date)-$started) free=$([System.IO.DriveInfo]::new('G').AvailableFreeSpace) resume=powershell -NoProfile -File .\scripts\run-mechanism-ablation-all.ps1" | Tee-Object @tee
} finally { Pop-Location }
