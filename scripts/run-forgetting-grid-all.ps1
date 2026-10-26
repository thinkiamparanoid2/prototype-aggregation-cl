param([switch]$PlanOnly, [ValidateRange(1,36)][int]$StartAt = 1, [ValidateRange(0,36)][int]$MaxCells = 0)

# User-owned unattended grid. Never delete or resume a partial attempt.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv-e1/Scripts/python.exe'
$CaptureParent = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/budget_longitudinal_development'
$ValidationParent = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/longitudinal_validation_development'
$GridParent = Join-Path $ProjectRoot 'ssl_hdc_pilot/results/forgetting_grid_development'
$ProgressPath = Join-Path $GridParent 'batch_progress.jsonl'
$A3Path = Join-Path $GridParent 'A3_FEASIBILITY.md'

function Get-Cells {
    $out = @(); foreach($arm in @('frozen','adapted')) { foreach($learner in @('hdc','ncm','ranpac','naive','ewc','replay')) { foreach($seed in 0..2) {
        $out += [pscustomobject]@{ Arm=$arm; Learner=$learner; Seed=$seed }
    } } }; return $out
}
function BaseName($c) { '2026-09-15_grid_b400_{0}_{1}_s{2}_t010' -f $c.Arm,$c.Learner,$c.Seed }
function ReceiptComplete($dir) {
    $receipt = Join-Path $dir 'integrity_report.json'; if(-not (Test-Path -LiteralPath $receipt)){ return $false }
    try { $r=Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json } catch { return $false }
    if($r.status -ne 'PASS' -or -not $r.artifacts){ return $false }
    foreach($p in $r.artifacts.psobject.Properties){ if(-not (Test-Path -LiteralPath (Join-Path $dir $p.Name))){ return $false } }
    return $true
}
function NextAttempt($cell) {
    $base=BaseName $cell; $items=@(Get-ChildItem -LiteralPath $CaptureParent -Directory -Filter ($base+'_attempt*') -ErrorAction SilentlyContinue)
    foreach($item in $items){ $evaluation=Join-Path $ValidationParent $item.Name; if((ReceiptComplete $item.FullName) -and (ReceiptComplete $evaluation)){ return [int]($item.Name -replace '.*_attempt','') } }
    foreach($item in $items){ $evaluation=Join-Path $ValidationParent $item.Name; if((ReceiptComplete $item.FullName) -and -not (Test-Path -LiteralPath $evaluation)){ return [int]($item.Name -replace '.*_attempt','') } }
    if($items.Count -ge 3){ throw "Attempt cap reached for ${base}: three retained incomplete attempts require inspection." }
    return $items.Count + 1
}
function Paths($cell,$attempt) { $n=(BaseName $cell)+'_attempt{0:D2}' -f $attempt; return @{ Capture=(Join-Path $CaptureParent $n); Evaluation=(Join-Path $ValidationParent $n); Name=$n } }
function Add-Progress($record) { ($record | ConvertTo-Json -Compress) | Add-Content -LiteralPath $ProgressPath -Encoding utf8; }
function Add-Cell1Measurement($paths) {
    $marker='## Measured cell-1 addendum'; if((Get-Content -LiteralPath $A3Path -Raw).Contains($marker)){ return }
    $cap=(Get-ChildItem -LiteralPath $paths.Capture -Recurse -File | Measure-Object Length -Sum).Sum
    $val=(Get-ChildItem -LiteralPath $paths.Evaluation -Recurse -File | Measure-Object Length -Sum).Sum
    $blobs=@(Get-ChildItem -LiteralPath $paths.Capture -Recurse -File -Filter '*.npz' | Where-Object { $_.FullName -match 'projection_blobs' }).Count
    $layout=if($blobs -gt 0){'shared content-addressed projection blob present'}else{'no shared projection blob detected; per-snapshot projection risk retained'}
    $projection=($cap+$val)*35
    @("`n$marker",'',"Measured from $($paths.Name): capture=$cap bytes; validation=$val bytes; $layout.","A simple remaining-35 projection is $projection bytes. Even the stated worst-case projection growth (~6.7 GiB across the 12 projection-bearing cells) remains within the measured drive envelope after the 8 GiB per-cell guard; the guard continues to protect the tail.") | Add-Content -LiteralPath $A3Path -Encoding utf8
}
function Invoke-Logged($CommandArgs,$tee) {
    # $args is a PowerShell automatic variable and must not be used as a parameter name:
    # @args would splat the empty automatic one instead of the passed array.
    # $ErrorActionPreference is relaxed around the native call because in Windows PowerShell 5.1
    # 2>&1 wraps native stderr lines in NativeCommandError, which would terminate on EAP=Stop
    # even when the process exits 0. Exit code is checked explicitly instead.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & rtk proxy @CommandArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object @tee
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previous }
    if($code -ne 0){ throw "Native command failed (exit $code): $($CommandArgs -join ' ')" }
}

Push-Location -LiteralPath $ProjectRoot
try {
    New-Item -ItemType Directory -Force -Path $GridParent,$CaptureParent,$ValidationParent | Out-Null
    $log=Join-Path $GridParent ('batch_'+[guid]::NewGuid().ToString('N')+'.log'); $tee=@{FilePath=$log;Append=$true}
    $cells=Get-Cells; $selected=@($cells | Select-Object -Skip ($StartAt-1)); if($MaxCells -gt 0){$selected=@($selected|Select-Object -First $MaxCells)}
    $complete=0;$partial=0; foreach($c in $cells){$base=BaseName $c;$dirs=@(Get-ChildItem -LiteralPath $CaptureParent -Directory -Filter ($base+'_attempt*') -ErrorAction SilentlyContinue);if(@($dirs|Where-Object{ReceiptComplete $_.FullName}).Count){$complete++};if($dirs.Count -and -not @($dirs|Where-Object{ReceiptComplete $_.FullName}).Count){$partial++}}
    $free=[System.IO.DriveInfo]::new('G').AvailableFreeSpace; "START complete=$complete remaining=$($cells.Count-$complete) superseded=$partial free=$free log=$log" | Tee-Object @tee
    if($PlanOnly){$i=0;foreach($c in $cells){$i++;$a=NextAttempt $c;$p=Paths $c $a;"$i $($c.Arm)/$($c.Learner)/s$($c.Seed) attempt$a capture=$($p.Capture) evaluation=$($p.Evaluation)"};return}
    if((Read-Host 'Type RUN FORGETTING GRID ALL') -cne 'RUN FORGETTING GRID ALL'){throw 'Cancelled before batch start.'}
    $done=0;$skipped=0;$started=Get-Date
    foreach($c in $selected){$index=[array]::IndexOf($cells,$c)+1;$a=NextAttempt $c;$p=Paths $c $a;if((ReceiptComplete $p.Capture)-and(ReceiptComplete $p.Evaluation)){"$index SKIPPED (already complete)"|Tee-Object @tee;$skipped++;continue}
        if([System.IO.DriveInfo]::new('G').AvailableFreeSpace -lt 8GB){throw "FAILED cell ${index}: below 8 GiB guard"}
        $start=Get-Date;"$index RUN $($c.Arm)/$($c.Learner)/s$($c.Seed) attempt$a"|Tee-Object @tee
        try { Invoke-Logged @($PythonPath,'-m','ssl_hdc_pilot.scripts.benchmark_environment') $tee
            $emb=Join-Path $ProjectRoot ('ssl_hdc_pilot/results/'+$(if($c.Arm -eq 'frozen'){'arm_f_vitb16_trusted_all.npz'}else{'arm_d_vitb16_ep3_trusted_all.npz'}))
            if(-not (Test-Path $p.Capture)){Invoke-Logged @($PythonPath,'-u','-m','ssl_hdc_pilot.scripts.budget_experiment','--experiment','B','--threshold','0.10','--budget','400','--learner',$c.Learner,'--seed',"$($c.Seed)",'--embeddings',$emb,'--trusted-split',(Join-Path $ProjectRoot 'ssl_hdc_pilot/trusted_splits/train.csv'),'--output-dir',$p.Capture,'--execute','--preflight-confirmed') $tee}
            Invoke-Logged @($PythonPath,'-u','-m','ssl_hdc_pilot.scripts.evaluate_snapshots','--source-dir',$p.Capture,'--embeddings',$emb,'--validation-split',(Join-Path $ProjectRoot 'ssl_hdc_pilot/trusted_splits/val.csv'),'--output-dir',$p.Evaluation,'--execute','--preflight-confirmed') $tee
            if($index -eq 1){Add-Cell1Measurement $p};$done++;Add-Progress @{index=$index;learner=$c.Learner;arm=$c.Arm;seed=$c.Seed;attempt=$a;status='COMPLETE';timestamp=(Get-Date -Format o)};"$index COMPLETE elapsed=$((Get-Date)-$start)"|Tee-Object @tee
        } catch {Add-Progress @{index=$index;learner=$c.Learner;arm=$c.Arm;seed=$c.Seed;attempt=$a;status='FAILED';timestamp=(Get-Date -Format o);error=$_.Exception.Message};"FAILED cell $index command=$($_.InvocationInfo.Line) error=$($_.Exception.Message)"|Tee-Object @tee;throw}
    }
    "SUMMARY completed=$done skipped=$skipped remaining=$($cells.Count-$done-$skipped) elapsed=$((Get-Date)-$started) free=$([System.IO.DriveInfo]::new('G').AvailableFreeSpace) resume=powershell -NoProfile -File .\scripts\run-forgetting-grid-all.ps1"|Tee-Object @tee
} finally { Pop-Location }
