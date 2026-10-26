param(
    [ValidateSet('s_frozen','b_frozen','l_frozen','b_adapted','l_adapted')] [string]$Arm,
    [ValidateSet('hdc','ncm','ranpac','sign_ncm','median_ncm')] [string]$Learner,
    [ValidateRange(0,9)] [int]$Seed,
    [switch]$PlanOnly
)
$ErrorActionPreference='Stop'
$root=(Resolve-Path $PSScriptRoot\..).Path
$files=@{s_frozen='arm_f_trusted_all.npz';b_frozen='arm_f_vitb16_trusted_all.npz';l_frozen='arm_f_vitl16_trusted_all.npz';b_adapted='arm_d_vitb16_ep3_trusted_all.npz';l_adapted='arm_d_vitl16_ep4_trusted_all.npz'}
$py=Join-Path $root '.venv-e1\Scripts\python.exe'
$emb=Join-Path $root ('ssl_hdc_pilot\results\'+$files[$Arm])
$train=Join-Path $root 'ssl_hdc_pilot\trusted_splits\train.csv'
& rtk proxy $py -m ssl_hdc_pilot.scripts.budget_experiment --experiment B --threshold 0.10 --budget 400 --learner $Learner --seed "$Seed" --embeddings $emb --trusted-split $train --ablation-mode
if($LASTEXITCODE -ne 0){throw 'Input plan failed; no learner was constructed.'}
if(-not $PlanOnly){throw 'Prepared only. Official runs remain user-owned; use the approved launch procedure after confirmation.'}
