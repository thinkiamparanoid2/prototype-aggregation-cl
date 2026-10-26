# SSL-HDC Pilot Run Commands

## Current execution status — 2026-09-07

The sanctioned manifest, S/B/L ladder, approved A/B split construction, probe
gates, stream spec, selected adapted extraction, 72-run original grid,
18-cell EWC replacement, and 36-run prototype addition are complete in this
checkout. There are 108 summaries and 108 histories. Completed blocks below
are reproduction history, not pending launch instructions; their old gate
labels describe the state when written. In particular, the EWC cleanup/rerun
block must not be used as a routine startup step, and the prototype block is
expected to refuse already existing outputs.

Frozen 7B extraction has no completed artifact here. Equal-budget work and
encoder-geometry Study 2 need a specified continuation protocol. Scientific
status remains PROVISIONAL; see [the report](results/SSL_HDC_REPORT.md) and
[handoff](HANDOFF.md). Command bodies and registered parameters below are
preserved unchanged by this documentation refresh.

All blocks in this file are user-launch only unless explicitly labeled as a
historical smoke. Run from `G:\FOR THESIS\MAIN` on branch `local-shafin`.

```powershell
$PY = "G:\FOR THESIS\MAIN\.venv-e1\Scripts\python.exe"
Set-Location -LiteralPath "G:\FOR THESIS\MAIN"
```

## Sanctioned manifest construction

This is the only sanctioned Arm-D image pool. It enumerates `Data/train/`
once, documents the `Data/val/` duplicate subset, and hard-excludes trusted
`Data/test/` IDs. The agent executed this non-training construction once to
verify the required 51,906/5,193 reconciliation; the command remains here for
reproducibility.

```powershell
& $PY -u .\ssl_hdc_pilot\scripts\build_unlabeled_manifest.py --output-csv .\ssl_hdc_pilot\results\unlabeled_manifest.csv --summary-json .\ssl_hdc_pilot\results\unlabeled_manifest_summary.json
```

## Completed ViT-S Arm-F extraction — retained provenance, do not rerun

Trusted 5,767 images:

```powershell
# & $PY -u .\ssl_hdc_pilot\scripts\extract_embeddings.py --input-csv .\Data\test.csv --image-root .\Data\test --output-npz .\ssl_hdc_pilot\results\arm_f_trusted_all.npz --metadata-json .\ssl_hdc_pilot\results\arm_f_trusted_all_metadata.json --model-id facebook/dinov3-vits16-pretrain-lvd1689m --batch-size 64
```

## Unlabeled 51,906-image encoder pool:

```powershell
# & $PY -u .\ssl_hdc_pilot\scripts\extract_embeddings.py --input-csv .\ssl_hdc_pilot\results\unlabeled_manifest.csv --output-npz .\ssl_hdc_pilot\results\arm_f_unlabeled_pool.npz --metadata-json .\ssl_hdc_pilot\results\arm_f_unlabeled_pool_metadata.json --model-id facebook/dinov3-vits16-pretrain-lvd1689m --batch-size 64
```

## Official trusted split — BLOCKED, user launch only after supervisor sign-off

Do not run either command until the tail-class policy is confirmed. Option A
is the preregistered headline default; Option B is the all-25-class appendix.

```powershell
# Option A: fixed 60/20/20 headline split, 19 classes.
& $PY -u .\ssl_hdc_pilot\scripts\build_trusted_splits.py --policy option-a-19class --input-csv .\Data\test.csv --output-dir .\ssl_hdc_pilot\trusted_splits --seed 20260716 --supervisor-approved

# Option B: 5-fold x 10-repeat all-25-class appendix assignments.
& $PY -u .\ssl_hdc_pilot\scripts\build_trusted_splits.py --policy option-b-cv --input-csv .\Data\test.csv --output-dir .\ssl_hdc_pilot\trusted_splits_option_b --seed 20260716 --folds 5 --repeats 10 --supervisor-approved
```

## Historical Arm-D placeholder - superseded by the 2026-07-18 amendment

This commented command records the pre-amendment blocked state. Do not use it;
the active, resume-safe user-launch command is appended below with the measured
wall-time estimate.

```powershell
# & $PY -u .\ssl_hdc_pilot\scripts\adapt_dinov3.py --manifest .\ssl_hdc_pilot\results\unlabeled_manifest.csv --output-dir .\ssl_hdc_pilot\results\arm_d --model-id facebook/dinov3-vits16-pretrain-lvd1689m --epochs 5 --learning-rate 1e-5 --ema-momentum 0.996 --global-crops 2 --local-crops 4
```

## Streaming protocol and CNN continual baselines — deferred by binding spec

SUPERSEDED 2026-07-19: this paragraph records the historical deferral before
the Phase 4 code existed. The implementation and blocked official commands in
the immediately following section are authoritative.

No official run was launched by the agent.

## SUPERSEDED - 2026-07-19 Phase 4 continual placeholder

The simulator now runs HDC, naive incremental fine-tuning, online EWC, and
reservoir replay for every stream-order seed and confidence threshold declared
in the stream spec. One invocation therefore preserves the same embedding
artifact, trusted split, per-seed stream order, trigger definition, and
label-budget rule across all four update mechanisms.

Both official blocks below remain BLOCKED on:

1. a supervisor-approved trusted stream split; and
2. the supervising agent checking in the preregistered canonical
   `ssl_hdc_pilot/stream_spec.json` with `smoke=false`.

The runner hard-errors if either requirement is absent. Replace the split
placeholder only after approval; it is intentionally not selected here.

```powershell
$ProjectRoot = (Get-Location).Path
$PY = Join-Path $ProjectRoot ".venv-e1\Scripts\python.exe"
$StreamSpec = Join-Path $ProjectRoot "ssl_hdc_pilot\stream_spec.json"
$TrustedStreamSplit = Join-Path $ProjectRoot "ssl_hdc_pilot\trusted_splits\<SUPERVISOR_SELECTED_STREAM_SPLIT>.csv"

if ($TrustedStreamSplit -match "<") {
    throw "BLOCKED: supervisor must select the trusted stream split"
}
if (-not (Test-Path -LiteralPath $TrustedStreamSplit -PathType Leaf)) {
    throw "BLOCKED: supervisor-approved trusted stream split is absent"
}
if (-not (Test-Path -LiteralPath $StreamSpec -PathType Leaf)) {
    throw "BLOCKED: preregistered ssl_hdc_pilot/stream_spec.json is absent"
}
$StreamConfig = Get-Content -LiteralPath $StreamSpec -Raw | ConvertFrom-Json
if ($StreamConfig.smoke -ne $false) {
    throw "BLOCKED: official stream spec must declare smoke=false"
}
```

### Arm F ViT-B/16 headline continual grid - BLOCKED

This consumes the already verified 768-d frozen ViT-B embedding NPZ. It reads
no raw images and trains no encoder.

```powershell
$ArmFTrustedEmbeddings = Join-Path $ProjectRoot "ssl_hdc_pilot\results\arm_f_vitb16_trusted_all.npz"
& $PY -u -m ssl_hdc_pilot.scripts.stream_runner --embeddings $ArmFTrustedEmbeddings --trusted-split $TrustedStreamSplit --stream-spec $StreamSpec --output-dir (Join-Path $ProjectRoot "ssl_hdc_pilot\results\continual\arm_f_vitb16") --learners hdc,naive,ewc,replay
```

### Arm D ViT-B/16 headline continual grid - BLOCKED on probe gate too

Do not fill or launch this block unless the preregistered ViT-B probe gate
passes and the supervising agent extracts a label-free trusted NPZ from the
selected EMA-teacher checkpoint. The continual learner receives that NPZ only;
it never reads an Arm-D checkpoint.

```powershell
$ArmDTrustedEmbeddings = Join-Path $ProjectRoot "ssl_hdc_pilot\results\<PROBE_GATE_SELECTED_VITB16_TRUSTED_EMBEDDINGS>.npz"
if ($ArmDTrustedEmbeddings -match "<") {
    throw "BLOCKED: ViT-B Arm-D probe gate has not selected a trusted embedding artifact"
}
& $PY -u -m ssl_hdc_pilot.scripts.stream_runner --embeddings $ArmDTrustedEmbeddings --trusted-split $TrustedStreamSplit --stream-spec $StreamSpec --output-dir (Join-Path $ProjectRoot "ssl_hdc_pilot\results\continual\arm_d_vitb16") --learners hdc,naive,ewc,replay
```

No official run was launched by the agent.

## Authoritative 2026-07-19 encoder-size LADDER — user launch only

This is the single authoritative ladder block. It supersedes every legacy
Arm-D block later in this file, including the former batch-32 ViT-S-only block.
The fixed order is Arm-F preparation, then Arm-D ViT-S, ViT-B, and ViT-L
sequentially. HEADLINE is ViT-B/16, locked before results. Every rung compares
frozen and adapted encoders of the same size.

### Launch 1 — complete required Arm-F ladder extraction

The existing ViT-S trusted and pool artifacts are retained. Re-extract the
trusted 5,767 images at ViT-B and ViT-L. Extract the 51,906-image unlabeled pool
at ViT-B only because the headline downstream study needs it. ViT-L pool
extraction is deliberately deferred.

```powershell
& $PY -u .\ssl_hdc_pilot\scripts\extract_embeddings.py --input-csv .\Data\test.csv --image-root .\Data\test --output-npz .\ssl_hdc_pilot\results\arm_f_vitb16_trusted_all.npz --metadata-json .\ssl_hdc_pilot\results\arm_f_vitb16_trusted_all_metadata.json --model-id facebook/dinov3-vitb16-pretrain-lvd1689m --batch-size 32 --device cuda

& $PY -u .\ssl_hdc_pilot\scripts\extract_embeddings.py --input-csv .\Data\test.csv --image-root .\Data\test --output-npz .\ssl_hdc_pilot\results\arm_f_vitl16_trusted_all.npz --metadata-json .\ssl_hdc_pilot\results\arm_f_vitl16_trusted_all_metadata.json --model-id facebook/dinov3-vitl16-pretrain-lvd1689m --batch-size 32 --device cuda

& $PY -u .\ssl_hdc_pilot\scripts\extract_embeddings.py --input-csv .\ssl_hdc_pilot\results\unlabeled_manifest.csv --output-npz .\ssl_hdc_pilot\results\arm_f_vitb16_unlabeled_pool.npz --metadata-json .\ssl_hdc_pilot\results\arm_f_vitb16_unlabeled_pool_metadata.json --model-id facebook/dinov3-vitb16-pretrain-lvd1689m --batch-size 32 --device cuda
```

### Launch 2 — Arm-D ViT-S/16 rung

Measured arithmetic:
`ceil(51,906 / 32) * 5 / 0.7580 = 1,623 * 5 / 0.7580 =
10,705.8 seconds = 2.97 training-loop hours`. Allow approximately 4
wall-clock hours for model startup, six checkpoint writes, and run variability.

```powershell
& $PY -u -m ssl_hdc_pilot.scripts.adapt_dinov3 --manifest .\ssl_hdc_pilot\results\unlabeled_manifest.csv --output-dir .\ssl_hdc_pilot\results\arm_d_vits16 --model-id facebook/dinov3-vits16-pretrain-lvd1689m --epochs 5 --learning-rate 1e-5 --ema-momentum 0.996 --global-crops 2 --local-crops 4 --batch-size 32 --num-workers 0 --device cuda
```

### Launch 3 — Arm-D ViT-B/16 rung (locked HEADLINE)

Measured arithmetic:
`ceil(51,906 / 32) * 5 / 0.5307 = 1,623 * 5 / 0.5307 =
15,291.5 seconds = 4.25 training-loop hours`. Allow approximately 6
wall-clock hours for model startup, six checkpoint writes, and run variability.

```powershell
& $PY -u -m ssl_hdc_pilot.scripts.adapt_dinov3 --manifest .\ssl_hdc_pilot\results\unlabeled_manifest.csv --output-dir .\ssl_hdc_pilot\results\arm_d_vitb16 --model-id facebook/dinov3-vitb16-pretrain-lvd1689m --epochs 5 --learning-rate 1e-5 --ema-momentum 0.996 --global-crops 2 --local-crops 4 --batch-size 32 --num-workers 0 --device cuda
```

### Launch 4 — Arm-D ViT-L/16 rung

WARNING: no other GPU applications may run during the ViT-L rung. Start it only
after the ViT-B command has completed and the GPU is otherwise idle.

Measured arithmetic:
`ceil(51,906 / 32) * 5 / 0.3091 = 1,623 * 5 / 0.3091 =
26,253.6 seconds = 7.29 training-loop hours`. Allow approximately 10
wall-clock hours for model startup, six checkpoint writes, and run variability.

```powershell
& $PY -u -m ssl_hdc_pilot.scripts.adapt_dinov3 --manifest .\ssl_hdc_pilot\results\unlabeled_manifest.csv --output-dir .\ssl_hdc_pilot\results\arm_d_vitl16 --model-id facebook/dinov3-vitl16-pretrain-lvd1689m --epochs 5 --learning-rate 1e-5 --ema-momentum 0.996 --global-crops 2 --local-crops 4 --batch-size 32 --num-workers 0 --device cuda
```

### Launch 5 — per-rung probe gates — completed; original command retained

Do not launch these until the supervisor-approved Option-A trusted train and
validation CSVs exist. Each command reads the recorded model ID from its
checkpoint directory, extracts the matching 384/768/1024-d trusted features,
and writes a verdict naming the rung.

```powershell
& $PY -u -m ssl_hdc_pilot.scripts.probe_gate --checkpoint-dir .\ssl_hdc_pilot\results\arm_d_vits16 --trusted-train-csv .\ssl_hdc_pilot\trusted_splits\train.csv --trusted-val-csv .\ssl_hdc_pilot\trusted_splits\val.csv --output-json .\ssl_hdc_pilot\results\arm_d_vits16_probe_gate.json --batch-size 32 --device cuda

& $PY -u -m ssl_hdc_pilot.scripts.probe_gate --checkpoint-dir .\ssl_hdc_pilot\results\arm_d_vitb16 --trusted-train-csv .\ssl_hdc_pilot\trusted_splits\train.csv --trusted-val-csv .\ssl_hdc_pilot\trusted_splits\val.csv --output-json .\ssl_hdc_pilot\results\arm_d_vitb16_probe_gate.json --batch-size 32 --device cuda

& $PY -u -m ssl_hdc_pilot.scripts.probe_gate --checkpoint-dir .\ssl_hdc_pilot\results\arm_d_vitl16 --trusted-train-csv .\ssl_hdc_pilot\trusted_splits\train.csv --trusted-val-csv .\ssl_hdc_pilot\trusted_splits\val.csv --output-json .\ssl_hdc_pilot\results\arm_d_vitl16_probe_gate.json --batch-size 32 --device cuda
```

No official run was launched by the agent.

## SUPERSEDED - DO NOT RUN - Official Arm-D adaptation

The 2026-07-19 one-step implementation smokes measured 1.0033-1.7294 seconds
for two processed images. Direct throughput extrapolation for 51,906 images x
5 epochs is 36.16-62.34 GPU-hours; budget approximately 65 hours because the
tiny smokes do not model all checkpoint and long-run variability. The command
is resume-safe: rerunning it continues from the last complete epoch, while a
configuration mismatch hard-errors.

```powershell
# & $PY -u -m ssl_hdc_pilot.scripts.adapt_dinov3 --manifest .\ssl_hdc_pilot\results\unlabeled_manifest.csv --output-dir .\ssl_hdc_pilot\results\arm_d --model-id facebook/dinov3-vits16-pretrain-lvd1689m --epochs 5 --learning-rate 1e-5 --ema-momentum 0.996 --global-crops 2 --local-crops 4 --batch-size 2 --num-workers 0 --device cuda
```

## SUPERSEDED - DO NOT RUN - Single-size Arm-D probe gate

This command extracts trusted train/validation embeddings for every complete
checkpoint, fits the amendment-fixed linear probe, and writes the selection
verdict. It fails closed if the supervisor-approved split provenance is absent.

```powershell
# & $PY -u -m ssl_hdc_pilot.scripts.probe_gate --checkpoint-dir .\ssl_hdc_pilot\results\arm_d --trusted-train-csv .\ssl_hdc_pilot\trusted_splits\train.csv --trusted-val-csv .\ssl_hdc_pilot\trusted_splits\val.csv --output-json .\ssl_hdc_pilot\results\arm_d_probe_gate.json --batch-size 64 --device cuda
```

No official run was launched by the agent.

## SUPERSEDED - DO NOT RUN - Round 3 Arm-D adaptation

This is the active long-run command. Round 3 exports the EMA teacher for the
later probe, retains the student for exact resume, aborts on non-finite loss,
and has passed a fresh-process CUDA epoch-1 to epoch-2 continuation smoke. The
measured tiny-smoke range is approximately 33.65-61.51 GPU-hours; retain the
conservative 65-hour budget. Rerun this exact command after interruption.

```powershell
# & $PY -u -m ssl_hdc_pilot.scripts.adapt_dinov3 --manifest .\ssl_hdc_pilot\results\unlabeled_manifest.csv --output-dir .\ssl_hdc_pilot\results\arm_d --model-id facebook/dinov3-vits16-pretrain-lvd1689m --epochs 5 --learning-rate 1e-5 --ema-momentum 0.996 --global-crops 2 --local-crops 4 --batch-size 2 --num-workers 0 --device cuda
```

The probe gate remains user-launched only after the approved Option-A splits
exist and all epochs 0 through 5 are complete; it now refuses partial cadence.

No official run was launched by the agent.

## SUPERSEDED - DO NOT RUN - Batch-16 Arm-D adaptation

This block supersedes the earlier batch-2 Arm-D command blocks. A bounded CUDA
sweep tested batch sizes 2/4/8/16 with `num_workers` 0/2 for exactly ten
optimization steps each. Batch 16 with workers 0 was the largest tested setting
below the memory rule: 2,155 MiB peak whole-device use, or 17.62% of the
12,227 MiB reported GPU capacity. It measured 1.1417 steps/s (18.2672 images/s).
`ceil(51,906 / 16) * 5 = 16,225` steps extrapolates to 3.95 training-loop
hours; allow approximately 5 wall-clock hours for startup, checkpoint writes,
and smoke-to-full-run variability. Only batch size changed from the earlier
official command; workers remain 0 and LR/all other recipe parameters remain
unchanged. Rerun this exact command after interruption.

```powershell
# & $PY -u -m ssl_hdc_pilot.scripts.adapt_dinov3 --manifest .\ssl_hdc_pilot\results\unlabeled_manifest.csv --output-dir .\ssl_hdc_pilot\results\arm_d --model-id facebook/dinov3-vits16-pretrain-lvd1689m --epochs 5 --learning-rate 1e-5 --ema-momentum 0.996 --global-crops 2 --local-crops 4 --batch-size 16 --num-workers 0 --device cuda
```

No official run was launched by the agent.

## Authoritative 2026-07-19 continual study - user launch only

This is the single authoritative continual-study section. The simulator runs
the fixed grid `{HDC, naive, EWC, replay} x {stream seeds 0,1,2} x
{margin thresholds 0.05,0.10,0.20}` for each embedding NPZ. The headline pair
is frozen ViT-B epoch 0 versus adapted ViT-B epoch 3. Learners receive frozen
embeddings only and never open the checkpoints.

### A. Generate the locked Section 6a stream spec - completed; reproduction only

The novel selection, 25% known-class holdback, entry fractions, seeds,
thresholds, and confirmed-label accounting are locked constants with no
tunable flags.

```powershell
& $PY -u -m ssl_hdc_pilot.scripts.build_stream_spec --split-dir .\ssl_hdc_pilot\trusted_splits
```

Expected outputs: `ssl_hdc_pilot\stream_spec.json` (`smoke=false`) and
`ssl_hdc_pilot\stream_spec_summary.log`.

### B. Extract selected adapted-checkpoint embeddings - user launch only

The shared probe/extraction loader validates the immutable run config, model
identity, raw dimension, EMA-teacher role, rung, epoch, and checkpoint state.
ViT-B epoch 3 is required for the headline pair; ViT-L epoch 4 is the scale
appendix artifact.

```powershell
& $PY -u -m ssl_hdc_pilot.scripts.extract_embeddings --input-csv .\Data\test.csv --image-root .\Data\test --checkpoint-dir .\ssl_hdc_pilot\results\arm_d_vitb16\checkpoint_epoch3 --output-npz .\ssl_hdc_pilot\results\arm_d_vitb16_ep3_trusted_all.npz --metadata-json .\ssl_hdc_pilot\results\arm_d_vitb16_ep3_trusted_all_metadata.json --batch-size 32 --device cuda

& $PY -u -m ssl_hdc_pilot.scripts.extract_embeddings --input-csv .\Data\test.csv --image-root .\Data\test --checkpoint-dir .\ssl_hdc_pilot\results\arm_d_vitl16\checkpoint_epoch4 --output-npz .\ssl_hdc_pilot\results\arm_d_vitl16_ep4_trusted_all.npz --metadata-json .\ssl_hdc_pilot\results\arm_d_vitl16_ep4_trusted_all_metadata.json --batch-size 32 --device cuda
```

### C. Official headline continual grid - completed; reproduction only

The frozen ViT-B NPZ already exists. Do not launch until A has produced the
canonical non-smoke spec and B has produced the adapted ViT-B epoch-3 NPZ.
Each invocation produces 36 runs; both invocations produce the complete
72-run headline grid.

Measured basis: a synthetic 768-d, one-seed, one-threshold smoke across all
four learners took 3.865 seconds for 44 stream examples on this machine. Linear
scaling to the generated 976-example stream is about 86 seconds per
seed-threshold bundle, 13 minutes per embedding grid, and 26 minutes for the
two-embedding headline grid. Budget up to one hour because request rates,
filesystem writes, and GPU contention can make smoke scaling optimistic.

```powershell
$ProjectRoot = (Get-Location).Path
$PY = Join-Path $ProjectRoot ".venv-e1\Scripts\python.exe"
$StreamSpec = Join-Path $ProjectRoot "ssl_hdc_pilot\stream_spec.json"
$TrustedTrain = Join-Path $ProjectRoot "ssl_hdc_pilot\trusted_splits\train.csv"
$FrozenB = Join-Path $ProjectRoot "ssl_hdc_pilot\results\arm_f_vitb16_trusted_all.npz"
$AdaptedB = Join-Path $ProjectRoot "ssl_hdc_pilot\results\arm_d_vitb16_ep3_trusted_all.npz"

foreach ($RequiredFile in @($StreamSpec, $TrustedTrain, $FrozenB, $AdaptedB)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "BLOCKED: required continual-study artifact is absent: $RequiredFile"
    }
}
$StreamConfig = Get-Content -LiteralPath $StreamSpec -Raw | ConvertFrom-Json
if ($StreamConfig.smoke -ne $false) {
    throw "BLOCKED: official stream spec must declare smoke=false"
}

& $PY -u -m ssl_hdc_pilot.scripts.stream_runner --embeddings $FrozenB --trusted-split $TrustedTrain --stream-spec $StreamSpec --output-dir (Join-Path $ProjectRoot "ssl_hdc_pilot\results\continual\frozen_vitb16") --learners hdc,naive,ewc,replay

& $PY -u -m ssl_hdc_pilot.scripts.stream_runner --embeddings $AdaptedB --trusted-split $TrustedTrain --stream-spec $StreamSpec --output-dir (Join-Path $ProjectRoot "ssl_hdc_pilot\results\continual\adapted_vitb16_ep3") --learners hdc,naive,ewc,replay
```

No official run was launched by the agent.

## Completed 2026-07-20 EWC-only correction rerun - reproduction history

The decay-1.0 EWC outputs are invalid: frozen-B currently has nine stale cells,
while adapted-B may contain only the orphaned cells written before its
non-finite-loss abort. This block handles that partial state without assuming
an EWC file count. It snapshots every non-EWC file in both output directories
before deleting anything, removes every existing EWC history/summary, then
runs the corrected decay-0.9 grid in fixed frozen-B then adapted-B order.
The two arms produce 18 EWC cells total: three seeds x three thresholds per
arm. HDC, naive, and replay outputs must remain byte-identical.

~~~powershell
$ProjectRoot = (Get-Location).Path
$PY = Join-Path $ProjectRoot ".venv-e1\Scripts\python.exe"
$StreamSpec = Join-Path $ProjectRoot "ssl_hdc_pilot\stream_spec.json"
$TrustedTrain = Join-Path $ProjectRoot "ssl_hdc_pilot\trusted_splits\train.csv"
$FrozenB = Join-Path $ProjectRoot "ssl_hdc_pilot\results\arm_f_vitb16_trusted_all.npz"
$AdaptedB = Join-Path $ProjectRoot "ssl_hdc_pilot\results\arm_d_vitb16_ep3_trusted_all.npz"
$EwcSummaryPattern = "__ewc__margin_(0p05|0p1|0p2)_summary\.json$"
$EwcHistoryPattern = "__ewc__margin_(0p05|0p1|0p2)_history\.csv$"

$EwcReruns = @(
    @{
        Name = "frozen-B"
        Embeddings = $FrozenB
        OutputDir = Join-Path $ProjectRoot "ssl_hdc_pilot\results\continual\frozen_vitb16"
    },
    @{
        Name = "adapted-B"
        Embeddings = $AdaptedB
        OutputDir = Join-Path $ProjectRoot "ssl_hdc_pilot\results\continual\adapted_vitb16_ep3"
    }
)

foreach ($RequiredFile in @($PY, $StreamSpec, $TrustedTrain, $FrozenB, $AdaptedB)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "BLOCKED: required EWC-rerun artifact is absent: $RequiredFile"
    }
}
$StreamConfig = Get-Content -LiteralPath $StreamSpec -Raw | ConvertFrom-Json
if ($StreamConfig.smoke -ne $false) {
    throw "BLOCKED: official stream spec must declare smoke=false"
}
if ([double]$StreamConfig.baseline.ewc_lambda -ne 10.0) {
    throw "BLOCKED: ewc_lambda must remain 10.0"
}
if ([double]$StreamConfig.baseline.ewc_decay -ne 0.9) {
    throw "BLOCKED: amended ewc_decay must be 0.9"
}
$StreamSpecSha256 = (Get-FileHash -LiteralPath $StreamSpec -Algorithm SHA256).Hash.ToLowerInvariant()

$NonEwcHashesByArm = @{}
$ExistingEwcFilesByArm = @{}
foreach ($Rerun in $EwcReruns) {
    if (-not (Test-Path -LiteralPath $Rerun.OutputDir -PathType Container)) {
        throw "BLOCKED: existing continual output directory is absent: $($Rerun.OutputDir)"
    }

    $ExistingFiles = @(Get-ChildItem -LiteralPath $Rerun.OutputDir -File)
    $ExistingEwcFiles = @($ExistingFiles | Where-Object { $_.Name -match "__ewc__" })
    $ExistingEwcFilesByArm[$Rerun.Name] = $ExistingEwcFiles

    $NonEwcHashes = @{}
    foreach ($File in @($ExistingFiles | Where-Object { $_.Name -notmatch "__ewc__" })) {
        $NonEwcHashes[$File.FullName] = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash
    }
    $NonEwcHashesByArm[$Rerun.Name] = $NonEwcHashes
}

foreach ($Rerun in $EwcReruns) {
    foreach ($File in @($ExistingEwcFilesByArm[$Rerun.Name])) {
        Remove-Item -LiteralPath $File.FullName -Force
    }
}

foreach ($Rerun in $EwcReruns) {
    & $PY -u -m ssl_hdc_pilot.scripts.stream_runner --embeddings $Rerun.Embeddings --trusted-split $TrustedTrain --stream-spec $StreamSpec --output-dir $Rerun.OutputDir --ewc-only
    if ($LASTEXITCODE -ne 0) {
        throw "Corrected EWC rerun failed for $($Rerun.Name)"
    }

    $FreshEwcFiles = @(Get-ChildItem -LiteralPath $Rerun.OutputDir -File | Where-Object { $_.Name -match "__ewc__" })
    $FreshSummaries = @($FreshEwcFiles | Where-Object { $_.Name -match $EwcSummaryPattern })
    $FreshHistories = @($FreshEwcFiles | Where-Object { $_.Name -match $EwcHistoryPattern })
    if ($FreshEwcFiles.Count -ne 18 -or $FreshSummaries.Count -ne 9 -or $FreshHistories.Count -ne 9) {
        throw "Corrected EWC rerun must produce 9 summaries and 9 histories for $($Rerun.Name); found $($FreshSummaries.Count) and $($FreshHistories.Count)"
    }
    foreach ($Summary in $FreshSummaries) {
        $SummaryPayload = Get-Content -LiteralPath $Summary.FullName -Raw | ConvertFrom-Json
        if ($SummaryPayload.learner -ne "ewc") {
            throw "Fresh EWC summary has the wrong learner tag: $($Summary.FullName)"
        }
        if ($SummaryPayload.stream_spec_sha256 -ne $StreamSpecSha256) {
            throw "Fresh EWC summary has a stale stream-spec hash: $($Summary.FullName)"
        }
    }
}

foreach ($Rerun in $EwcReruns) {
    $ExpectedHashes = $NonEwcHashesByArm[$Rerun.Name]
    $CurrentNonEwcFiles = @(Get-ChildItem -LiteralPath $Rerun.OutputDir -File | Where-Object { $_.Name -notmatch "__ewc__" })
    if ($CurrentNonEwcFiles.Count -ne $ExpectedHashes.Count) {
        throw "Non-EWC file count changed during rerun for $($Rerun.Name)"
    }
    foreach ($Path in $ExpectedHashes.Keys) {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "Non-EWC output disappeared during rerun: $Path"
        }
        $CurrentHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
        if ($CurrentHash -ne $ExpectedHashes[$Path]) {
            throw "Non-EWC output changed during rerun: $Path"
        }
    }
}
~~~

No official rerun was launched by the agent.

## SUPERSEDED - DO NOT RUN - Batch-64 Arm-D adaptation

This final block supersedes every earlier Arm-D command block. The follow-up
bounded sweep tested batch sizes 32 and 64 with `num_workers` 0 and 2.
Batch 64 with workers 0 was the largest tested setting and remained below the
memory rule: 3,238 MiB peak whole-device use, or 26.48% of the 12,227 MiB GPU.
It measured 0.3533 steps/s (22.6101 images/s), while workers 2 measured 0.3109
steps/s. `ceil(51,906 / 64) * 5 = 4,060` steps extrapolates to 3.19
training-loop hours; allow approximately 4 wall-clock hours for startup,
checkpoint writes, and smoke-to-full-run variability. Only batch size changes
from the preceding authoritative command; workers remain 0 and LR/all other
recipe parameters remain unchanged. Rerun this exact command after interruption.

```powershell
# & $PY -u -m ssl_hdc_pilot.scripts.adapt_dinov3 --manifest .\ssl_hdc_pilot\results\unlabeled_manifest.csv --output-dir .\ssl_hdc_pilot\results\arm_d --model-id facebook/dinov3-vits16-pretrain-lvd1689m --epochs 5 --learning-rate 1e-5 --ema-momentum 0.996 --global-crops 2 --local-crops 4 --batch-size 64 --num-workers 0 --device cuda
```

No official run was launched by the agent.

## SUPERSEDED - DO NOT RUN - Final single-size batch-32 Arm-D adaptation

This final block supersedes every earlier Arm-D command block.

~~Superseded batch-2 planning basis: approximately 33.65-61.51 GPU-hours with
a conservative 65-hour budget.~~

The authoritative idle-GPU sweep selected batch 32 with `num_workers=0`.
Headroom already budgets for the user watching video during the official run:
the measured whole-device NVIDIA peak was 2,241 MiB, and the independent
PyTorch peak plus 1 GiB video margin was `1,052.03 + 1,024 = 2,076.03` MiB;
both are below the fixed 9,216 MiB line (75% of 12 GiB). Batch 32 preserves
`ceil(51,906 / 32) * 5 = 1,623 * 5 = 8,115` optimizer updates. Batch 64 was
measured but is excluded because it provides only 4,060 updates.

Measured wall-time arithmetic:
`1,623 steps/epoch * 5 epochs / 0.7580 steps/s = 10,705.8 seconds = 2.97
training-loop hours`. Allow approximately 4 wall-clock hours for model
startup, six checkpoint writes, and full-run variability. LR, epochs, EMA,
crops, AMP, gradient checkpointing, and every other recipe parameter are
unchanged. Rerun this exact command after interruption.

```powershell
# & $PY -u -m ssl_hdc_pilot.scripts.adapt_dinov3 --manifest .\ssl_hdc_pilot\results\unlabeled_manifest.csv --output-dir .\ssl_hdc_pilot\results\arm_d --model-id facebook/dinov3-vits16-pretrain-lvd1689m --epochs 5 --learning-rate 1e-5 --ema-momentum 0.996 --global-crops 2 --local-crops 4 --batch-size 32 --num-workers 0 --device cuda
```

No official run was launched by the agent.

## 2026-07-20 — Arm-F frozen 7B ceiling rung (user-launched on the 4090)

Frozen DINOv3 ViT-7B/16 extraction of the trusted 5,767 images, bfloat16, as
the strong-encoder ceiling reference. RUN THIS ONLY on the confirmed 24 GB
RTX 4090 — the 12 GB local 5070 cannot hold the ~13.4 GB bf16 weights.
Prerequisite: the gated 7B weights must be present in the project HF cache
(`.hf_cache`); the download command is recorded outside this file. Batch size
8 is a conservative starting point; raise it if VRAM headroom allows.

```powershell
# Smoke first (50 images), verify it loads in bf16 and writes a 4096-d NPZ:
& $PY -u -m ssl_hdc_pilot.scripts.extract_embeddings --input-csv .\Data\test.csv --image-root .\Data\test --output-npz .\ssl_hdc_pilot\results\arm_f_vit7b16_smoke_50.npz --metadata-json .\ssl_hdc_pilot\results\arm_f_vit7b16_smoke_50_metadata.json --model-id facebook/dinov3-vit7b16-pretrain-lvd1689m --dtype bfloat16 --device cuda --batch-size 8 --smoke --max-images 50

# Official trusted extraction (delete the smoke artifacts first):
& $PY -u -m ssl_hdc_pilot.scripts.extract_embeddings --input-csv .\Data\test.csv --image-root .\Data\test --output-npz .\ssl_hdc_pilot\results\arm_f_vit7b16_trusted_all.npz --metadata-json .\ssl_hdc_pilot\results\arm_f_vit7b16_trusted_all_metadata.json --model-id facebook/dinov3-vit7b16-pretrain-lvd1689m --dtype bfloat16 --device cuda --batch-size 8
```

The resulting `arm_f_vit7b16_trusted_all.npz` feeds the continual grid exactly
like the other Arm-F embeddings. Optionally run `probe_gate.py`-style linear
probing on it first to check whether the 7B features saturate the task (the
pre-registered gap-washout check) before spending the full grid on it.


## Completed prototype-only verification grid - August 2 protocol, committed September 7

This runs exactly 36 new cells: `{L2-normalized NCM, RanPAC-style prototype} x
{frozen-B, adapted-B} x {seeds 0,1,2} x {thresholds 0.05,0.10,0.20}`.
NCM averages per-sample L2-normalized embeddings. RanPAC uses HDC's seed and
10,000-dimensional Gaussian projection, ReLU, and online means; decorrelation/ridge are omitted.

Measured 2026-08-02 smoke: both learners, synthetic 768-D CUDA, 44 examples,
one seed and threshold: 0.514 s. Linear projection-count estimate: about
11 minutes for 36 cells; allow 25 minutes for I/O and scheduling variation.

The block refuses existing NCM/RanPAC files, hashes every pre-existing file,
and verifies those files byte-for-byte after adding the new outputs.

```powershell
Set-Location -LiteralPath "G:\FOR THESIS\MAIN"
$ProjectRoot = (Get-Location).Path
if ($ProjectRoot -ne "G:\FOR THESIS\MAIN") { throw "Unexpected project root: $ProjectRoot" }

$PY = Join-Path $ProjectRoot ".venv-e1\Scripts\python.exe"
$Spec = Join-Path $ProjectRoot "ssl_hdc_pilot\stream_spec.json"
$Split = Join-Path $ProjectRoot "ssl_hdc_pilot\trusted_splits\train.csv"
$Frozen = Join-Path $ProjectRoot "ssl_hdc_pilot\results\arm_f_vitb16_trusted_all.npz"
$Adapted = Join-Path $ProjectRoot "ssl_hdc_pilot\results\arm_d_vitb16_ep3_trusted_all.npz"
$AnyPrototypePattern = "__(ncm|ranpac)__"
$Pattern = "__(ncm|ranpac)__margin_(0p05|0p1|0p2)_(summary\.json|history\.csv)$"
$Arms = @(
    @{ Name="frozen-B"; Embeddings=$Frozen; Output=Join-Path $ProjectRoot "ssl_hdc_pilot\results\continual\frozen_vitb16" },
    @{ Name="adapted-B"; Embeddings=$Adapted; Output=Join-Path $ProjectRoot "ssl_hdc_pilot\results\continual\adapted_vitb16_ep3" }
)

foreach ($File in @($PY,$Spec,$Split,$Frozen,$Adapted)) {
    if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { throw "BLOCKED: missing $File" }
}
$Config = Get-Content -LiteralPath $Spec -Raw | ConvertFrom-Json
if ($Config.smoke -ne $false) { throw "BLOCKED: official spec must set smoke=false" }
if ((@($Config.seeds.stream_order) -join ",") -ne "0,1,2") { throw "BLOCKED: seeds changed" }
if ((@($Config.protocol.confidence_thresholds) -join ",") -ne "0.05,0.1,0.2") { throw "BLOCKED: thresholds changed" }
if ([int]$Config.hdc.hypervector_dim -ne 10000) { throw "BLOCKED: projection dimension changed" }
$SpecHash = (Get-FileHash -LiteralPath $Spec -Algorithm SHA256).Hash.ToLowerInvariant()

$Before = @{}
foreach ($Arm in $Arms) {
    if (-not (Test-Path -LiteralPath $Arm.Output -PathType Container)) { throw "BLOCKED: missing $($Arm.Output)" }
    $Files = @(Get-ChildItem -LiteralPath $Arm.Output -File)
    if (@($Files | Where-Object Name -match $AnyPrototypePattern).Count -ne 0) { throw "BLOCKED: prototype output exists for $($Arm.Name)" }
    $Hashes = @{}
    foreach ($File in $Files) { $Hashes[$File.FullName] = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash }
    $Before[$Arm.Name] = $Hashes
}

foreach ($Arm in $Arms) {
    & $PY -u -m ssl_hdc_pilot.scripts.stream_runner --embeddings $Arm.Embeddings --trusted-split $Split --stream-spec $Spec --output-dir $Arm.Output --learners ncm,ranpac
    if ($LASTEXITCODE -ne 0) { throw "Prototype grid failed for $($Arm.Name)" }

    $Fresh = @(Get-ChildItem -LiteralPath $Arm.Output -File | Where-Object Name -match $Pattern)
    $Summaries = @($Fresh | Where-Object Name -match "_summary\.json$")
    $Histories = @($Fresh | Where-Object Name -match "_history\.csv$")
    if ($Fresh.Count -ne 36 -or $Summaries.Count -ne 18 -or $Histories.Count -ne 18) {
        throw "Expected 18 summaries and 18 histories for $($Arm.Name)"
    }
    foreach ($File in $Summaries) {
        $SummaryPayload = Get-Content -LiteralPath $File.FullName -Raw | ConvertFrom-Json
        if ($SummaryPayload.stream_spec_sha256 -ne $SpecHash -or $SummaryPayload.learner -notin @("ncm","ranpac")) { throw "Bad prototype summary: $($File.FullName)" }
        if ($SummaryPayload.learner -eq "ncm" -and $SummaryPayload.learner_configuration.embedding_variant -ne "l2_normalized") { throw "NCM variant mismatch" }
        if ($SummaryPayload.learner -eq "ranpac" -and (
            [int]$SummaryPayload.learner_configuration.projection_dim -ne 10000 -or
            $SummaryPayload.learner_configuration.activation -ne "relu" -or
            $SummaryPayload.learner_configuration.decorrelation -ne $false -or
            $SummaryPayload.learner_configuration.ridge -ne $false
        )) { throw "RanPAC configuration mismatch" }
    }
}

foreach ($Arm in $Arms) {
    $Expected = $Before[$Arm.Name]
    $OldNow = @(Get-ChildItem -LiteralPath $Arm.Output -File | Where-Object Name -notmatch $Pattern)
    if ($OldNow.Count -ne $Expected.Count) { throw "Pre-existing file count changed for $($Arm.Name)" }
    foreach ($Path in $Expected.Keys) {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Pre-existing output disappeared: $Path" }
        if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Expected[$Path]) { throw "Pre-existing output changed: $Path" }
    }
}
Write-Host "Prototype grid complete: 36 new runs; all pre-existing hashes unchanged."
```

No official run was launched by the agent.

## Manual reconstruction first cell - 2026-09-10

NEW HDC-local user-launch block, not a rerun of any block above. Read
[RECONSTRUCTION.md](RECONSTRUCTION.md) first. Exactly one training-side fidelity
cell: frozen-B / NCM / seed 0 / threshold 0.10. No encoder training, held-out
evaluation, budget experiment, or timing benchmark. Stop after this cell and
return the report; do not automatically launch the remaining cells.

Run this block manually in PowerShell from HDC. It records the preflight,
requires explicit confirmation, and refuses an existing attempt directory.
If an attempt fails, preserve it and ask for review before choosing a new name.
No code/doc edits should occur while reconstruction is running.

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'G:\FOR THESIS\HDC'
$ReconstructionRoot = (Get-Location).Path
$ReconstructionPython = Join-Path $ReconstructionRoot '.venv-e1\Scripts\python.exe'
$ReconstructionTag = 'option_a_section6a_v1__arm_f_vitb16_trusted_all__stream_0__ncm__margin_0p1'
$ReconstructionSource = Join-Path $ReconstructionRoot "ssl_hdc_pilot\results\continual\frozen_vitb16\${ReconstructionTag}_summary.json"
$ReconstructionEmbeddings = Join-Path $ReconstructionRoot 'ssl_hdc_pilot\results\arm_f_vitb16_trusted_all.npz'
$ReconstructionTrain = Join-Path $ReconstructionRoot 'ssl_hdc_pilot\trusted_splits\train.csv'
$ReconstructionParent = Join-Path $ReconstructionRoot 'ssl_hdc_pilot\results\reconstruction'
$ReconstructionOutput = Join-Path $ReconstructionParent '2026-09-10_ncm_frozen_s0_t010_attempt01'
if (Test-Path -LiteralPath $ReconstructionOutput) { throw 'Attempt exists; preserve it and request review.' }
foreach ($ReconstructionInput in @($ReconstructionPython, $ReconstructionSource, $ReconstructionEmbeddings, $ReconstructionTrain)) {
    if (-not (Test-Path -LiteralPath $ReconstructionInput -PathType Leaf)) { throw "Missing: $ReconstructionInput" }
}

# Read-only plan: no learner execution and no output directory creation.
rtk proxy $ReconstructionPython -m ssl_hdc_pilot.scripts.reconstruct_run --source-summary $ReconstructionSource --embeddings $ReconstructionEmbeddings --trusted-split $ReconstructionTrain
if ($LASTEXITCODE -ne 0) { throw 'Reconstruction plan failed; stop here.' }

New-Item -ItemType Directory -Force -Path $ReconstructionParent | Out-Null
$ReconstructionTranscript = Join-Path $ReconstructionParent ("preflight_" + [guid]::NewGuid().ToString('N') + '.txt')
Start-Transcript -Path $ReconstructionTranscript -NoClobber
try {
    rtk proxy powershell -NoProfile -File scripts/hdc.ps1 -Action env
    if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
    rtk proxy nvidia-smi
    if ($LASTEXITCODE -ne 0) { throw 'GPU preflight unavailable; stop for review.' }
    Get-Process | Sort-Object ProcessName | Select-Object ProcessName, Id, CPU | Format-Table -AutoSize
    Write-Host 'Review GPU/process activity. Close competing training, browser/video, game, and recording workloads yourself.'
    Write-Host 'This list cannot prove the absence of GPU contamination. Do not continue if uncertain.'
    $ReconstructionConfirmation = Read-Host 'Type RUN ONE NCM CELL only after preflight is clean'
    if ($ReconstructionConfirmation -cne 'RUN ONE NCM CELL') { throw 'No experiment launched.' }
    rtk proxy $ReconstructionPython -u -m ssl_hdc_pilot.scripts.reconstruct_run --source-summary $ReconstructionSource --embeddings $ReconstructionEmbeddings --trusted-split $ReconstructionTrain --output-dir $ReconstructionOutput --execute --preflight-confirmed
    if ($LASTEXITCODE -ne 0) { throw 'Reconstruction failed or mismatched. Preserve outputs and return this transcript.' }
    Get-Content -LiteralPath (Join-Path $ReconstructionOutput 'fidelity_report.json')
    Write-Host "STOP HERE. Return fidelity_report.json from $ReconstructionOutput and transcript $ReconstructionTranscript"
} finally {
    Stop-Transcript
}
```

No official run was launched by the agent.

## Manual reconstruction remaining representative cells - 2026-09-11

Run only after the first frozen-B/NCM fidelity report passes and its artifacts
are verified. The first cell passed on 2026-09-11. This block runs the remaining
11 of the proposed 12 representative cells: all six learners on both ViT-B arms,
seed 0 and threshold 0.10, excluding the completed frozen-B/NCM cell.
It does NOT run all 108 historical cells or any budget/held-out experiment.
Keep code and inputs unchanged during execution. Stop on any mismatch; do not
relax tolerance or retry automatically. Preserve partial output for review.

```powershell
Set-Location -LiteralPath 'G:\FOR THESIS\HDC'
$ErrorActionPreference = 'Stop'
$repPython = '.\.venv-e1\Scripts\python.exe'
$repTrain = '.\ssl_hdc_pilot\trusted_splits\train.csv'
$repRoot = '.\ssl_hdc_pilot\results\reconstruction\2026-09-11_representative_attempt01'
$firstReport = Get-Content -Raw '.\ssl_hdc_pilot\results\reconstruction\2026-09-10_ncm_frozen_s0_t010_attempt01\fidelity_report.json' | ConvertFrom-Json
if ($firstReport.status -ne 'PASS') { throw 'First-cell PASS is required.' }
if (Test-Path -LiteralPath $repRoot) { throw 'Batch attempt exists. Preserve it and request review.' }

$repArms = @(
    @{ Name='frozen'; Folder='frozen_vitb16'; Stem='arm_f_vitb16_trusted_all' },
    @{ Name='adapted'; Folder='adapted_vitb16_ep3'; Stem='arm_d_vitb16_ep3_trusted_all' }
)
$repCells = @(
    foreach ($arm in $repArms) {
        foreach ($learner in @('hdc','ncm','ranpac','naive','ewc','replay')) {
            if ($arm.Name -eq 'frozen' -and $learner -eq 'ncm') { continue }
            $repTag = "option_a_section6a_v1__$($arm.Stem)__stream_0__${learner}__margin_0p1"
            [pscustomobject]@{
                Name = "$($arm.Name)_$learner"
                Source = ".\ssl_hdc_pilot\results\continual\$($arm.Folder)\${repTag}_summary.json"
                Embeddings = ".\ssl_hdc_pilot\results\$($arm.Stem).npz"
                Output = Join-Path $repRoot "$($arm.Name)_$learner"
            }
        }
    }
)
if ($repCells.Count -ne 11) { throw 'Expected exactly 11 new cells.' }
foreach ($cell in $repCells) {
    foreach ($repInput in @($repPython, $repTrain, $cell.Source, $cell.Embeddings)) {
        if (-not (Test-Path -LiteralPath $repInput -PathType Leaf)) { throw "Missing: $repInput" }
    }
}
$repCells | Select-Object Name | Format-Table -AutoSize

New-Item -ItemType Directory -Path $repRoot | Out-Null
Start-Transcript -Path (Join-Path $repRoot 'preflight_and_runs.txt') -NoClobber
try {
    rtk proxy powershell -NoProfile -File scripts/hdc.ps1 -Action env
    if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
    rtk proxy nvidia-smi
    if ($LASTEXITCODE -ne 0) { throw 'GPU preflight failed.' }
    Get-Process | Sort-Object ProcessName | Select-Object ProcessName, Id, CPU | Format-Table -AutoSize
    Write-Host 'Close competing GPU/browser/video/game/recording workloads. Do not edit code or inputs during this batch.'
    $repConfirm = Read-Host 'After reviewing preflight, type RUN 11 CELLS'
    if ($repConfirm -cne 'RUN 11 CELLS') { throw 'Cancelled before experiments.' }
    $repIndex = 0
    foreach ($cell in $repCells) {
        $repIndex++
        Write-Host "[$repIndex/11] $($cell.Name)"
        rtk proxy $repPython -u -m ssl_hdc_pilot.scripts.reconstruct_run --source-summary $cell.Source --embeddings $cell.Embeddings --trusted-split $repTrain --output-dir $cell.Output --execute --preflight-confirmed
        if ($LASTEXITCODE -ne 0) { throw "STOP: $($cell.Name) failed. Preserve this batch and send the report/error." }
    }
    Write-Host "STOP: 11 cells completed. Send the reports and transcript from $repRoot for review."
} finally {
    Stop-Transcript
}
```

The original NCM reconstruction is excluded, not overwritten or rerun.
A clean preflight is user-confirmed, not inferred from low aggregate GPU use.
The earlier first-cell transcript showed browser/overlay processes before
confirmation; it does not prove a clean timing environment. Reconstruction
PASS establishes observed-output fidelity only; no benchmark claim is made.
No official run was launched by the agent.

## Manual reconstruction remaining 96 cells - 2026-09-11

Prerequisite: all 12 representative cells have passed and their saved artifact
hashes have been verified. Completed on 2026-09-11. This block reconstructs
the other 96 historical cells, excluding every seed-0/threshold-0.10 cell.
It does not run budget experiments, held-out evaluation, or benchmarks.
The previous batch transcript ended before confirmation; the late Stop-Transcript
error was a logging failure, not a reconstruction failure. Its cause is unresolved.
This block uses direct preflight/per-cell log files, not transcript session state.

```powershell
Set-Location -LiteralPath 'G:\FOR THESIS\HDC'
$ErrorActionPreference = 'Stop'
$fullPython = '.\.venv-e1\Scripts\python.exe'
$fullTrain = '.\ssl_hdc_pilot\trusted_splits\train.csv'
$fullRoot = '.\ssl_hdc_pilot\results\reconstruction\2026-09-11_remaining96_attempt01'
$repRoot = '.\ssl_hdc_pilot\results\reconstruction\2026-09-11_representative_attempt01'
$firstPath = '.\ssl_hdc_pilot\results\reconstruction\2026-09-10_ncm_frozen_s0_t010_attempt01\fidelity_report.json'

if (Test-Path -LiteralPath $fullRoot) { throw 'Attempt exists. Preserve it and send its status; do not rerun.' }
$priorReports = @($firstPath) + @(Get-ChildItem -LiteralPath $repRoot -Recurse -Filter fidelity_report.json -File | ForEach-Object FullName)
if ($priorReports.Count -ne 12) { throw 'Expected 12 prior reports.' }
$priorTags = @()
foreach ($priorPath in $priorReports) {
    $prior = Get-Content -Raw -LiteralPath $priorPath | ConvertFrom-Json
    if ($prior.status -ne 'PASS' -or $prior.history.mismatch_count -ne 0 -or $prior.summary.mismatch_count -ne 0 -or $prior.stream_steps -ne 976 -or $prior.inputs_changed_during_run.Count -ne 0 -or $prior.code_changed_during_run.Count -ne 0) {
        throw "Prior fidelity check failed: $priorPath"
    }
    $priorTags += $prior.run_tag
    foreach ($artifact in $prior.artifacts.PSObject.Properties) {
        $artifactPath = Join-Path (Split-Path $priorPath) $artifact.Name
        if ((Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $artifact.Value) {
            throw "Prior artifact changed: $artifactPath"
        }
    }
}
if (($priorTags | Sort-Object -Unique).Count -ne 12) { throw 'Prior cells are not distinct.' }

$fullArms = @(
    @{ Name='frozen'; Folder='frozen_vitb16'; Stem='arm_f_vitb16_trusted_all' },
    @{ Name='adapted'; Folder='adapted_vitb16_ep3'; Stem='arm_d_vitb16_ep3_trusted_all' }
)
$fullCells = @(
    foreach ($arm in $fullArms) {
        foreach ($seed in @(0,1,2)) {
            foreach ($tau in @('0p05','0p1','0p2')) {
                if ($seed -eq 0 -and $tau -eq '0p1') { continue }
                foreach ($learner in @('hdc','ncm','ranpac','naive','ewc','replay')) {
                    $tag = "option_a_section6a_v1__$($arm.Stem)__stream_$($seed)__${learner}__margin_$tau"
                    if ($tag -in $priorTags) { throw "Unexpected duplicate: $tag" }
                    $name = "$($arm.Name)_s${seed}_t${tau}_$learner"
                    [pscustomobject]@{
                        Name = $name
                        Source = ".\ssl_hdc_pilot\results\continual\$($arm.Folder)\${tag}_summary.json"
                        Embeddings = ".\ssl_hdc_pilot\results\$($arm.Stem).npz"
                        Output = Join-Path $fullRoot $name
                    }
                }
            }
        }
    }
)
if ($fullCells.Count -ne 96) { throw 'Expected exactly 96 new cells.' }
foreach ($cell in $fullCells) {
    foreach ($inputPath in @($fullPython,$fullTrain,$cell.Source,$cell.Embeddings)) {
        if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) { throw "Missing: $inputPath" }
    }
}
$driveInfo = [System.IO.DriveInfo]::new((Get-Location).Drive.Root)
if ($driveInfo.AvailableFreeSpace -lt 4GB) { throw 'Keep at least 4 GiB free for this reconstruction batch.' }
Write-Host "96 new cells; free disk GiB: $([math]::Round($driveInfo.AvailableFreeSpace / 1GB, 1))"

New-Item -ItemType Directory -Path $fullRoot | Out-Null
$preflightLog = Join-Path $fullRoot 'preflight.txt'
Get-Date -Format o | Tee-Object -FilePath $preflightLog
rtk proxy powershell -NoProfile -File scripts/hdc.ps1 -Action env 2>&1 | Tee-Object -FilePath $preflightLog -Append
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
rtk proxy nvidia-smi 2>&1 | Tee-Object -FilePath $preflightLog -Append
if ($LASTEXITCODE -ne 0) { throw 'GPU preflight failed.' }
Get-Process | Sort-Object ProcessName | Select-Object ProcessName,Id,CPU |
    Format-Table -AutoSize | Out-String | Tee-Object -FilePath $preflightLog -Append

Write-Host 'Close competing GPU/browser/video/game/recording workloads. Do not edit code or inputs during the batch.'
$fullConfirm = Read-Host 'After reviewing preflight, type RUN 96 CELLS'
if ($fullConfirm -cne 'RUN 96 CELLS') { throw 'Cancelled before experiments.' }
'User confirmed preflight and RUN 96 CELLS' | Tee-Object -FilePath $preflightLog -Append

$index = 0
foreach ($cell in $fullCells) {
    $index++
    Write-Host "[$index/96] $($cell.Name)"
    $cellLog = Join-Path $fullRoot ($cell.Name + '.log')
    rtk proxy $fullPython -u -m ssl_hdc_pilot.scripts.reconstruct_run --source-summary $cell.Source --embeddings $cell.Embeddings --trusted-split $fullTrain --output-dir $cell.Output --execute --preflight-confirmed 2>&1 | Tee-Object -FilePath $cellLog
    if ($LASTEXITCODE -ne 0) { throw "STOP: $($cell.Name) failed. Preserve outputs and send its log/report." }
}
Write-Host "DONE: 96 new cells. Stop and send the reports/logs from $fullRoot."
```

The 4 GiB free-space floor is a conservative operational guard, not a measured
peak or completion guarantee. Keep all prior results and failed attempts.
Direct logs avoid Stop-Transcript, but do not prove the machine was uncontaminated.
No official run was launched by the agent.

## Reconstruction completion checkpoint - 2026-09-11

All three reconstruction blocks above are now COMPLETED user-run provenance:
1 initial cell + 11 representative cells + 96 remaining cells = 108/108 PASS.
All 648 artifact hashes and exact original-grid coverage were independently
verified. See [the audit](results/RECONSTRUCTION_AUDIT_2026-09-11.md).
Do not rerun these blocks. Subsequent Experiment-A development approval and
implementation are recorded below; this historical checkpoint is complete.
This checkpoint does not authorize held-out access or close A1-A4.
No official run was launched by the agent.

## Manual Experiment A first development cell - 2026-09-11

User-approved acquisition settings are implemented and synthetically tested.
Run only frozen-B/NCM/seed-0/B=25 here. This is training-side development, not
fixed validation evaluation, a benchmark, or the 216-cell study. Expected:
976 stream events, 25 additional labels, 2,445 labels supplied to the learner.
PASS checks schedule/accounting/provenance, not a scientific performance claim.
Logs stay outside the new cell directory; no transcript state is used.

```powershell
Set-Location -LiteralPath 'G:\FOR THESIS\HDC'
$ErrorActionPreference = 'Stop'
$aPython = '.\.venv-e1\Scripts\python.exe'
$aEmbeddings = '.\ssl_hdc_pilot\results\arm_f_vitb16_trusted_all.npz'
$aTrain = '.\ssl_hdc_pilot\trusted_splits\train.csv'
$aParent = '.\ssl_hdc_pilot\results\experiment_a_development'
$aOutput = "$aParent\2026-09-11_frozen_ncm_s0_b25_attempt01"
if (Test-Path -LiteralPath $aOutput) {
    throw 'Attempt already exists. Preserve it and send its report; do not rerun.'
}

# Read-only input/schedule check; no learner runs.
rtk proxy $aPython -m ssl_hdc_pilot.scripts.experiment_a --embeddings $aEmbeddings --trusted-split $aTrain --learner ncm --seed 0 --budget 25
if ($LASTEXITCODE -ne 0) { throw 'Input/schedule check failed.' }

New-Item -ItemType Directory -Force -Path $aParent | Out-Null
$aLog = Join-Path $aParent ("preflight_run_" + [guid]::NewGuid().ToString('N') + '.log')
rtk proxy powershell -NoProfile -File scripts/hdc.ps1 -Action env 2>&1 | Tee-Object -FilePath $aLog
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
rtk proxy nvidia-smi 2>&1 | Tee-Object -FilePath $aLog -Append
if ($LASTEXITCODE -ne 0) { throw 'GPU preflight failed.' }
Get-Process | Sort-Object ProcessName | Select-Object ProcessName, Id, CPU |
    Format-Table -AutoSize | Out-String -Width 220 | Tee-Object -FilePath $aLog -Append
Write-Host 'Review preflight. Close competing GPU workloads, games, browser/video and recording/overlay apps.'
$aConfirmation = Read-Host 'When the machine is clear, type RUN A25'
if ($aConfirmation -cne 'RUN A25') { throw 'Cancelled. No experiment launched.' }
Write-Output ("User confirmed RUN A25 at " + (Get-Date -Format o)) | Tee-Object -FilePath $aLog -Append

rtk proxy $aPython -u -m ssl_hdc_pilot.scripts.experiment_a --embeddings $aEmbeddings --trusted-split $aTrain --learner ncm --seed 0 --budget 25 --output-dir $aOutput --execute --preflight-confirmed 2>&1 | Tee-Object -FilePath $aLog -Append
if ($LASTEXITCODE -ne 0) { throw "STOP: cell failed. Preserve outputs and send $aLog." }
Write-Host "Report: $aOutput\integrity_report.json"
Write-Host "Log: $aLog"
Write-Host 'Stop here and send the integrity report for review.'
```

No Experiment-A execution was performed by the agent. Do not extrapolate this
single development check into permission to run the full grid or open test data.

## Manual A25 saved-state validation - 2026-09-11

The preceding A25 training block is COMPLETED: user-run PASS, all seven artifact
hashes independently verified. Do not rerun it. This new block only loads its
saved final learner and scores the fixed 1,132-image validation population.
No training, test access, model selection or benchmarking. It remains development
evidence, not a full-study result. Return both integrity_report.json and metrics.json.

```powershell
Set-Location -LiteralPath 'G:\FOR THESIS\HDC'
$ErrorActionPreference = 'Stop'
$vPython = '.\.venv-e1\Scripts\python.exe'
$vSource = '.\ssl_hdc_pilot\results\experiment_a_development\2026-09-11_frozen_ncm_s0_b25_attempt01'
$vEmbeddings = '.\ssl_hdc_pilot\results\arm_f_vitb16_trusted_all.npz'
$vSplit = '.\ssl_hdc_pilot\trusted_splits\val.csv'
$vParent = '.\ssl_hdc_pilot\results\validation_development'
$vOutput = "$vParent\2026-09-11_frozen_ncm_s0_b25_attempt01"
if (Test-Path -LiteralPath $vOutput) { throw 'Validation attempt exists. Preserve it and send its reports.' }

# Preparation only: no learner loaded and no predictions.
rtk proxy $vPython -m ssl_hdc_pilot.scripts.evaluate_validation --source-dir $vSource --embeddings $vEmbeddings --validation-split $vSplit
if ($LASTEXITCODE -ne 0) { throw 'Validation input check failed.' }

New-Item -ItemType Directory -Force -Path $vParent | Out-Null
$vLog = Join-Path $vParent ('preflight_run_' + [guid]::NewGuid().ToString('N') + '.log')
rtk proxy powershell -NoProfile -File scripts/hdc.ps1 -Action env 2>&1 | Tee-Object -FilePath $vLog
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
rtk proxy nvidia-smi 2>&1 | Tee-Object -FilePath $vLog -Append
if ($LASTEXITCODE -ne 0) { throw 'GPU preflight failed.' }
Get-Process | Sort-Object ProcessName | Select-Object ProcessName, Id, CPU |
    Format-Table -AutoSize | Out-String -Width 220 | Tee-Object -FilePath $vLog -Append
Write-Host 'Review preflight. Close competing GPU workloads, games, browser/video and recording/overlay apps.'
$vConfirmation = Read-Host 'When the machine is clear, type RUN VAL'
if ($vConfirmation -cne 'RUN VAL') { throw 'Cancelled. No scoring launched.' }
Write-Output ('User confirmed RUN VAL at ' + (Get-Date -Format o)) | Tee-Object -FilePath $vLog -Append

rtk proxy $vPython -u -m ssl_hdc_pilot.scripts.evaluate_validation --source-dir $vSource --embeddings $vEmbeddings --validation-split $vSplit --output-dir $vOutput --execute --preflight-confirmed 2>&1 | Tee-Object -FilePath $vLog -Append
if ($LASTEXITCODE -ne 0) { throw "STOP: validation failed. Preserve outputs and send $vLog." }
Write-Host "Integrity report: $vOutput\integrity_report.json"
Write-Host "Metrics: $vOutput\metrics.json"
Write-Host "Log: $vLog"
Write-Host 'Stop here and send the integrity report and metrics for review.'
```

No official run was launched by the agent. This block does not authorize any
other budget condition or trusted-test evaluation.

## A25 training and validation completion checkpoint - 2026-09-11

Both A25 blocks above are now COMPLETED user-run provenance; do not rerun them.
Validation receipt PASS: 1,132 images/19 classes, unchanged state/RNG. Independent
audit checked all three validation artifact hashes and reconstructed common
classification metrics/ECE from saved predictions. See VALIDATION_EVALUATION.md.

The agent reads saved reports and performs artifact/analysis checks directly.
There is no new launch block at this checkpoint: the next implementation is
gated on the Experiment-B and longitudinal setting decisions documented in
LONGITUDINAL_METRICS.md. No official run was launched by the agent.

## Manual B400 snapshot development - 2026-09-12

COMPLETED user-run PASS, independently verified: 27 artifacts, 976 events,
400 purchases and 20 states. DO NOT RERUN the training block below. Its manual
paste split Tee-Object from -FilePath; preflight/confirmation are in the original
log and training output is in b400_recovery_01.log. The later shell error did
not invalidate the saved artifacts. Next action is the script launch below.

The approval gate described above is now resolved for bounded development.
This new block trains ONE frozen-B/NCM/seed-0 Experiment-B cell, allowance 400,
threshold .10, with approved retention/uptake snapshots. It does not rerun A25
or historical cells, score validation/test, benchmark, or launch the full grid.
Expect 976 events, at most 400 additional labels (at most 2,820 supplied learner
labels), ten retention times plus available uptake times, with overlaps combined.
The 4-GiB free-space guard is operational headroom, not a measured storage claim.

```powershell
Set-Location -LiteralPath 'G:\FOR THESIS\HDC'
$ErrorActionPreference = 'Stop'
$bPython = '.\.venv-e1\Scripts\python.exe'
$bEmbeddings = '.\ssl_hdc_pilot\results\arm_f_vitb16_trusted_all.npz'
$bTrain = '.\ssl_hdc_pilot\trusted_splits\train.csv'
$bParent = '.\ssl_hdc_pilot\results\budget_longitudinal_development'
$bOutput = "$bParent\2026-09-12_b_frozen_ncm_s0_b400_t010_attempt01"
if (Test-Path -LiteralPath $bOutput) { throw 'Attempt exists. Preserve it; tell me it exists.' }
if ((Get-PSDrive -Name G).Free -lt 4GB) { throw 'Less than 4 GiB free. Stop before training.' }

# Input plan only: no learner constructed.
rtk proxy $bPython -m ssl_hdc_pilot.scripts.budget_experiment --experiment B --threshold 0.1 --embeddings $bEmbeddings --trusted-split $bTrain --learner ncm --seed 0 --budget 400
if ($LASTEXITCODE -ne 0) { throw 'Input plan failed.' }

New-Item -ItemType Directory -Force -Path $bParent | Out-Null
$bLog = Join-Path $bParent ('preflight_run_' + [guid]::NewGuid().ToString('N') + '.log')
rtk proxy powershell -NoProfile -File scripts/hdc.ps1 -Action env 2>&1 | Tee-Object -FilePath $bLog
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
rtk proxy nvidia-smi 2>&1 | Tee-Object -FilePath $bLog -Append
if ($LASTEXITCODE -ne 0) { throw 'GPU preflight failed.' }
Get-Process | Sort-Object ProcessName | Select-Object ProcessName, Id, CPU |
    Format-Table -AutoSize | Out-String -Width 220 | Tee-Object -FilePath $bLog -Append
Write-Host 'Review preflight. Close competing GPU workloads, games, browser/video and recording/overlay apps.'
$bConfirmation = Read-Host 'When the machine is clear, type RUN B400'
if ($bConfirmation -cne 'RUN B400') { throw 'Cancelled. No training launched.' }
Write-Output ('User confirmed RUN B400 at ' + (Get-Date -Format o)) | Tee-Object -FilePath $bLog -Append

rtk proxy $bPython -u -m ssl_hdc_pilot.scripts.budget_experiment --experiment B --threshold 0.1 --embeddings $bEmbeddings --trusted-split $bTrain --learner ncm --seed 0 --budget 400 --output-dir $bOutput --execute --preflight-confirmed 2>&1 | Tee-Object -FilePath $bLog -Append
if ($LASTEXITCODE -ne 0) { throw "STOP: run failed. Preserve outputs and log: $bLog" }
Write-Host "Report: $bOutput\integrity_report.json"
Write-Host "Log: $bLog"
Write-Host 'Done. Tell me done; I will inspect the saved files directly.'
```

No real experiment was launched by the agent. Do not pass this new artifact
format to the old A25 validation block; new-format evaluation integration is
still pending. See BUDGET_SNAPSHOTS.md.

## Manual B400 longitudinal validation - 2026-09-12

COMPLETED user-run PASS, independently verified: 22 hashes, 48 input pins and
all 20 fixed-population ledgers/derived metrics. DO NOT RERUN the command below;
it remains completed-run provenance. Next input is benchmark policy approval.

New evaluator/launcher ready; real scoring has NOT been launched by the agent.
This scores 20 EXISTING states on the same 1,132 validation examples. No training,
test access, calibration fitting, checkpoint selection or benchmark. It preserves
the completed B400 training directory and refuses an existing validation attempt.

```powershell
rtk proxy powershell -NoProfile -File .\scripts\validate-b400-snapshots.ps1
```

Review the recorded environment/GPU/process preflight, clear competing workloads,
then type RUN SNAPSHOT VAL. Logging paths are supplied inside the script; do not
paste the old multiline block or supply FilePath manually. Optional -PlanOnly
runs just input checks without loading a learner. When complete, tell the agent
done; it reads artifacts/metrics directly. See SNAPSHOT_VALIDATION.md.

## Historical A25 benchmark attempt01 - blocked before setup, 2026-09-13

Do not rerun the old default command as an attempt01 recovery. The user launch
stopped at the initial guard before learner setup/timing. Preserve its failure
report and preflight_run_92f3f17a44ac48b7845f7a7048db0643.log. The block below
records the original instructions; use the recovery block at the end instead.

Approved bounded development policy; this is ONE frozen-B/seed-0/A25 condition,
not the full grid. Includes new reference/warm-up/update passes, five paired
throughput/latency blocks and separate memory/inventory passes for all six heads.
Do not confuse this fresh timing workload with rerunning completed A25/B400
prediction or validation blocks. No test/validation scoring or model selection.

```powershell
rtk proxy powershell -NoProfile -File .\scripts\benchmark-a25.ps1
```

Review preflight and close competing workloads (including browser background
processes and recording/overlay apps), then type RUN BENCH A25. Keep the machine
idle. Optional -PlanOnly performs only read-only input checks. The script binds
the log path internally, requires 8 GiB operational headroom and refuses existing
outputs. A detected competing workload or mismatch stops and preserves the attempt.
Output: results/benchmark_development/2026-09-13_a_frozen_s0_b25_attempt01, plus
adjacent unique log. No automatic retry or deletion. Tell the agent done after
completion; it audits files directly. See BENCHMARKS.md.

No real benchmark or other official run was launched by the agent.

## Manual A25 benchmark recovery - fresh attempt02, 2026-09-13

COMPLETED: attempt02 PASS and independently audited. Do not execute this block
again. See results/A25_BENCHMARK_AUDIT_2026-09-13.md. First benchmark acceptance
does not authorize full-grid launches, tuning or final test access.

User-approved exact-path desktop baseline and logging repair. Attempt01 failed
before learner setup/timing; its original report/log remain unchanged.
Close Overwolf, browsers/background WebView apps, games, recording/overlay apps
and unnecessary GPU apps (including Settings and Sticky Notes). Do not kill
Windows shell processes. Only approved exact desktop/Terminal/VS Code paths
are accepted and recorded; this is a shared desktop, not an isolated GPU.

```powershell
rtk proxy powershell -NoProfile -File "G:\FOR THESIS\HDC\scripts\benchmark-a25.ps1" -Attempt 2
```

The script checks inputs, disk headroom, environment and current workloads
before asking for RUN BENCH A25. A blocked preflight does not create attempt02.
If clear, review the recorded baseline and type RUN BENCH A25, then keep the
machine idle. Native stderr is fully logged before exit status is checked.
Output: results/benchmark_development/2026-09-13_a_frozen_s0_b25_attempt02 and
an adjacent unique preflight log. Existing output is refused, never reused or
auto-incremented. Optional -PlanOnly does input planning only, not a cleanliness
check. The agent has not launched a real benchmark; tell it done after completion.

## Manual A25 shared-desktop acceptance - fresh attempt04, 2026-09-14

COMPLETED: attempt04 PASS and independently audited. DO NOT EXECUTE THIS BLOCK
AGAIN. See results/A25_COMPACT_BENCHMARK_AUDIT_2026-09-14.md. Shared storage is
accepted for this bounded condition; broader feasibility/test release remain open.

User-approved amendment accepts/logs the identified Windows Settings process,
including automatic recurrence/PID changes, within the existing desktop baseline.
It does not require suppressing ordinary accepted OS activity. Avoid games,
recording/overlays, browser workloads and other training jobs. See the dated
BENCHMARK_DEVELOPMENT_PLAN.md amendment; no isolated-machine claim is permitted.

Attempt03 stopped during block4 under the prior Settings exclusion. Preserve it
and preflight_run_655e0b085a5d4566b39606f9653180dd.log; no resume, retrospective PASS,
or mixing partial timing blocks into this fresh attempt. Attempt02 remains
completed provenance. The agent has not launched a real benchmark.

Paste the entire block so a failed launcher cannot fall through to report reads:

```powershell
& {
    Set-Location -LiteralPath 'G:\FOR THESIS\HDC'
    $ErrorActionPreference = 'Stop'

    rtk proxy powershell -NoProfile -File "G:\FOR THESIS\HDC\scripts\benchmark-a25.ps1" -Attempt 4

    if ($LASTEXITCODE -ne 0) {
        throw 'Preflight or run failed. Stop here and preserve the log.'
    }

    Get-Content -LiteralPath 'G:\FOR THESIS\HDC\ssl_hdc_pilot\results\benchmark_development\2026-09-13_a_frozen_s0_b25_attempt04\integrity_report.json'
}
```

Review preflight and type RUN BENCH A25 when prompted. Existing attempt04 is
refused, never overwritten or automatically incremented. The retained date prefix
identifies the launcher series; logs/manifests record the actual launch. This is
one six-learner development condition with real updates, not a full-grid release.

## Completed B400 feasibility benchmark - attempt01, 2026-09-14

COMPLETE and independently audited. Do not execute this historical block again.
See results/B400_BENCHMARK_AUDIT_2026-09-14.md. All six learners/five blocks passed;
background warnings are retained, not a reason to rerun. No further run is
authorized by this acceptance. The preparation text/command below is provenance.

User removed the total runtime cap. Next bounded measurement is frozen-B/seed0/
Experiment-B/allowance400/threshold0.10, all six learners and five paired blocks.
Actual purchases may differ by learner; do not assume the allowance is exhausted.
No full-grid/untouched-test release. Completed A25 benchmarks and NCM B400
capture/validation remain preserved. This new timing run includes real updates
and is user-owned. The agent has not launched it.

Paste the whole block:

```powershell
& {
    Set-Location -LiteralPath 'G:\FOR THESIS\HDC'
    $ErrorActionPreference = 'Stop'

    rtk proxy powershell -NoProfile -File "G:\FOR THESIS\HDC\scripts\benchmark-b400.ps1" -Attempt 1

    if ($LASTEXITCODE -ne 0) {
        throw 'Preflight or run failed. Stop here and preserve the log.'
    }

    Get-Content -LiteralPath 'G:\FOR THESIS\HDC\ssl_hdc_pilot\results\benchmark_development\2026-09-14_b_frozen_s0_b400_t010_attempt01\integrity_report.json'
}
```

Review the recorded preflight and type RUN BENCH B400. Background apps may recur:
the September14 background-warning amendment logs unclassified GPU entries as
PASS_WITH_WARNINGS (exit0), not automatic aborts. Known browser/game/recording/
overlay/Python workload exclusions still block. Warning-bearing reports explicitly
retain their shared-desktop qualification; integrity PASS is not clean timing. The
same shared-desktop limitations apply across all methods.8GiB is a prelaunch
headroom guard, not unlimited storage or a guaranteed mid-run reserve. Existing
attempts fail closed without deletion/resume/automatic numbering. Tell the agent
done afterward; it reads the saved reports directly without rerunning training.
