# SSL-HDC Pilot Results Log

Append-only. Metrics must never be reported as accuracy alone.

## 2026-07-18 — implementation session opened

- Scope: scaffold, dependency verification, leakage-safe data plumbing,
  Arm-F smoke, split dry-run tests, and Arm-D ambiguity guard.
- Official run status: none.

No official run was launched by the agent.

## 2026-07-18 - unlabeled manifest construction and leakage audit

- Command: `.venv-e1\Scripts\python.exe -u ssl_hdc_pilot\scripts\build_unlabeled_manifest.py --output-csv ssl_hdc_pilot\results\unlabeled_manifest.csv --summary-json ssl_hdc_pilot\results\unlabeled_manifest_summary.json`.
- Artifacts: `results/unlabeled_manifest.csv` and
  `results/unlabeled_manifest_summary.json`.
- Train folders: 51,906; unique manifest IDs: 51,906.
- Val duplicate subset: 5,193/5,193 IDs already present in train and not added
  a second time.
- Trusted folders checked: 5,767; trusted overlap in manifest: 0.
- Labels used: no.

No official run was launched by the agent.

## 2026-07-18 - Arm-F frozen-encoder smoke

- Scope: first 50 IDs from the sanctioned unlabeled manifest only.
- Model: `facebook/dinov3-vits16-pretrain-lvd1689m`, loaded offline from the
  existing cache; frozen and in eval mode.
- Device/batch: CUDA, batch size 16.
- Command: `.venv-e1\Scripts\python.exe -u ssl_hdc_pilot\scripts\extract_embeddings.py --input-csv ssl_hdc_pilot\results\unlabeled_manifest.csv --output-npz ssl_hdc_pilot\results\arm_f_smoke_50.npz --metadata-json ssl_hdc_pilot\results\arm_f_smoke_50_metadata.json --batch-size 16 --device cuda --smoke --max-images 50`.
- Artifact: `results/arm_f_smoke_50.npz` plus metadata JSON.
- Result: 50 embeddings, shape `(50, 384)`, dtype `float32`; elapsed extraction
  time 1.680 seconds after model loading.
- This was a permitted tiny smoke, not a full extraction.

No official run was launched by the agent.

## 2026-07-18 - tests and Arm-D blocker

- Full suite: 14 tests passed.
- Covered seams: manifest trusted-ID exclusion, embedding shape/dtype and
  model/budget guards, deterministic stratification, supervisor write guard,
  and Arm-D budget/mitigation guards.
- Arm-D smoke was not launched. Its required labeled probe policy remains
  ambiguous, so the entry point exits before loading data or a model.
- No accuracy-only metric was produced.
- Post-review hardening locks official split writes to canonical `Data/test.csv`,
  the preregistered 5,767-row/25-class source, exact six-class Option-A tail,
  19 retained classes, seed 20260716, and Option-B 5-fold x 10-repeat settings.
  Both report
  types now retain all mandatory disclosure statements.

No official run was launched by the agent.

## 2026-07-19 — Official Arm-F extraction complete (user-launched)

Both official frozen-DINOv3 extraction runs completed on the RTX 5070 from
the G: HF cache (vits16, CLS pooling, fp32):

- `arm_f_trusted_all.npz`: (5767, 384), 72.7 s, no NaN, no zero rows,
  image_ids stored alongside embeddings.
- `arm_f_unlabeled_pool.npz`: (51906, 384), 592.4 s, no NaN, no zero rows.

Verified by the supervising agent (shape/dtype/NaN/zero-row/count checks
against metadata). No label was read at any point; no quality probe was run
on trusted embeddings — deliberately deferred to the official probe gate
after the supervisor-approved splits exist. Arm F is complete.

## 2026-07-19 - Arm-D amendment implementation and bounded smoke

- Implemented label-free DINO self-distillation using the sanctioned manifest
  only: ViT-S/16, EMA teacher, two global plus four local crops, AMP, gradient
  checkpointing, and atomic per-epoch resume checkpoints.
- Implemented the fixed post-split logistic-regression gate from the dated
  2026-07-18 section 5.2 amendment. Selection metric is validation macro-F1;
  no accuracy-only result was produced.
- Smoke command scope: eight manifest images, batch size 2, one optimization
  step, one epoch, CUDA on the RTX 5070.
- Smoke result: loss 8.123485 in both checks; training-loop elapsed time was
  1.7294 seconds initially and 1.0033 seconds after final guard hardening, each
  for two processed images. Complete `checkpoint_epoch0` and
  `checkpoint_epoch1` were written; identical reruns detected epoch 1 and
  exited as a no-op.
- Linear extrapolation of the two measured training-loop throughputs to 51,906
  images x 5 epochs at batch size 2 spans approximately 36.16-62.34 GPU-hours.
  This is a rough tiny-smoke range; the user-launch block budgets about 65
  hours.
- The temporary `results/arm_d_smoke_20260719/` artifacts were removed after
  this entry was recorded.
- Unit suite before the smoke: 19 passed.

No official run was launched by the agent.

## 2026-07-19 - post-review verification

- Parallel standards and binding-spec reviews were applied.
- Fixed official commands to use package-safe module execution.
- Hardened checkpoint completeness to reject malformed NPZ state, retained all
  manifest remainder images, and locked the probe input to exact Option-A
  3,396/1,132 row counts, the exact 19 retained classes, canonical trusted
  labels, and raw 384-d embeddings. Epoch-0 reuse now requires matching state
  schema/provenance and is strict-loaded before resumed training.
- Final unit suite: 21 passed in 5.50 seconds. Modified scripts also compile.
- No accuracy-only metric was produced.

No official run was launched by the agent.

## 2026-07-19 - Round 3 launch hardening and CUDA continuation smoke

- Corrected the inference contract: every adapted `encoder.npz` now contains
  the EMA teacher used by the probe/downstream path; the student encoder is
  stored separately inside `training_state.npz` for exact continuation.
- Added a per-step non-finite-loss abort before backward, optimizer, EMA, or
  center updates. The probe now refuses anything except the complete epoch
  cadence declared by the run's immutable configuration.
- Fresh-process CUDA smoke scope: eight sanctioned images, batch size 2, two
  epochs, one optimization step per epoch (two total). Process 1 wrote epoch 1
  and stopped; process 2 restored epoch 1 and wrote epoch 2; process 3 exited as
  an already-complete no-op.
- Results: epoch-1 loss 8.123485 in 1.7065 seconds; epoch-2 loss 8.183393 in
  0.9337 seconds. Both were finite. Epoch-2 metadata records
  `process_resumed_from_epoch=1`; full cadence 0/1/2 validated.
- The exported EMA teacher and resumable student were distinct after training
  (observed maximum tensor difference 1.9997e-05), proving the gate artifact is
  not an accidental student copy.
- Current tiny-smoke extrapolation remains within the earlier conservative
  launch budget: approximately 33.65-61.51 GPU-hours versus a 65-hour budget.
- Full unit suite: 27 passed in 9.8 seconds; modified scripts compiled. Resume
  refuses checkpoint cadence gaps, exact optimizer/head restoration is tested,
  and the official gate refuses smoke checkpoint directories.
- Temporary `results/arm_d_round3_resume_smoke_20260719/` artifacts were
  removed after this entry was recorded.
- No accuracy-only metric was produced.

No official run was launched by the agent.

## 2026-07-19 - Arm-D CUDA batch-size and worker throughput smoke

- Engineering-only sweep: ViT-S/16, sanctioned manifest, one smoke epoch,
  160-image cap, exactly 10 optimization steps per setting, CUDA, AMP,
  gradient checkpointing, two global plus four local crops, EMA 0.996, and
  learning rate 1e-5. Only batch size and DataLoader worker count varied.
- Peak VRAM is whole-device used memory polled with `nvidia-smi` at 100 ms
  intervals across each process. The RTX 5070 reported 12,227 MiB total; idle
  baselines varied from 1,021 to 1,261 MiB. The selection ceiling was strictly
  below 9,781.6 MiB (80%).

| Batch | Workers | Steps | Images | Loop seconds | Steps/s | Images/s | Peak VRAM MiB | Peak of GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 10 | 20 | 4.5416 | 2.2018 | 4.4036 | 2,093 | 17.12% |
| 2 | 2 | 10 | 20 | 16.0562 | 0.6228 | 1.2456 | 2,085 | 17.05% |
| 4 | 0 | 10 | 40 | 5.1606 | 1.9378 | 7.7512 | 1,881 | 15.38% |
| 4 | 2 | 10 | 40 | 16.2365 | 0.6159 | 2.4636 | 1,898 | 15.52% |
| 8 | 0 | 10 | 80 | 6.4723 | 1.5451 | 12.3608 | 1,950 | 15.95% |
| 8 | 2 | 10 | 80 | 14.6715 | 0.6816 | 5.4528 | 1,949 | 15.94% |
| 16 | 0 | 10 | 160 | 8.7587 | 1.1417 | 18.2672 | 2,155 | 17.62% |
| 16 | 2 | 10 | 160 | 14.8752 | 0.6723 | 10.7568 | 2,168 | 17.73% |

- Every setting completed with finite loss and validated labels/trusted-pixels
  provenance set to false. No official output directory was created.
- Selected batch size 16 because it is the largest tested candidate and its
  2,155 MiB peak with workers 0 is far below the 80% ceiling. Selected workers
  0 because 1.1417 steps/s is 69.8% faster than 0.6723 steps/s for workers 2 at
  batch 16 on this Windows host.
- Measured-basis estimate: `ceil(51,906 / 16) * 5 = 16,225` optimization
  steps; at 1.1417 steps/s, training-loop time is approximately 3.95 hours.
  Allow 5 wall-clock hours for model startup, six checkpoint writes, final
  partial batches, and smoke-to-full-run variability.
- The eight temporary smoke checkpoint directories and their temporary
  stdout/stderr logs were removed after validation and recording.
- No accuracy-only metric was produced.

No official run was launched by the agent.

## 2026-07-19 - Arm-D extended CUDA throughput smoke for batches 32 and 64

- Follow-up engineering sweep retained the sanctioned manifest, ViT-S/16,
  CUDA/AMP, gradient checkpointing, EMA 0.996, two global plus four local
  crops, learning rate 1e-5, one smoke epoch, `--max-images 512`, and
  `--max-steps 10`. Only batch size and DataLoader workers varied.
- Batch 32 completed ten steps/320 images per worker setting. Batch 64
  naturally completed eight steps/512 images because the image cap was reached.
  Peak whole-device VRAM was polled with `nvidia-smi` at 100 ms intervals.

| Batch | Workers | Steps | Images | Loop seconds | Steps/s | Images/s | Peak VRAM MiB | Peak of GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0 | 10 | 320 | 15.8750 | 0.6299 | 20.1574 | 2,526 | 20.66% |
| 32 | 2 | 10 | 320 | 20.8328 | 0.4800 | 15.3604 | 2,523 | 20.63% |
| 64 | 0 | 8 | 512 | 22.6448 | 0.3533 | 22.6101 | 3,238 | 26.48% |
| 64 | 2 | 8 | 512 | 25.7302 | 0.3109 | 19.8988 | 3,238 | 26.48% |

- Every setting completed with finite loss and labels/trusted-pixels provenance
  set to false. The 9,781.6 MiB 80% ceiling was not approached.
- Selected batch size 64 because it is the largest tested setting and its
  3,238 MiB peak with workers 0 is only 26.48% of the 12,227 MiB device.
  Selected workers 0 because 0.3533 steps/s is 13.6% faster than 0.3109
  steps/s for workers 2 at batch 64.
- Measured-basis estimate: `ceil(51,906 / 64) * 5 = 4,060` optimization
  steps; at 0.3533 steps/s, training-loop time is approximately 3.19 hours.
  Allow approximately 4 wall-clock hours for startup, six checkpoint writes,
  final partial batches, and smoke-to-full-run variability.
- The four extended-smoke checkpoint directories and their temporary
  stdout/stderr logs were removed after validation and recording.
- No accuracy-only metric was produced.

No official run was launched by the agent.

## 2026-07-19 - Authoritative Arm-D 12-setting idle-GPU throughput sweep

- Idle condition verified before measuring: no Python/CUDA compute workload,
  no video stream or other meaningful GPU workload, 694 MiB total device use,
  and 2% utilization. Per-setting idle guards observed 759-776 MiB and 0-5%
  utilization; only normal Windows/UI processes were listed by `nvidia-smi`.
- Sweep recipe was identical across settings: sanctioned manifest, ViT-S/16,
  one smoke epoch, `--max-images 512`, `--max-steps 10`, LR 1e-5, EMA
  0.996, two global plus four local crops, AMP, gradient checkpointing, and
  CUDA. Only batch size and `num_workers` varied. Batch 64 reached the
  512-image cap in eight steps; every other setting completed ten steps.
- PyTorch peak is `torch.cuda.max_memory_allocated` for the timed epoch.
  NVIDIA peak is 100 ms polling of whole-device total-used memory.

| Batch | Workers | Steps | Steps/s | Torch peak MiB | NVIDIA peak total MiB | OOM |
|---:|---:|---:|---:|---:|---:|:---:|
| 2 | 0 | 10 | 2.8650 | 564.65 | 1,600 | No |
| 2 | 2 | 10 | 0.8609 | 564.65 | 1,600 | No |
| 4 | 0 | 10 | 2.4769 | 567.25 | 1,595 | No |
| 4 | 2 | 10 | 0.8505 | 567.25 | 1,595 | No |
| 8 | 0 | 10 | 1.8751 | 590.26 | 1,691 | No |
| 8 | 2 | 10 | 0.8207 | 590.26 | 1,691 | No |
| 16 | 0 | 10 | 1.2906 | 741.68 | 1,901 | No |
| 16 | 2 | 10 | 0.7313 | 741.68 | 1,901 | No |
| 32 | 0 | 10 | 0.7580 | 1,052.03 | 2,241 | No |
| 32 | 2 | 10 | 0.5437 | 1,052.03 | 2,241 | No |
| 64 | 0 | 8 | 0.4232 | 1,674.86 | 2,951 | No |
| 64 | 2 | 8 | 0.3858 | 1,674.86 | 2,951 | No |

- Fixed headroom rule: 75% of 12 GiB is 9,216 MiB. For the selected batch
  32/workers 0 setting, NVIDIA total-used peak was 2,241 MiB, and the separate
  conservative PyTorch-plus-video margin was `1,052.03 + 1,024 = 2,076.03`
  MiB. Both are strictly below 9,216 MiB, so the rule already budgets for
  approximately 1 GiB of video-streaming use during the official run.
- Fixed schedule rule: batch 32 yields
  `ceil(51,906 / 32) * 5 = 1,623 * 5 = 8,115` optimizer updates and passes
  the 8,000-update floor. Batch 64 yields only 4,060 updates and is excluded
  regardless of throughput or memory.
- Mechanical selection: batch 32 is the largest schedule-eligible batch, and
  workers 0 is faster than workers 2 (0.7580 versus 0.5437 steps/s).
- Wall-time basis: `1,623 steps/epoch * 5 epochs / 0.7580 steps/s =
  10,705.8 seconds = 2.97 training-loop hours`; allow approximately 4
  wall-clock hours for startup, six checkpoint writes, and run variability.
- All losses were finite; labels/trusted-pixels provenance remained false.
- All 12 sweep checkpoint directories and 24 temporary stdout/stderr logs were
  removed after validation and recording.
- Full pilot suite: 28 tests passed; modified Python files compiled.
- No accuracy-only metric was produced.

No official run was launched by the agent.

## 2026-07-19 - ViT-B / ViT-L feasibility measurement

- This was a smoke-only feasibility measurement. Immediately before each
  model family, `nvidia-smi` showed no Python/CUDA compute process and no
  meaningful competing GPU workload. The ViT-B gate showed 837 MiB total
  device use at 4% utilization; the ViT-L gate showed 837 MiB at 4%.
- Recipe was held identical to the authoritative ViT-S sweep: sanctioned
  manifest, LR 1e-5, EMA momentum 0.996, two global plus four local crops,
  AMP, gradient checkpointing, CUDA, workers 0, `--max-images 512`, and
  `--max-steps 10`. Only model ID and batch size varied.
- PyTorch peak is `torch.cuda.max_memory_allocated`; NVIDIA peak is polled
  whole-device total-used memory. Every setting completed ten measured steps.

| Model | Batch | Workers | Steps | Steps/s | Torch peak MiB | NVIDIA peak total MiB | OOM |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| ViT-B/16 | 16 | 0 | 10 | 0.9999 | 2,120.03 | 3,417 | No |
| ViT-B/16 | 32 | 0 | 10 | 0.5307 | 2,705.91 | 4,006 | No |
| ViT-L/16 | 16 | 0 | 10 | 0.6083 | 7,044.23 | 8,660 | No |
| ViT-L/16 | 32 | 0 | 10 | 0.3091 | 7,583.83 | 9,780 | No |

- ViT-B/16 fixed-rule outcome: batch 32 passes. Its NVIDIA peak is
  `4,006 < 9,216` MiB and its separate PyTorch-plus-video check is
  `2,705.91 + 1,024 = 3,729.91 < 9,216` MiB. It retains
  `ceil(51,906 / 32) * 5 = 1,623 * 5 = 8,115` optimizer updates, so batch
  32 is the largest passing measured batch. Projected training-loop time is
  `1,623 * 5 / 0.5307 = 15,291.5 seconds = 4.25 hours`; using the same
  checkpoint/startup/variability allowance method as the ViT-S estimate gives
  approximately 6 wall-clock hours.
- ViT-L/16 fixed-rule outcome: batch 32 fails the whole-device headroom rule
  because `9,780 > 9,216` MiB, although its separate PyTorch-plus-video value
  is `7,583.83 + 1,024 = 8,607.83 < 9,216` MiB. Batch 16 passes both checks:
  `8,660 < 9,216` MiB whole-device and
  `7,044.23 + 1,024 = 8,068.23 < 9,216` MiB. It retains
  `ceil(51,906 / 16) * 5 = 3,245 * 5 = 16,225` optimizer updates, so batch
  16 is the largest passing measured batch. Projected training-loop time is
  `3,245 * 5 / 0.6083 = 26,672.7 seconds = 7.41 hours`; the same allowance
  method gives approximately 10 wall-clock hours.
- All losses were finite; labels/trusted-pixels provenance remained false.
- Measurement only: no model selection, preregistration amendment, or
  official command change was made. The user and supervising agent decide
  whether to upgrade both encoder arms.
- All four feasibility checkpoint directories and eight temporary
  stdout/stderr logs were removed after validation and recording.
- Full pilot suite: 29 tests passed; modified Python files compiled.
- No accuracy-only metric was produced.

No official run was launched by the agent.

## 2026-07-19 - ViT-L feasibility selection-rule correction

- This append-only correction supersedes only the ViT-L selection paragraph
  immediately above. The current measurement request defines headroom as
  PyTorch peak plus the fixed 1,024 MiB video margin, not NVIDIA whole-device
  peak as an additional rejection criterion.
- ViT-L/16 batch 32 therefore passes the specified headroom check:
  `7,583.83 + 1,024 = 8,607.83 < 9,216` MiB. It also passes the schedule
  rule with `ceil(51,906 / 32) * 5 = 1,623 * 5 = 8,115` optimizer updates,
  so batch 32 is the largest passing measured ViT-L batch.
- Its required NVIDIA whole-device telemetry remains reported above at
  9,780 MiB; this is an operational warning for the pending supervisor/user
  decision, but it is not a rejection condition in the enumerated rule.
- Corrected projected training-loop time:
  `1,623 * 5 / 0.3091 = 26,253.6 seconds = 7.29 hours`. Applying the same
  checkpoint/startup/variability allowance method as the ViT-S estimate gives
  9.81 hours, rounded to approximately 10 wall-clock hours.
- Measurement report only: no model selection, preregistration amendment, or
  official command change was made.
- No accuracy-only metric was produced.

No official run was launched by the agent.

## 2026-07-19 - Encoder-size LADDER implementation and three-rung smokes

- Implemented the binding ViT-S/16, ViT-B/16, and ViT-L/16 ladder without
  changing the preregistered training recipe.
- Arm-D accepts exactly the three ladder model IDs and binds each immutable run
  config to its size-specific directory: `arm_d_vits16`, `arm_d_vitb16`,
  or `arm_d_vitl16`.
- Frozen extraction now validates and records the model ID, rung, and expected
  embedding dimension: 384, 768, or 1,024.
- The probe gate derives the model ID and expected dimension from the checkpoint
  directory's run config, validates every checkpoint against it, and includes
  the rung in the verdict.
- Targeted ladder guard suite: 24 tests passed. This includes a hard-error when
  a ViT-B checkpoint run is paired with 384-d ViT-S trusted features.
- Permitted CUDA smokes used two sanctioned unlabeled images, batch size 2, one
  epoch, and exactly one optimization step per rung:

| Rung | Model ID | Loss | Checkpoint directory |
|:---|:---|---:|:---|
| vits16 | `facebook/dinov3-vits16-pretrain-lvd1689m` | 8.130706 | `G:\tmp\ssl_hdc_ladder_smokes\arm_d_vits16` |
| vitb16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 8.130492 | `G:\tmp\ssl_hdc_ladder_smokes\arm_d_vitb16` |
| vitl16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 8.115825 | `G:\tmp\ssl_hdc_ladder_smokes\arm_d_vitl16` |

- Every loss was finite, and each directory contained complete
  `checkpoint_epoch0` and `checkpoint_epoch1` artifacts under the correct
  rung name.
- Cleanup status at log time: the isolated smoke tree is ready for immediate
  deletion; cleanup completion is recorded in the next entry.
- No accuracy-only metric was produced.

No official run was launched by the agent.

## 2026-07-19 - Encoder-size LADDER smoke cleanup

- Deleted the fully logged `G:\tmp\ssl_hdc_ladder_smokes` tree, including
  all three rung checkpoint directories.
- Verified the smoke root no longer exists.

No official run was launched by the agent.

## 2026-07-19 — Encoder-size ladder Arm-D COMPLETE (user-launched); Arm-F B/L extractions complete

All three Arm-D rungs finished, full 5-epoch cadence each, verified by the
supervising agent (run_config model ids and recipe exact, smoke=false,
losses finite, EMA-teacher encoder.npz in every checkpoint_epoch0-5):

- arm_d_vits16: 2.69 h, final epoch loss 0.0467
- arm_d_vitb16: 3.27 h, final epoch loss 0.0343
- arm_d_vitl16: 5.68 h, final epoch loss 0.0226 (11.6 h ladder total)

Arm-F extractions verified: vitb16 trusted (5767,768) and pool (51906,768),
vitl16 trusted (5767,1024) — fp32, no NaN, no zero rows. No trusted split
exists (supervisor gate intact).

WATCH ITEM for the probe gate: final DINO losses are very low (0.02-0.05),
which can indicate either healthy convergence or partial representation
collapse; the per-rung probe gate (frozen vs adapted, fixed parameters) is
the pre-registered arbiter and must not be pre-empted by informal checks.

Artifact locality: checkpoint directories (~33 GB total) and
arm_f_vitb16_unlabeled_pool.npz (159 MB, over GitHub's 100 MB hard limit)
are LOCAL-ONLY, now gitignored; reproducible from committed code + seeds +
cached weights. Trusted-embedding NPZs and all metadata are committed.

## 2026-07-19 - Phase 4 continual-learning implementation verification

- Implementation verification only; no scientific or official stream result
  was produced.
- Synthetic planted-structure tests proved HDC accumulator/re-binarization
  math, confirmed novel-class registration, a measurably constraining EWC
  update, deterministic reservoir retention/sampling, and fixed-spec runner
  determinism.
- End-to-end synthetic specs were explicitly marked `smoke=true`. The
  official path refused a missing spec and refused a smoke spec.
- The same all-four-learner simulator path passed at 384 and 1,024 embedding
  dimensions. The production 768-d ViT-B path uses the same inferred-dimension
  interface and was not launched.
- Full `ssl_hdc_pilot` suite: 46 passed plus 6 subtests in 8.09 seconds.
  New/modified Python files compiled; the v1 JSON schema parsed successfully.
- Ruff was not installed in `.venv-e1`; no package was installed or
  environment state changed for this implementation round.
- No accuracy-only result was produced. Summaries require macro-F1,
  macro-recall, per-class recall, forgetting, uptake, compute, and label-budget
  reporting.
- No Arm-D checkpoint or raw image was consumed by any continual learner.
- Post-review tests additionally proved that every preregistered stream-order
  seed runs and that multiple novel classes may share one activation step.
  Neural heads now optimize the same cosine scores used for prediction.
  `prequential_metrics` and post-stream `final_metrics` are separate and
  both reuse shared helpers from `common/thesis_eval.py`.

No official run was launched by the agent.

## 2026-07-19 — Official trusted splits built (user-launched, supervisor-approved A+B)

Option A (19-class headline): train/val/test = 3,396/1,132/1,132 = 5,660
(5,767 − 107 tail samples); 19 classes in every partition; zero partition
overlap; zero tail leakage; per-row provenance (option-a-19class, seed
20260716, supervisor_approved). Option B (25-class appendix): 57,670
assignments = 5,767 × 10 repeats × 5 folds, each image once per repeat,
same provenance. Both verified by the supervising agent. The labeled phase
is open: per-rung probe gates are now runnable.

## 2026-07-19 — All three probe gates PASSED (user-launched); adaptation benefit is monotone in scale

Fixed probe (LogisticRegression, seed 20260716, macro-F1 on trusted_val,
tolerance 0.005), per rung:

| Rung | Frozen | Best adapted | Delta | Selected |
|---|---|---|---|---|
| vits16 | 0.8767 | 0.8719 (epoch1) | -0.005 | checkpoint_epoch0 (frozen) |
| vitb16 | 0.8870 | 0.8908 (epoch3) | +0.004 | checkpoint_epoch3 (adapted) |
| vitl16 | 0.8944 | 0.9044 (epoch4) | +0.010 | checkpoint_epoch4 (adapted) |

Findings: (1) the collapse watch-item is RESOLVED — no rung shows feature
collapse (all adapted checkpoints 0.86-0.90); the low DINO losses were
healthy gentle convergence. (2) Adaptation benefit is monotone in encoder
scale (-0.5pt / +0.4pt / +1.0pt) — the encoder-scale study's headline
observation, reported regardless of downstream outcomes. (3) The headline
downstream study proceeds on the ViT-B pair: frozen epoch0 vs selected
adapted epoch3, per the ladder amendment and §6a rule 8 (gates have run,
so the adapted embedding comparison joins the grid). (4) Absolute linear-
probe quality is high (0.87-0.90 macro-F1, 19-class) — DINOv3 features are
strong on this task.

## 2026-07-19 - Section 6a generator and checkpoint-extraction verification

Implementation verification only; no official stream spec, full adapted
embedding artifact, or continual-study result was generated.

- Synthetic Option-A fixtures proved the exact four-class rare-disease rule,
  alphabetical tie handling, 25% per-known-class holdback, deterministic
  seed-specific manifests, descending-count 20/40/60/80% entry steps, fixed
  seeds and thresholds, summary logging, and unapproved-provenance refusal.
- The generated non-smoke payload passed the official runner parser. A runner
  test also proved that first novel registration requests and charges a label
  even when the low-confidence trigger is false. Post-review guards reject
  legacy hand-authored official protocols and force the first novel sample at
  the exact preregistered entry step.
- The shared EMA-teacher loader hard-errors on checkpoint/run model mismatch
  and recorded model-dimension mismatch. The probe gate now uses that same
  validator and strict state-loading path.
- Permitted extraction smoke: selected ViT-B `checkpoint_epoch3`, 50 images,
  768-d float32, 1.084 seconds of batched extraction after model loading.
  Metadata records `rung=vitb16`, `epoch=3`, `exported_encoder=ema_teacher`,
  and run-config SHA-256
  `7420a28f478c0484fa6bde28dc3de1c661e80e6396e2fc72346d240884c35174`.
- Permitted continual timing smoke: synthetic 768-d embeddings, one seed, one
  threshold, all four learners, 44 stream examples, 3.865 seconds wall time.
  Linear basis: about 86 seconds per 976-example seed-threshold bundle,
  13 minutes per embedding grid, and 26 minutes for both headline embeddings;
  the runbook budgets up to one hour because this extrapolation is approximate.
- Official validation recomputes the locked rare-class selection, novel entry
  order, stream length, and deterministic 25%-per-class manifests from the
  canonical provenance-stamped train/validation/test CSVs; copied inputs,
  stale reports, and hand-authored structural lookalikes are refused.
- Full suite: 58 passed plus 6 subtests in 10.5 seconds. Modified Python files
  compiled and `stream_spec.schema.json` parsed.

No official run was launched by the agent.

## 2026-07-20 - EWC baseline invalidation and implementation correction

- Verification found that every existing EWC cell was behaviorally equivalent
  to naive fine-tuning. EWCLearner._after_update() reset the anchor to the
  current parameters after every confirmed example; with the preregistered
  update_steps=1, the next penalty was therefore always zero.
- All existing frozen-B and adapted-B EWC histories and summaries are stale
  and must not be reported. HDC, naive, and replay results are unaffected.
- The correction keeps every-update online Fisher accumulation with the same
  ewc_decay, while the penalty uses a frozen Fisher and parameter anchor copied
  only when a genuinely novel class is first registered.
- A synthetic task-boundary regression using ewc_lambda=10.0 and
  update_steps=1 now proves a nonzero regularization loss after consolidation,
  a different parameter trajectory, and at least one different prediction
  from naive fine-tuning. The prior EWC constraint test remains active with an
  explicit novel-class boundary.
- The EWC-only runner path passed alongside the unchanged complete-grid
  default. Full ssl_hdc_pilot verification: 60 passed plus 6 subtests in
  9.71 seconds.
- RUN_COMMANDS.md now contains the user-launch-only 18-cell correction block:
  frozen-B and adapted-B, each with three seeds and three thresholds. It
  removes only stale EWC histories/summaries and verifies non-EWC file hashes
  are unchanged.

No official EWC rerun was launched by the agent; no corrected EWC scientific
result is recorded yet.

## 2026-07-20 — EWC decay amendment (ewc_decay 1.0 -> 0.9); EWC rerun invalidated & pending

Recorded BEFORE the corrected rerun, following the thesis's standing
amendment discipline (cf. G2 censored-IAT, G5 gate amendments): a parameter
set before it was ever exercised, shown miscalibrated the moment it engages.

After the anchor-consolidation bugfix (d46b3ea) made the EWC penalty active
for the first time, the pre-set `ewc_decay = 1.0` was exposed as degenerate:
with decay 1.0 the online Fisher is a pure cumulative SUM of grad^2 (no
decay), growing unbounded over the 976-step stream. Combined with the
boundary-frozen penalty and lambda=10 this (a) over-regularized on frozen-B
(macro-F1 0.25-0.35, WORSE than naive 0.37) and (b) diverged to non-finite
loss on adapted-B. The 1.0 value was never validated because the prior
anchor bug kept the penalty identically zero.

Amendment: `ewc_decay` 1.0 -> **0.9** (a standard online-EWC value, not
tuned to any outcome), yielding a bounded running-average Fisher. lambda
stays 10.0. This is a correctness fix (finite, bounded penalty), not an
attempt to make EWC win; whatever macro-F1 the corrected EWC produces is
the reported baseline, even if it still trails naive or HDC.

State at amendment time: frozen-B holds 9 completed-but-degenerate EWC
summaries (decay 1.0); adapted-B holds 5 orphaned EWC summaries from the
crashed rerun (its 27 non-EWC HDC/naive/replay outputs are hash-intact).
ALL EWC cells in BOTH arms are invalidated and must be regenerated under
decay 0.9. HDC/naive/replay results are untouched and remain final.

## 2026-07-20 - EWC decay-amendment implementation verification

Implementation verification only; the binding amendment immediately above
remains the scientific authority.

- build_stream_spec.py is the canonical source of the baseline block. Its
  ewc_decay is now 0.9 and ewc_lambda remains 10.0. Regenerating from the
  approved Option-A split produced the unchanged 976-step, three-seed,
  three-threshold protocol with stream-spec SHA-256
  62b545d31d592735be4404451bddd7f603576d127351f337e1e34c8fd04604cc.
- The observed partial state was left untouched: frozen-B has nine stale EWC
  summaries and nine histories; adapted-B has five orphaned summaries and
  five histories. Each arm retains 54 non-EWC files (27 HDC/naive/replay
  cells).
- A controlled repeated-observation test confirms the implemented recurrence
  observed = decay * running + grad^2: decay 1.0 grows linearly over 200
  observations, while decay 0.9 remains below observed/(1 - 0.9).
- Non-finite EWC updates now abort with an EWC-specific message containing
  ewc_decay and ewc_lambda. No penalty normalization or clamp was added, and
  d46b3ea consolidation remains unchanged.
- The replacement RUN_COMMANDS.md block parses as PowerShell and snapshots
  both arms before touching EWC files. It accepts any existing EWC file count,
  then requires exactly nine new summaries and nine histories per arm,
  validates their stream-spec hash, and byte-verifies every non-EWC file.
- Focused generator/EWC/runner verification: 23 passed. Full ssl_hdc_pilot
  suite: 62 passed plus 6 subtests in 10.01 seconds. Modified Python files
  compiled successfully.

No official EWC rerun was launched by the agent; corrected decay-0.9 EWC
scientific results remain pending.

## 2026-07-20 — EWC reruns complete under decay 0.9; FINAL 72-cell grid locked

Both arms back to 36/36 (9 EWC each), all finite, all on amended spec sha
62b545d3. Corrected EWC (decay 0.9, penalty active and bounded) now differs
from naive but only marginally: frozen 0.363-0.369 vs naive 0.324-0.370;
adapted 0.250-0.257 vs naive 0.253-0.257. Honest finding: a functioning
regularization-based continual method does not measurably beat naive
fine-tuning on this short class-incremental stream, and both trail the HDC
prototype learner by a wide margin. HDC/naive/replay outputs were
hash-verified byte-identical across the rerun. Grid is final; report next.

## 2026-07-20 — Report demoted to PROVISIONAL after external review

An external review flagged a decisive missing ablation: without online
nearest-class-mean and RanPAC-style random-projection prototype baselines run
through the identical stream, the study cannot attribute the win to HDC
specifically rather than to frozen-feature prototypes in general. Indicative
static check recorded (frozen ViT-B, trusted_train fit / trusted_val score):
plain NCM macro-F1 0.7274 (raw) / 0.7268 (L2-normed) vs linear probe 0.8870 —
suggestive but not decisive.

Also flagged and accepted: replay scoring below naive needs an audit;
pre-registered ECE / prequential / per-stage-recall were unreported (ECE
measured: HDC 0.780, naive 0.622, EWC 0.622, replay 0.554); no formal
average-forgetting metric; the "essentially immune to forgetting" phrasing
was an overclaim and has been removed; the ~1 ms update figure must be scoped
to the classifier update only (DINOv3 extraction ~14 ms/image is a shared
excluded cost); label-budget fairness needs an equal-budget curve; no
related-work positioning (OnlineHD, C-FSCIL, RanPAC, LifeHD) — the
contribution is applied/empirical, not a new algorithm.

Report §8 "Pending verification" enumerates all items. The gradient-baseline
comparison stands; the HDC-specific attribution does not, pending the
prototype baselines.


## 2026-08-02 - Prototype baselines implemented; replay configuration audited

This checkpoint addresses only provisional-report section 8 items 1 and 2.

### Item 1: decisive prototype baselines

- Online NCM maintains the exact streaming mean of each class's per-sample
  L2-normalized frozen embeddings, with no refit or backpropagation.
- RanPAC-style prototypes use a fixed standard-Gaussian projection seeded by
  the existing projection seed, the same 10,000-dimensional width as HDC,
  ReLU, online means, and cosine classification.
- RanPAC design source is cited in the learner docstring: McDonnell et al.,
  *RanPAC: Random Projections and Pre-trained Models for Continual Learning*,
  NeurIPS 2023. Gram decorrelation and ridge inversion are explicitly absent.
- Both learners register a class only on its first confirmed label and expose
  the same top-1 minus top-2 cosine-score margin used by HDC.
- The requested 36-cell official grid is in `RUN_COMMANDS.md` as a
  non-overwriting, hash-protected, user-launch-only block. NCM is explicitly
  the L2-normalized variant so the requested two-learner/36-cell grid is not
  silently expanded.

### Item 2: replay audit

Verified correct:

- reservoir replacement is seeded Algorithm R and therefore uniform over
  confirmed events;
- sampled embedding/label pairs stay aligned;
- the current confirmed sample is included in every update;
- the current sample enters the reservoir only after its update, preventing
  same-event duplication;
- dynamic class indices are resolved only after novel-class registration;
- replay uses the locked LR 0.1, update_steps 1, capacity 512, and sample 32.

Conclusion: no implementation defect was found.

- Ordinary mean cross-entropy over one current item plus 32 replay items gives
  the current event 1/33 of that update. This is a direct consequence of the
  registered current-plus-reservoir batch, not evidence of a code defect.
- Existing frozen-B threshold-0.10 artifacts show median final Macro-F1
  0.234358 replay versus 0.369629 naive, median final known-class accuracy
  0.294422 versus 0.345941, and 0/4 stabilized novel classes for replay versus
  1/4 for naive.
- Replay also requested fewer labels, so supervision and batch dilution are
  plausible explanations, but the present fixed-threshold data cannot
  attribute their relative effects.
- No replay implementation or hyperparameter was changed. The 0.234 result is
  retained as real under the registered configuration; changing loss weights
  after observing the outcome would be an unregistered, outcome-driven
  protocol change.

Verification:

- focused prototype/runner tests: 19 passed;
- full SSL-HDC suite: 73 passed plus 6 subtests in 11.53 seconds;
- Python source compilation and the appended PowerShell block parse passed;
- synthetic 768-D CUDA smoke, 44 examples, both prototype learners:
  0.514 seconds wall time.

No official run was launched by the agent.

## 2026-09-07 — Prototype grid present; report and audit reconciled

Inspected source HEAD `9cfd20a`: the 36 NCM/RanPAC comparison runs are now
committed. Verified 108 summary JSONs and 108 history CSVs across six learners,
two ViT-B encoder arms, three stream seeds, and three thresholds. Every history
has 976 rows; final-evaluation populations contain 3,396 examples.

Recomputed frozen-B medians at threshold 0.10:

| Learner | Final Macro-F1 | Prequential Macro-F1 | Stream labels charged |
| --- | ---: | ---: | ---: |
| HDC | 0.772032 | 0.745100 | 449 |
| RanPAC-style | 0.760232 | 0.756634 | 484 |
| NCM | 0.748892 | 0.750410 | 439 |
| Naive | 0.369629 | 0.314677 | 49 |
| EWC | 0.368889 | 0.317358 | 49 |
| Replay | 0.234358 | 0.210668 | 39 |

Final scores evaluate bootstrap+stream trusted-training examples after
learning, not trusted_test. All learners also receive 2,420 initial confirmed
examples excluded from the displayed stream charges. Prototype methods are
strong under this protocol; HDC-specific and equal-budget superiority are
not established. Replay is stronger on adapted-B final Macro-F1 than frozen-B.

Updated the scientific report with both-arm score, ECE, stage-recall,
retention, uptake, and recorded timing tables. ECE uses softmax-derived top
probability, not query margin. The rolling forgetting metric is not a fixed
held-out task metric. The host update timer is not explicitly CUDA-synchronized
and excludes encoder/prediction cost; no hardware speed claim is made.

Report remains PROVISIONAL. Prototype coverage and replay implementation audit
are resolved; equal-budget evaluation, generalization, and final inference/
positioning remain open. Old summary PNG is marked historical. No scientific
artifacts, protocol parameters, or command bodies were changed, and no
experiment or test-suite run was launched in this documentation session.

## 2026-09-09 — Planning checkpoint; no new scientific results

Created the [core evaluation protocol draft](../docs/TODO/HDC_CORE_EVALUATION_PROTOCOL_DRAFT_2026-09-09.md)
from the user's TODO v2 and current implementation/specification. It proposes
both matched-example and matched-allowance experiments, faithful state recovery,
fixed-population uptake/retention, statistics, and validation-before-test gates.

The draft is not approved or frozen. The previously approved 25-class repeated-CV
appendix and longitudinal-test evaluation require explicit scope reconciliation.
Existing results remain PROVISIONAL; quantum is optional, not the core claim.
The user's TODO was not edited or checked off. No source code, canonical stream
specification, historical scientific artifact, or runbook command was changed.
No test-suite run or held-out evaluation occurred in this checkpoint.

No official run was launched by the agent.

## 2026-09-10 — Review amendments; no new scientific results

Revised the [protocol draft](../docs/TODO/HDC_CORE_EVALUATION_PROTOCOL_DRAFT_2026-09-09.md)
to incorporate all five requested clarifications: the explicit predictive-endpoint
hierarchy and required secondary RQs, claim-to-evidence decision rules, upstream
train/validation probe supervision, research-machine latency/state/storage scope,
and an actionable A1–A4 amendment checklist. The user's AULC preference is recorded;
formal freeze decisions and feasibility evidence remain pending, not fabricated.

No binding specification, TODO, code, canonical stream, runbook, or historical
results were changed in this checkpoint. Results remain PROVISIONAL. No tests,
benchmarking, or held-out evaluation were run. These are documentation amendments
to a draft, not new scientific evidence or an approved preregistration amendment.

No official run was launched by the agent.


## 2026-09-10 — supervisor forgetting/throughput priorities (documentation checkpoint)

- User authorized updating the HDC documents after clarifying that the supervisor
  wants evidence of reduced forgetting and higher throughput. This supersedes the
  earlier same-day AULC-only primary hierarchy; historical log entries are retained.
- Revised the core protocol draft: primary fixed-population forgetting and cached-
  embedding continual-head throughput; mandatory HDC-versus-NCM/RanPAC-style
  comparisons plus all three gradient comparators; paired contrasts, joint claim
  rules, predictive/uptake criteria, longitudinal resampling, timing repetitions,
  and explicit benchmark boundaries and feasibility accounting.
- Updated README, HANDOFF, and the report's current claim assessment; appended a
  dated scope reconciliation to the binding specification. Existing report tables,
  stream spec, learner configurations, results, and the user-edited TODO text remain
  unchanged. Archived parent context is preserved under docs/source_context.
- Current evidence remains provisional. HDC's rolling-forgetting comparison with
  NCM/RanPAC-style depends on encoder arm; their recorded updates are shorter.
  No validated throughput or matched-supervision retention win is established.
- Validation: documentation diff/whitespace, local link, and preserved-content
  checks only. No training, predictive evaluation, benchmark, or test-suite run.
  The draft remains unapproved/unfrozen; A1–A4 decisions remain open.

No official run was launched by the agent.

## 2026-09-10 — Learner preservation verified; no new scientific results

Under the user's explicit next-steps 1–2 authorization, aligned the TODO and
implemented [learner checkpoints](LEARNER_CHECKPOINTS.md) for HDC, NCM,
RanPAC-style, naive, EWC, and replay. Saved states retain the information needed
for predictions and subsequent learner updates; the runner can opt in to final
NPZ exports with progress and provenance. No stream-controller resume is claimed.

Synthetic CPU checks established exact prediction and continuation equality,
including dynamic classes, tie history, Fisher/anchor persistence, replay sampling,
mode/gradient flags, and explicit RNG restoration. Runner histories/summaries are
byte-identical with saving enabled/disabled under a deterministic test clock.
File overwrite and corrupted-array guards were tested. Code review identified a
deferred CUDA-seeding side effect on neural reload; a checkpoint-only constructor
path fixes it and preserves ordinary initialization. The regression test passes.

Focused verification: 36 tests passed. Full `scripts/hdc.ps1 -Action test`:
89 passed, six subtests passed, 10.84 seconds (CPU). The approved test invocation
outside the restricted sandbox avoided its Windows temporary-directory ACL issue.
These are implementation tests, not an experiment or a latency measurement.

The 108 historical runs were not replayed or rescored. Existing numerical results,
canonical stream specification, split artifacts, and runbook commands are untouched.
No new benchmark, budget study, trusted-test evaluation, commit, or push occurred.
A1–A4 remain open; the scientific report remains PROVISIONAL. Further evaluation
implementation and official runs require their own authorization.

No official run was launched by the agent.

## 2026-09-11 — Reconstruction tooling verified; awaiting manual first cell

The user authorized continuing the TODO until input is needed and explicitly
retained manual ownership of every experiment. Added one-cell reconstruction,
default read-only planning, exact historical spec recovery, fresh-output guards,
learner checkpoints, ID-aligned final training predictions, provenance manifests,
and fail-closed fidelity reports. Canonical specification, registered learner
rules, historical results, and existing runbook blocks are preserved.

Review caught and resolved two data-access issues before any real reconstruction:
reject non-training paths before opening inputs, and avoid the generator helper
that rederives rarity using validation/test labels. Reconstruction validates the
locked specification and training-only holdback membership instead. Review also
closed final prediction-ledger and provenance gaps. Both review axes report no
remaining infrastructure blockers; summary-envelope duplication is a minor
maintenance observation, covered by fidelity tests.

Final `scripts/hdc.ps1 -Action test`: 114 passed, six subtests passed, 31.53 seconds
(CPU). Tests include synthetic reconstruction for all six learners, changed
event/count detection, timing exclusions, exact spec-hash recovery, no-overwrite
guards, virtual-filesystem held-out read refusal, and final ledger/metric equality.
An intermediate test assertion used the wrong summary key; it was corrected and
the full suite now passes. An earlier test-launch approval/usage-limit rejection
prevented execution; the later explicitly authorized invocation completed.

The new manual runbook block passed PowerShell syntax checks without execution;
`git diff --check` passed. The actual first NCM cell passed only read-only
metadata/hash planning: no learner construction, scoring, or result writes.
Current hashes cannot retroactively prove original embedding/split identity
because the historical summaries did not record those content hashes.

TODO 1.2/B infrastructure is complete. TODO 1.3/C and 1.4 remain pending:
the user must launch the guarded frozen-B/NCM/seed-0/threshold-0.10 first cell
and return fidelity_report.json plus its preflight transcript. No historical
reconstruction, budget experiment, validation/test evaluation or benchmark was
launched. A1-A4 remain unresolved. Changes are uncommitted; no push occurred.

No official run was launched by the agent.

## 2026-09-11 — User-run first reconstruction PASS (1 of 12 representative cells)

- Cell: frozen-B / NCM / stream seed 0 / confidence threshold 0.10.
- Directory: results/reconstruction/2026-09-10_ncm_frozen_s0_t010_attempt01.
- Report: PASS, 976 steps, zero history/summary mismatches, atol=1e-6/rtol=0,
  no input/code changes reported during the run. Durations remain excluded.
- Read-only follow-up: all six artifact hashes match the report; current
  source/embedding/train/spec and code hashes match the manifest. Final training
  prediction ledger contains 3,396 rows. No inference was rerun for this check.
- Final state SHA-256:
  6b63ea47d106fb188917a008f69860bae1c6587d842b4cbca588a5c56c9a9b8b.
- Transcript: results/reconstruction/preflight_d1e7114757364f0581372b88ab388019.txt.
  Recorded preflight showed 5% aggregate GPU activity, 876 MiB usage, browser,
  overlay and Game Bar processes before user confirmation. Later closure is not
  independently established; this is fidelity evidence, not clean timing data.
- Next manual stage: remaining 11 representative learner/arm cells, seed 0 and
  threshold 0.10; completed NCM cell excluded; stop at any failure. Commands
  appended to RUN_COMMANDS.md. All 108-cell fidelity and protocol A1-A4 remain open.

The user launched this experiment. No official run was launched by the agent.

## 2026-09-11 — Representative reconstruction 12/12 PASS; logging caveat

The user manually ran the remaining 11 representative cells under
results/reconstruction/2026-09-11_representative_attempt01. All report PASS,
976 stream events, zero history/summary mismatches, and no during-run input/code
changes at atol=1e-6/rtol=0. Read-only verification checked all 66 report-listed
artifacts and current code/input hashes across the full 12-cell representative
set. There are 12 distinct learner/arm cells, seed 0, threshold 0.10.

The saved preflight_and_runs.txt exists (9,651 bytes when inspected), but its
recorded end is 03:29:30 before the confirmation prompt and learner outputs.
The user's pasted console output supplies completion/error context. Cleanup
Stop-Transcript failed because the host was no longer transcribing; an isolated
call with no active transcript reproduced that exact error. This reproduces the
cleanup symptom, not the unexplained early termination of the user's transcript.
No claim of complete transcript coverage or clean timing is made. Artifacts are
preserved; no completed cell should be rerun merely to repair logging.

Appended user-launch commands for the remaining 96 historical cells, excluding
the completed 12. They use direct preflight/per-cell log files rather than
transcript state. Syntax and read-only prior-artifact/count/input/disk guards
passed: 96 pending conditions and approximately 33 GiB free. The 12-cell output
tree measured 133,563,902 bytes at inspection; this is storage, not runtime memory.
The new 4 GiB free-space floor is an operational guard, not a measured peak.
No learner code was changed. All 108-cell acceptance, A1-A4, budget experiments,
held-out evaluation and benchmark evidence remain pending.

The user launched all experiments. No official run was launched by the agent.

## 2026-09-11 — Full historical reconstruction 108/108 PASS

The user completed the remaining 96 cells. Independent read-only audit verified
96 distinct PASS reports and all 576 report-listed artifacts in that batch,
then rechecked all three attempt trees: 108 unique cells exactly covering the
original 108 summaries, 648 artifact hashes, and all current source/input/code
hashes. Each report has 976 steps, zero history/summary mismatches, no during-run
input/code changes and atol=1e-6/rtol=0. Each final training prediction ledger
contains 3,396 rows with unique IDs. No inference or experiment was rerun.

The 96-cell directory has 96 direct run logs plus a preflight log recording user
confirmation. Its size at audit was 1,068,464,871 bytes. Preflight showed 3% GPU
activity and 825 MiB used; these are not benchmark-quality timing measurements.
The previous 11-cell transcript coverage caveat remains recorded, not erased.

Full evidence and preserved locations are in
results/RECONSTRUCTION_AUDIT_2026-09-11.md. TODO 1.3, 1.4, C and R2 are complete
for training-side observed-output fidelity and saved historical final states.
Original input identity cannot be proved retrospectively where hashes were not
recorded. The scientific report remains PROVISIONAL; no equal-budget, held-out,
fixed-population retention or validated throughput conclusion follows.

Next input gate: explicit agreement on Experiment A's proposed additional-label
budgets and label-blind nested shared-event schedule before implementing that
new supervision policy. A1-A4 and test access remain separately gated. Historical
reconstruction blocks are complete and must not be restarted. No code or result
artifact was altered by this documentation checkpoint; changes remain uncommitted.

The user launched all experiments. No official run was launched by the agent.

## 2026-09-11 — Experiment-A shared-label development implementation

Subsequent user approval resolved the bounded Experiment-A acquisition settings:
budgets 0/25/50/100/200/400, identical label-blind nested events and timing across
six learners and both arms, registration only upon purchased scheduled labels.
Added experiment_a.py, an opt-in shared-label runner branch, 41 synthetic tests,
and EXPERIMENT_A.md. Historical acquisition/update formulas remain unchanged.
The new branch changes the runner code hash; original reconstruction manifests
and artifacts retain their original provenance and must not be rewritten.

Real frozen-B/NCM/seed-0/B=25 read-only preparation passed: 2,420 bootstrap IDs,
976 stream IDs, 25 selected events, schedule hash
94efc01097a4efda9e398880df988e4571ad0df62aa88ac612b607021ab75e1f.
No learner was constructed or executed by that plan. Standards review identified
an input hashing window; hashing before parsing plus rechecking after preparation
closed it, with a filesystem-boundary regression. Spec review found no blockers
within this bounded scope. Full wrapper suite: 155 tests and six subtests passed
in 54.80 seconds. The separate attempted red-regression invocation was blocked
by tool approval/usage limits; no pre-fix failing test execution is claimed.

TODO 2A.1-2A.2 now complete for implementation/synthetic checks only. Added a
guarded manual first-cell command with direct logs. Actual Experiment-A execution,
fixed validation evaluation (2A.3), longitudinal/primary metrics, cost measures,
full-grid feasibility and A1-A4 remain pending. No held-out labels were opened.
Changes remain uncommitted; unrelated edits are preserved.

No official run was launched by the agent.

## 2026-09-11 — A25 integrity verified; fixed-validation evaluator implemented

User-run frozen-B/NCM/seed-0/B=25 receipt PASS independently checked on disk:
all seven artifact hashes match; history has 976 rows and 25 purchases matching
the scheduled count; summary records 2,420 bootstrap plus 25 stream labels;
final training ledger contains 3,396 unique IDs. No training was rerun.

Added evaluate_validation.py, synthetic tests and VALIDATION_EVALUATION.md.
Default read-only planning passed against that saved source: 1,132 validation
examples, 19 fixed classes, 768-d same-arm embeddings, no train/val overlap.
No learner construction or real-data prediction occurred. Real execution is
manual, validation-only, and writes a fresh directory. It preserves full-class
common metric semantics, checks source/input/code hashes, and rejects learner/RNG
mutation. Historical learner/runner/metric code and completed outputs are unchanged.

Initial focused suite: 22 passed. Added the review-suggested mid-read replacement
regression and tightened code-hash pinning/receipt consistency. Final full wrapper:
178 tests and six subtests passed in 42.10 seconds. Initial sandbox temporary-
directory permission errors were resolved using approved synthetic test execution
outside the sandbox. Standards and Spec reviews report no blocking findings.

Next manual step is saved-state A25 validation, not A25 retraining or a full grid.
TODO 2A.3 remains open for actual validation acceptance across budgets. No test
labels, longitudinal outcomes, benchmark or new predictive result were accessed.
Changes remain uncommitted; unrelated existing edits are preserved.

No official run was launched by the agent.

## 2026-09-11 — Validation accepted as development evidence; longitudinal arithmetic

User completed saved-state frozen-B/NCM/seed-0/B=25 validation. All three saved
artifact hashes match. The 1,132 unique prediction rows reconstruct common
classification metrics and 15-bin ECE exactly: Macro-F1 0.6600447531111898,
macro recall 0.7225543249560078, accuracy 0.8356890459363958,
ECE 0.7468861113055593. Classes 3_a9_3 and 6_a11_1 remain unregistered.
The receipt reports unchanged state/RNG. This is one development cell, not an
HDC comparison or untouched-test result. Reading/reducing saved predictions
did not rerun a learner. Both A25 launch blocks are now completed provenance.

Added pure longitudinal_metrics.py and 44 synthetic tests: fixed-population
retention on explicit grids, post-purchase novel uptake with missing/undefined
cases and later drops, complete-grid normalized AULC, paired descriptive seed
contrasts. There are no runtime loaders, model calls or approved defaults in
this module. Common metrics and existing training/checkpoint code are unchanged.
Standards/Spec reviews found no blockers. Full wrapper verification passed
222 tests and six subtests in 46.26 seconds; git diff --check passed.

Updated TODO/handoff to record partial arithmetic completion, not completed
longitudinal experiments. Next scientific decision is explicit approval of the
draft Experiment-B acquisition and retention/uptake snapshot/target settings
for development only, documented in LONGITUDINAL_METRICS.md. Full grids, test
release, benchmark placement/repetitions and A1-A4 remain gated. Existing
artifacts and unrelated edits remain preserved; changes are uncommitted.

Workflow: agent reads reports/checks artifacts/analyzes existing outputs directly;
user receives commands for new training/evaluation/benchmark launches, not file
display. No official run was launched by the agent.

## 2026-09-12 - Approved A/B acquisition and longitudinal state capture

Implemented budget_experiment.py and an opt-in stream observer following the
user's explicit bounded approval: B budgets 0/25/50/100/200/400, thresholds
.05/.10/.20, uncertainty plus privileged novel requests, one charge per purchase,
underspending and exhaustion guards. Both A/B capture the approved ten retention
times and available per-class uptake milestones, combining overlapping states.
Event/accounting ledgers, snapshot indexes and artifact/code/input hash checks
make acceptance auditable. Completed historical/A25 artifacts were preserved.

Full wrapper passed 281 tests plus six subtests in 89.66s. Synthetic six-learner
checks compare exact state/RNG and history against an unserialized reference;
post-capture snapshot corruption is rejected. Initial test fixture/RNG setup
issues were corrected without changing learner formulas or scientific guards.
See BUDGET_SNAPSHOTS.md for review and scope details.

Read-only real-input planning passed for frozen-B/NCM/seed-0, Experiment B
allowance 400, threshold .10. No real B or snapshot learner run was launched by
the agent. RUN_COMMANDS.md contains one guarded manual development launch.
New-format longitudinal validation integration, real curves, benchmarks and
full-grid/test release remain pending; A1-A4 remain unresolved. Runner code hash
changes are versioned provenance, not a reason to rewrite or rerun old outputs.
TODO/handoff now distinguish completed acquisition/capture from pending evidence.
Changes remain uncommitted; unrelated work is preserved.

## 2026-09-12 - B400 capture accepted; snapshot validation integration

User completed frozen-B/NCM/seed-0/Experiment-B allowance400/threshold.10 capture.
Independent read-only audit verified 27 artifact hashes and current recorded
input/code hashes, 976 history/event rows, 3,396 unique training prediction IDs,
450 potential requests, 400 purchases, 50 denials, zero unused allowance and
all four novel registrations. Twenty snapshots cover every approved retention
time plus available uptake states. Total supplied learner labels: 2,820.
No predictive comparison, forgetting or benchmark claim follows from capture.

The pasted pipeline separated Tee-Object from -FilePath. Original preflight log:
preflight_run_6c7795364b0b4fe3b4471dc604ce66ef.log; training output was redirected
at the prompt to b400_recovery_01.log. The later standalone -FilePath error is
shell logging, not learner failure. Both logs and all source artifacts preserved.

Added evaluate_snapshots.py with strict source/code/policy and purchase-ledger
validation, fixed validation-only embedding exposure, checkpoint-context and
state/RNG checks, per-snapshot common metrics and approved forgetting/uptake
arithmetic. Added scripts/validate-b400-snapshots.ps1: one-command manual launch,
preflight, explicit splatted log path, cancellation and non-overwrite guards.
Existing learner/runner/metric modules and original artifact hashes unchanged.

Full wrapper: **324 tests plus six subtests passed in 98.46 seconds**. New tests
cover all six learners and five fully mocked launcher paths. Initial sandbox
temp-directory denial was handled by approved synthetic test execution outside
the sandbox. Standards review: zero findings. Spec review: partition-filter
finding fixed and regression-tested; no other blockers. Real-input PlanOnly
passed: 20 states x 1,132 validation examples, no learner loaded or scored.

No real training, validation or benchmark was launched by the agent. Next manual
action is existing-state longitudinal validation; B400 training is complete.
Comparative grids, real curves, benchmarks, paired uncertainty, A1-A4 and final
test release remain open. Updated TODO/handoff/runbook/guides accordingly.
Changes remain uncommitted to preserve the preexisting mixed worktree.

## 2026-09-13 - B400 longitudinal validation independently accepted

The user completed the 20-state validation launch. Read-only audit matched all
22 output hashes and 48 recorded pins, checked 22,640 predictions against the
same 1,132 validation IDs/labels, and reconstructed classification/stage/ECE/
confusion matrices plus forgetting/uptake exactly. Before/after learner/RNG
fingerprints match at every state. No learner was loaded or scored by the audit.

Final Macro-F1 .7260553027746831, macro recall .8108204852601534, accuracy
.8162544169611308, ECE .7359850954580327. Mean old-class forgetting
.04387119555312556 (4.3871 percentage points); 1,074 old-class images. Novel
final recalls: 3_a9_2 .409091, 5_a7_2 1.0, 3_a9_3 .75, 6_a11_1 1.0. First/third
never reached absolute .80 at observed uptake milestones. No comparator claim.
Detailed supports, milestone counts and storage observations are in
results/B400_VALIDATION_AUDIT_2026-09-12.md. Stored file bytes are not runtime memory.

Marked B400 training/validation complete, updated TODO 3.1-3.5 and bounded dry
runs 4.2-4.4, maintained report/handoff/runbook. Prepared an explicitly unapproved
benchmark development plan; device/placement, warm-up/repetitions and timing
boundaries need user input. Full grids, uncertainty, feasibility and A1-A4/test
release remain open. No new experiment, source rewrite or code change occurred.
Prior full suite remains 324 tests plus six subtests; this checkpoint changes
documentation only. Mixed worktree remains uncommitted and preserved.

## 2026-09-13 - Approved benchmark development implemented

The user explicitly approved the bounded benchmark policy. Added separate
benchmark_core.py, benchmark_environment.py and benchmark_study.py, synthetic
tests and scripts/benchmark-a25.ps1. Existing learner/runner/checkpoint/common
metric modules and completed scientific artifacts were not edited by this
benchmark implementation. This is infrastructure, not measured performance.

Implemented common CUDA placement with host-resident embeddings/transfers,
restored bootstrap state/RNG, untimed original-runner references, one disposable
warm-up and five rotating paired timing blocks per learner. Full-stream timing,
synchronized update/bootstrap latency, process/CUDA peak sampling and retention-
grid payload/storage exports are separate passes. Every pass checks event and
final training-prediction fidelity. Manifests explicitly identify experiment,
budget and stream seed alongside input/code/policy/hardware provenance.

Prediction/resumable/replay/global-RNG payload, logical file bytes, actual file
allocation and hypothetical content deduplication are distinguished. Process
peaks are sampled, include shared runtime/embeddings and do not establish
per-head native object overhead. Prediction-only NPZs are cost artifacts, not
deployment interfaces. Workload inventories reject recognized competing apps
and other GPU workers; empty/malformed Windows inventories fail closed. No
process is killed and failed attempts are preserved without automatic retries.

Standards review: two nonblocking observations addressed; follow-up zero
unresolved findings. Spec review: three findings addressed; follow-up no static
blockers. Subsequent local checks added explicit condition provenance and tests
for inventory-hook exclusion and unreadable process inventories. The guarded
launcher has six fully mocked control-flow tests, including logging and refusal.

Read-only real-input PlanOnly passed: frozen-B/seed-0/A25, 2,420 bootstrap
labels, 976 stream events, six learners and five blocks. No learner constructed
or scored. Next user action is the one-command benchmark pilot in BENCHMARKS.md
and RUN_COMMANDS.md. It performs new updates/timing in a fresh attempt, not a
rerun or replacement of completed A25/B400 predictive evidence.

TODO 3.6/3.7/F1 now mark implementation complete; 4.4a real acceptance and
feasibility remain open. A1-A4, comparative grids, uncertainty, final templates,
protocol freeze and untouched-test release are not completed or newly approved.
No real training, evaluation, benchmark or official run was launched by the
agent. Changes remain uncommitted in the preserved mixed worktree.

Final verification after guard/provenance fixes: **371 tests plus six subtests
passed in 119.72 seconds** through scripts/hdc.ps1 -Action test, with CUDA
disabled. Includes 47 benchmark tests. Focused process-inventory checks: seven
passed. git diff --check passed. No real benchmark attempt directory exists.

## 2026-09-13 - Attempt01 preflight failure; approved desktop guard recovery

The user launched A25 benchmark attempt01 at 17:01:16 +06:00. Initial guard
rejected Windows desktop processes listed by the driver compute query, plus
Overwolf. It stopped before learner setup or any training/timing. The attempt
contains only failure.json. Windows PowerShell Stop error handling interrupted
native stderr at the first traceback line, so that text was absent from the log.

Preserved original SHA-256:
- failure.json: 6e072588ef69ddece1437a65e6a2d9c4a3b9eb5b85e306ea961bd4ad60ef4155
- preflight_run_92f3f17a44ac48b7845f7a7048db0643.log:
  a7851311d61df5676b6cadacf2603ead3d62cf77e502f7bb6b4f4325cdbd746b

The user approved a documented Windows desktop/Terminal/VS Code baseline.
The guard now matches only the pinned exact executable paths, records accepted
PID/path entries, blocks Overwolf independently of GPU presence, and keeps
unknown GPU processes blocked. It does not blanket-allow C+G, WebView or optional
apps, attest executable signatures, or claim accepted desktop apps have no cost.
Malformed GPU rows fail before allowance checks. No apps are terminated.

The launcher drains/stringifies full native stderr and checks the actual exit
code with its prior error preference restored. Added a read-only workload CLI
before the RUN prompt/attempt creation. Default and explicit next attempt is02;
the directory is never reused or automatically incremented. Learner, timing,
checkpoint and metric semantics are unchanged. No attempt02 has been launched.

Regression reproduction: eight intended new tests failed on the old guard/helper.
Focused guard/logging checks then passed; actual harmless native processes test
both stderr-with-exit0 and stderr-with-exit9. All 64 benchmark tests passed before
the final six malformed-GPU-row additions; final targeted set passed 17 tests.
Standards review: zero findings. Spec review: one malformed-row fail-closed issue
fixed and tested; follow-up zero remaining findings. Reviews cover only this
bounded patch against saved turn-start copies, not the unrelated dirty worktree.

Read-only absolute-path -Attempt 2 -PlanOnly passed from outside the repo. A live
workload check correctly accepted the desktop baseline but still blocked WebView,
Sticky Notes and Settings; operator must recheck/close these before launching.
Sandbox tasklist access was denied; the read-only check used approved escalation.
No learner was constructed/scored and no real benchmark was launched by the agent.
TODO/handoff/policy/runbook updated. A1-A4, feasibility and scientific claims stay
open. Mixed changes are preserved and uncommitted; no cleanup, commit or push.

Recovery verification: full suite **394 tests plus six subtests passed in
122.90 seconds**. Subsequent read-only parent inspection identified the remaining
WebView as Windows Search-owned (11540 -> 11596 -> SearchHost 4464). Added a narrow
pinned-path/verified-ancestry desktop exception, not generic WebView allowance.
Parent creation times, cycles, missing records and path mismatches are checked;
accepted ownership chains are recorded. **31 focused tests passed in 6.98 seconds**
after this refinement. User requested immediate finish; full suite was not rerun
after the final ancestry-only refinement. No benchmark launched or apps killed.

## 2026-09-13 - User-run A25 benchmark attempt02 accepted

Read-only audit matched 251 artifact hashes and 21 input/code/policy pins;
205 recorded workload checks passed. Compared all 60 timed-pass ledgers against
their references (58,560 events, 203,760 final training predictions), rebuilt
rates/paired ratios and checked latency/payload/storage accounting. No learner
loaded, experiment rerun, source artifact changed or final test inspected.

Median events/s: HDC532.3372, NCM1628.2175, RanPAC1183.0046, naive1313.1340,
EWC1473.5588, replay1386.0947. HDC was slower in every paired block against all
five controls. Its lower update latency versus neural heads does not imply a
full-stream throughput advantage. One budget/arm/seed, five technical repeats;
results remain development-only. Whole attempt: 1,408,083,356 logical bytes.

Added results/A25_BENCHMARK_AUDIT_2026-09-13.md; updated TODO/handoff/guide/runbook.
Attempt02 is completed provenance; do not rerun. First-pilot acceptance done,
but full-grid feasibility, validation templates/uncertainty, A1-A4 and test
release remain open. Saved-input/policy/code pins were checked before these
documentation updates; pinned code/policy files were not edited by the audit.

## 2026-09-13 - Preliminary resource feasibility worksheet

Read-only artifact/log/disk accounting: accepted six-learner pilot1.3114GiB,
about11.3minutes confirmation-to-final-file time; G: available33,005,576,192bytes
(30.7388GiB). HDC/RanPAC NPZs dominate preserved pilot storage. Added
FEASIBILITY_WORKSHEET_2026-09-13.md with explicit conditional A/B scaling:
144 six-learner groups ~188.84GiB and ~27.1hours at pilot size/time, not a
validated all-grid forecast or upper bound. Capture/evaluation/analysis/backup
costs remain additional. Completed108 reconstructions must not be rerun.
Updated TODO/handoff; A3 awaits storage location/budget and runtime ceiling.
No scientific code/policy changes, runs, deletions or relocations performed.

## 2026-09-14 - Authorized shared-projection storage prototype passed

User reported ~70GB free; measured71.0974GiB on G:. Scratch-only storage test
factored46 HDC/RanPAC NPZs into one content-addressed projection and compressed
deltas/references. All1,066 arrays/metadata entries reconstructed byte-for-byte;
92 corrupted-projection/missing-inventory cases rejected. All252 source files
unchanged by SHA-256 and inventory. No learner constructed, trained or scored.

Equivalent full-pilot layout:1,408,083,356 ->127,790,994bytes (90.9245% reduction).
Compact test artifacts44,559,948bytes; other artifacts83,231,046bytes counted
unchanged. Originals were NOT replaced; scratch copies are additional storage.
Conditional144-group benchmark-only scaling17.14GiB, not a validated all-study
estimate. Production loader/writer remains unchanged; integration and synthetic
learner round trips are next. Added results/PROJECTION_STORAGE_TEST_2026-09-14.md,
updated feasibility/TODO/handoff. Throwaway program removed, scratch evidence
retained. No full-suite run needed: no production scientific code changed.

## 2026-09-14 - Approved production shared-projection save/load integration

User approved integration after the lossless storage prototype. New budget
longitudinal and six-head benchmark attempts share actual projection arrays in
run-local content-addressed blobs, including prediction-only exports. Logical
learner schema1 and default standalone callers remain compatible. The independent
storage envelope is version1; compact loads require an explicit run root. No
completed checkpoint was rewritten, converted, deleted or automatically rerun.

Atomic exclusive publication verifies existing blobs before reuse. Tests found
and fixed a concurrent Windows path-spelling mismatch (ordinary versus extended
namespace); containment is still checked after resolving links/junctions. Saved
array shape/dtype/content and learner/RNG semantics remain unchanged. Benchmark
reports now separate file bytes, dependency-closure bytes and live tensor memory;
receipts hash each shared blob once and snapshot validation checks exact closure.

Focused post-fix verification: 62 synthetic storage/snapshot-validation tests
passed in41.24s. Final full CPU suite:434 tests plus6 subtests passed in133.93s;
git diff --check clean. Attempt03 does not exist;
its read-only launcher PlanOnly passed (2420 bootstrap labels,976 stream events,
A25,all six learners,5 paired blocks). No learner construction/training/scoring
or timing was performed by that plan. Real production-layout size/acceptance is
still pending a manual launch, not implied by the prior90.92% prototype saving.
Review details/limitations: results/SHARED_PROJECTION_INTEGRATION_2026-09-14.md.

## 2026-09-14 - Settings recurrence and approved shared-desktop amendment

User-run attempt03 failed after the HDC latency pass in block4/5. Saved failure
records142 successful boundary checks, last before/3/hdc/latency, and Settings
PID10824 at failure. User confirmed it reappeared automatically. No final integrity
PASS exists. This is a conservative guard stop, not proof of measured interference
or a checkpoint/learner defect. Original failure/log preserved:

- failure.json SHA256:d5459151181656e709c1b7ea75c68748b7665099298e075958a3bbdf1e6c1938
- preflight_run_655e0b085a5d4566b39606f9653180dd.log SHA256:c181fb7ac09dbc40a26f9335e7d68492358991cad0433d49d23bc8934afd12ef

The user approved accepting/logging ordinary identified Windows activity. Added
the exact observed Settings path to the existing desktop baseline; recurrence
and changed PID no longer abort. Inventory reports identify the new policy and
interference_measured=false. Other competitor checks and unknown GPU exclusions
remain; no directory-wide allowance, process killing, timing or learner changes.
All five paired blocks/variability remain required under shared-desktop limits.
Attempt03 is not resumed or retrospectively accepted; attempt04 is a fresh manual
run with new policy/code provenance. No official run was launched by the agent.

Synthetic regression:4 failures under old policy, then12/12 passed after amendment
in2.46s. Final full CPU suite:446 tests plus6 subtests passed in128.75s. Current
read-only workload preflight PASS (no blocked/other GPU entries) and attempt04
PlanOnly PASS; no actual benchmark launched. git diff --check clean.
Standards/spec independent reviews both
completed with no findings; see results/SHARED_DESKTOP_AMENDMENT_2026-09-14.md.
Updated policy/core draft, launcher messages, benchmark guide, TODO/handoff and
full guarded attempt04 runbook block. No A1-A4/test-release approval implied.

## 2026-09-14 - Compact A25 benchmark attempt04 independently accepted

User-run PASS verified against252 artifact hashes,22 input/code/policy pins,
205 boundary inventories,60 timed passes,58,560 event rows and203,760 training
prediction rows. Bootstrap ledgers and summary ratios/medians reconciled.
All46 compact archives share one verified blob;78 resumable states and1,128
array hashes/shape/dtype records verified. Per-file/dependency-inclusive storage
counts and exact receipt closure reconcile. No source file mutated by the audit.

Directory127,790,310bytes (121.87MiB),90.9245% smaller than attempt02; adjacent
log86,496bytes excluded. Original attempts remain intact. About10.2minutes
confirmation-to-final-file time; benchmark-only144-group scenario17.138GiB /
24.5hours remains conditional, not high-budget/B/all-study feasibility.

Median stream events/s:HDC553.06,NCM1507.45,RanPAC1624.39,naive1557.56,
EWC1384.37,replay1367.65. HDC slower than all five controls in every paired block.
No throughput advantage or held-out retention/predictive claim. Shared-desktop
limits apply; Settings observed at zero recorded boundaries, so its recurrence
was not exercised by this real attempt. Synthetic recurrence tests remain separate.

Added results/A25_COMPACT_BENCHMARK_AUDIT_2026-09-14.md; updated handoff/TODO,
checkpoint/benchmark guides, runbook and feasibility worksheet. Attempt04 complete;
do not rerun. Broader feasibility/resource ceilings, templates and A1-A4 remain.
Saved-file analysis only: no official run, learner construction or scoring by agent.

## 2026-09-14 - Runtime ceiling resolved; B400 feasibility launch prepared

User said "as long as is needed" for total machine time. Recorded no user-imposed
runtime cap, without implying unlimited storage, new spend, full-grid/test release
or agent-owned execution. Prepared scripts/benchmark-b400.ps1 for ONE frozen-B/
seed0/Experiment-B/allowance400/threshold0.10 six-head/five-block condition under
existing policy. Actual labels consumed remain method-specific. No learner or
timing engine changes, no rerun/replacement of completed NCM B400/A25 artifacts.

Read-only input plan passed:2420bootstrap labels,976stream events,400allowance,
threshold0.10,all six heads. Preparation check:G: free73,537,986,560bytes, fresh
B400 benchmark attempt01 absent. Existing8GiB launch guard/logging/shared-desktop
policy and RUN BENCH B400 confirmation are preserved. No automatic retry/resume.

Verification:14 targeted launcher tests passed in9.47s. Initial full-suite request
was blocked by the approval service usage limit (not test failures); unchanged
CPU-only retry was subsequently accepted:453 tests plus6 subtests passed in208.14s.
git diff --check passed. The user committed
current work during the interruption (HEAD52c1ee4); the agent did not commit it.
Independent reviews found0 standards violations/1 nonblocking duplication note
and0 Spec findings; see results/B400_BENCHMARK_LAUNCHER_REVIEW_2026-09-14.md. Added
full paste block/guide, updated TODO/feasibility/handoff. No official run was
launched by the agent. Broader storage/backup estimates and A1-A4 remain open.

## 2026-09-14 - B400 blocked preflight and background-warning amendment

User reported a read-only preflight block for GitHub Desktop, Spotify Game Bar
WebView and Edge Game Assist. Log preserved unchanged at
results/benchmark_development/preflight_run_d2ad9f4836c24975b2793339284290fa.log
(30,986 bytes); B400 attempt01 confirmed absent. No benchmark result is implied.

At user request, unknown GPU PID/path entries now produce warnings rather than
abort. Known browser/game/recording/overlay/Python exclusions still stop through
either primary inventory. Primary inventory failures remain fatal; optional
Search ownership failures warn without claiming verified ownership. No process
was killed. Preflight PASS_WITH_WARNINGS exits0. All in-run warning boundaries
survive into summary and integrity receipt; integrity PASS does not establish
clean timing. Keep all five paired blocks; no isolated-machine or HDC advantage claim.

The exact-three-app synthetic regression failed before the fix and passed after;
63 targeted tests passed in4.25s. Read-only B400 input plan passed. Independent
Standards/Spec reviews found0 defects; one nonblocking integration-coverage
suggestion is documented. Full verification details, including any interrupted
test execution, are in results/BACKGROUND_WARNING_REVIEW_2026-09-14.md.
No scientific run was launched or completed artifact changed by the agent.
Final verification:475 CPU tests and6 subtests passed in144.61s. Live read-only
preflight PASS_WITH_WARNINGS/exit0, no known blocked workloads; PhoneExperienceHost
retained as unclassified. Original three apps absent live, tested synthetically.
Initial sandbox tasklist denial resolved by authorized read-only retry; interrupted
full-suite session was not counted as PASS. B400 remains a manual next step.

## 2026-09-14 - User-run B400 six-head benchmark independently accepted

Completed benchmark_development/2026-09-14_b_frozen_s0_b400_t010_attempt01;
do not rerun. Saved-file audit verified251 artifact hashes,22 current input/code/
policy pins,60 timed ledgers,58,560 event rows and203,760 training predictions.
All205 warning boundaries retained; summary/receipt qualification matches.
All78 resumable states/1,182 array hashes and46 compact archives/one blob verified.
See results/B400_BENCHMARK_AUDIT_2026-09-14.md for hashes, tables and audit scope.

Total129,973,339bytes (123.95MiB), plus229,122-byte launch log; approximately
10.44minutes confirmation-to-receipt, including guards/untimed work. HDC median
428.58events/s versus NCM1206.89,RanPAC1370.35,naive1441.91,EWC1287.47,replay1430.51.
HDC slower in all paired blocks; purchases400/400/400/53/53/37 respectively.
Equal allowance is not identical supervision. No throughput advantage or new
held-out/retention claim. Warning activity is not measured interference.

Updated TODO,handoff,README,runbook,benchmark guide,scientific report and feasibility.
The high-budget pilot is complete; full-study output/backup planning and A1-A4
remain open. No official run was launched by the agent; only saved files were
read/reduced, with no learner construction, updates or scoring. No outputs changed.

## 2026-09-15 - Comparative forgetting-grid preparation

Prepared, but did not launch, a 35-new-cell Experiment-B/threshold0.10/budget400
fixed-validation capture grid across both ViT-B arms, all six learners and three
stream seeds, reusing and protecting the completed frozen/NCM/seed-0 cell. Added
the per-cell guarded launcher, A3 measured feasibility and manifest, plus a
post-hoc support-aware uptake supplement. Original snapshot uptake remains visible
and unchanged. No test partition, learner, capture, scoring or benchmark was run
by the agent. A3 fits at the current free-space reading subject to the 8 GiB guard.

## 2026-09-15 - Forgetting-grid provenance follow-up

Amended the grid to 36 fresh cells: frozen/NCM/seed-0 is now recaptured into a
new path, while its 2026-09-12 capture and validation remain archived unchanged.
This removes the cross-code provenance confound. Frozen/HDC/seed-0 is required
as measured cell 1; its actual projection-storage layout must revise A3 before
cell 2. No capture, scoring, benchmark or test access was launched by the agent.

## 2026-09-15 - Unattended batch preparation

Added a separate 36-cell batch wrapper with one initial confirmation, receipt-based
completion checks, per-cell background/free-space guards, durable JSONL progress and
append-only logs. Power cuts preserve partial attempts and escalate to fresh attempts
02/03; a fourth stops. No official run was launched by the agent.

## 2026-09-15 - Comparative forgetting grid COMPLETE; encoder-dependent result

User-run 36-cell Experiment-B/threshold0.10/budget400 grid completed across both
ViT-B arms, all six learners and seeds 0/1/2. All 72 capture and validation
integrity receipts are PASS with hash-pinned artifacts and empty
code_changed_during_run / inputs_changed_during_run. No attempt escalation
occurred; every cell is attempt01. Operational span 26.5 minutes, median about
0.6 minutes per cell. The earlier 16.1-hour A3 projection extrapolated the
2026-09-12 NCM cell's log timestamps, which included manual confirmation delay
and the documented shell-pipeline break; it overestimated compute by about 36x.
Shared content-addressed projection blobs are confirmed present in HDC captures.

Supervision is matched at 400 stream labels for HDC, NCM and RanPAC-style in every
cell. Naive/EWC consumed 17-53 and replay 37-79; those heads are not label-matched
and must be reported separately.

Final Macro-F1 on trusted_val, medians over three seeds. Frozen-B: HDC 0.7220,
NCM 0.7313, RanPAC-style 0.7406 - HDC best in 0/3 seeds. Adapted-B: HDC 0.6001,
NCM 0.5275, RanPAC-style 0.5243 - HDC best in 3/3 seeds with non-overlapping
per-seed ranges (HDC minimum 0.5943 above NCM maximum 0.5346 and RanPAC-style
maximum 0.5451). This reproduces the historical training-pool gap of about +10
points as +7.3/+7.6 on a population the learners never trained on, at equal
supervision. Direction reverses by encoder arm, consistent with the prior
ENCODER_GEOMETRY_PREREG expectation.

Mean fixed-population forgetting, medians. Frozen-B: HDC 0.0497 versus NCM 0.0435
and RanPAC-style 0.0451. Adapted-B: HDC 0.0395 versus 0.0533 and 0.0651, but the
adapted margin holds in only 2/3 seeds - seed 2 reverses at 0.0640 versus NCM
0.0627. Reduced forgetting is therefore NOT established; it is encoder-dependent
and seed-inconsistent. Forgetting must not be read alone: replay records the
grid's lowest forgetting (0.0004 frozen) with its worst Macro-F1 (0.2369), and
naive/EWC show the same pattern on frozen.

Uptake repair executed post-hoc over all 36 cells; original snapshot uptake in
metrics.json is unchanged. Support is 67/59/26/24 examples for the four novel
classes, giving 48/40/7/5 twenty-example window positions. Relaxing to
ten-example windows raises frozen stabilisation from 2/4 to 3/4, confirming one
class was unreachable by construction, but all three prototype learners move
together. Mean terminal accuracy over the novel classes is HDC 0.7000 versus
0.7125/0.7125 frozen and 0.4500 versus 0.4625/0.4375 adapted; HDC leads in 0/3
frozen and 1/3 adapted seeds. Novel-class uptake is a null result and does not
separate the prototype learners on any of the four criteria.

trusted_val is a development split and has now been used for development; this is
not held-out evidence and no test partition was accessed. Throughput and memory
remain measured losses, unchanged by this grid. Report remains PROVISIONAL.
Supervisor briefing: https://claude.ai/artifact/36wbcTZRNCayj22t6M619q

## 2026-09-15 - Adapted-arm advantage is concentrated in two classes

Post-hoc per-class analysis of the completed grid's saved confusion matrices. No
rerun, rescoring or artifact modification; development-only, no test access.

The adapted-B Macro-F1 margin is not broad. Mean per-class F1 over three seeds
gives HDC at or above the best control on 13/19 classes and behind on 6. Two
classes supply almost all of the aggregate: 2_00_0 at +0.678 (HDC 0.741 versus
NCM 0.062 / RanPAC-style 0.059) and 3_00_0 at +0.412 (HDC 0.429 versus
0.017/0.017). Excluding those two, the macro delta falls from +0.0635 to +0.0069.

Those two classes are control failures rather than HDC strengths. Seed-0 recall on
3_00_0 (n=236): frozen HDC 67.8% / NCM 63.1%; adapted HDC 27.1% / NCM 0.8%, two
correct of 236, with NCM predictions scattered across 5_b7_1, 3_b6_1 and 5_a7_2
rather than a single collapse target. On 2_00_0 (n=29): frozen both 100%; adapted
HDC 69.0% / NCM 3.4%, with 19 of 29 going to 1_00_0. HDC also degrades sharply on
the adapted arm; it does not fail.

Consequence for wording: the defensible claim is graceful degradation under
representation damage, not a general HDC predictive advantage. Any aggregate
Macro-F1 figure must be reported with the per-class breakdown. Note also that the
best cell in the whole grid is RanPAC-style on frozen at 0.7406, above HDC's own
best of 0.7220; the advantage exists only on a representation that is worse for
every learner.

The aggregation mechanism (sign-based majority accumulation versus cosine-mean
displacement) is a hypothesis consistent with these observations, not a tested
attribution. A median/sign-aggregation prototype control would be required to
separate hyperdimensional encoding from robust aggregation.
## 2026-09-16 - Mechanism ablation COMPLETE (120 cells); collapse resistance is bounded-influence aggregation, not the hyperdimensional projection

Backfilled/merged from `PENDING_CHECKPOINT_runner.md` (mechanism_ablation) and
`PENDING_CHECKPOINT_aggregator.md` (forgetting_grid_development), both written
on the `restore-point/2026-09-15-grid-and-ablation-prep` branch and left in
place per instruction (not deleted by this entry).

**Unblocking work** (commit `3259b7a`, merging the two pending-checkpoint
files above): three defects were found and fixed, verified against real
artifacts, not left open. (1) `evaluate_snapshots.py` did not thread
`ablation_mode=bool(manifest.get("ablation_mode", False))` into the
`capture._prepare(...)` call inside `_prepare()` (previously always `False`),
so evaluating any ablation capture hit the canonical gates and failed with
`ValueError: unknown learner`. Reproduced against a real PASS capture
(`sign_ncm`/`b_frozen`/seed 0, `ablation_mode: true`) before the fix; after
the fix, `evaluate_snapshots --execute --preflight-confirmed` against the same
capture produced `status: PASS`, `snapshots_scored: 20`,
`state_and_rng_unchanged: true`. Not part of the capture's pinned
`code_hashes` set, so the archived 36-cell grid is unaffected. (2) No CLI path
existed for the Phase-3 hypervector-dimension sweep; `budget_experiment.py`
gained `--hypervector-dim` (int, optional, requires `--ablation-mode`,
`parser.error` otherwise), threaded through `experiment_a._prepare(...,
hypervector_dim=None)` (locked to `{1000, 2000, 10000}`, requires
`ablation_mode`) and `stream_runner.py` (`hdc` learner only; `ranpac`'s
`projection_dim` still reads `spec.hypervector_dim` unchanged). Recorded only
as `ablation_hypervector_dim` in manifest/report, only when `ablation_mode` is
true - never in `policy` (so `policy_sha256` is untouched) and never added to
the 13-key checkpoint-context whitelist; canonical (non-ablation) manifests
are byte-structurally unchanged. `scripts/run-mechanism-ablation-all.ps1`
(new, 120-cell batch launcher mirroring `run-forgetting-grid-all.ps1`) was
added with receipt-based completion, 3-attempt escalation, per-cell 8 GiB
guard and workload preflight, and a single case-sensitive `RUN ABLATION`
confirmation; `-PlanOnly` verified 120 cells in phase2(30)->phase1(45)->
phase3(45) order with no writes. Tests added (none modified):
`test_aggregation_prototypes.py`, `test_hypervector_dim_ablation.py`,
`test_mechanism_ablation_batch_runner.py`. (3) `aggregate_grid.py` rejected
Phase-3 directory names (e.g. `p3_b_adapted_hdc_d10000`); fixed in commit
`126ea06`.

**Result** (120/120 cells, 240/240 capture+validation integrity receipts
PASS, all attempt01, 400 labels consumed in every cell for every
label-matched learner; run order Phase 2 -> Phase 1 -> Phase 3, chosen because
Phase 2 is decisive and, if interrupted, leaves the decisive result already in
hand). Source: `results/mechanism_ablation/aggregate_all_120/cells.csv`,
`summary.json`, `PROVENANCE.md` (post-hoc, read-only aggregation; no capture,
evaluation, training, or `trusted_splits/test.csv` access).

Phase 1 (HDC, NCM, RanPAC-style x five arms) exactly reproduces the archived
36-cell grid on the shared frozen-B/adapted-B arms: frozen ViT-B HDC 0.7220 /
NCM 0.7313 / RanPAC-style 0.7406; adapted ViT-B HDC 0.6001 / NCM 0.5275 /
RanPAC-style 0.5243 (median macro-F1, seeds 0-2).

Phase 2 (Sign-NCM, Median-NCM x five arms) is the decisive phase: does a
bounded-influence aggregation rule with no hyperdimensional projection
reproduce HDC's collapse resistance on adapted ViT-B? Adapted ViT-B medians
over seeds 0-2, macro-F1 / `3_00_0` F1 / `2_00_0` F1: HDC 0.6001/0.4238/0.7407;
Median-NCM 0.5925/0.4080/0.7407; Sign-NCM 0.5350/0.2918/0.7407; NCM
0.5275/0.0168/0.0625; RanPAC-style 0.5243/0.0168/0.0588. Under the
pre-declared locked decision rule (indeterminate band 0.10-0.30 on `3_00_0`,
adapted ViT-B): Sign-NCM scores 0.2918 - inside the indeterminate band,
recorded as indeterminate for that cell, not a pass or a collapse. Median-NCM
clears 0.30 (0.4080). Sign-NCM clears 0.30 on adapted ViT-L (0.5094). Nothing
tested fell below the 0.10 floor. Adapted ViT-L is an independent check from a
different angle: plain NCM does not collapse there (`3_00_0` F1 = 0.5521),
and NCM's macro-F1 (0.6579) beats HDC's (0.6308) on that arm - the
adapted-ViT-B collapse in NCM/RanPAC-style is specific to that one encoder
arm. Taken together, the result is consistent with collapse resistance
tracking the aggregation rule, not the hyperdimensional projection.

Phase 3 (HDC at hypervector dimension D in {1000, 2000, 10000} x five arms),
adapted ViT-B, macro-F1 / `3_00_0` F1 / `2_00_0` F1: D=1000
0.5374/0.3505/0.7692; D=2000 0.5720/0.3933/0.7407; D=10000
0.6001/0.4238/0.7407. HDC does not collapse at any tested D, but macro-F1
rises with D; D=10000 reproduces the Phase-1 HDC numbers exactly on all five
arms. The projection is not inert - it improves HDC's own accuracy with
dimension - but dimension does not supply the collapse resistance itself,
which Phase 2 isolates to the aggregation rule.

Best macro-F1 across all 120 cells: Median-NCM on frozen ViT-B, 0.7420 - above
every HDC cell in the ablation. Median-NCM's implementation stores every
confirmed example (an exact running median); its accuracy figures are valid
evidence, but no throughput or memory figure may be claimed for it - that cost
was never measured. Full test suite at last verified run: 559 passed, 6
subtests passed.

A draft release protocol for the sealed trusted-test partition was added on
the same branch as commit `4859fb2`
(`docs/TODO/HDC_TEST_RELEASE_PROTOCOL_2026-09-15.md`) - status DRAFT, not
approved; it proposes closing gate A2 (defined in
`docs/TODO/HDC_CORE_EVALUATION_PROTOCOL_DRAFT_2026-09-09.md:88-90`) but
requires explicit user and supervisor sign-off first. `trusted_splits/test.csv`
remains sealed and was never opened by this ablation.

Note on framing (labelled, not a decision): this result points toward a
possible reframing of the thesis around bounded-influence aggregation rather
than HDC specifically. This does not close gate A2 or any other gate and does
not change the thesis title; that decision belongs to the user and
supervisor. `trusted_val` remains a development split; RanPAC-style remains an
ablation, not published RanPAC; naive/EWC/replay remain not label-matched.
Supervisor briefing updated: https://claude.ai/artifact/36wbcTZRNCayj22t6M619q
(Version 4). Report remains PROVISIONAL. No experiment, capture, evaluation,
or GPU work was launched by this record-refresh pass; no code was modified by
it.
