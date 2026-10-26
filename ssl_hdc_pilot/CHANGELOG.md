# SSL-HDC Pilot Changelog

Append-only.

## 2026-07-18

- Created the `ssl_hdc_pilot/` scaffold.
- Verified `transformers==5.12.1` and `scikit-learn==1.7.2` in `.venv-e1`.
- Installed `torch-hd==5.8.4` (Python import name `torchhd`) into `.venv-e1`.
- Kept the headline encoder default locked to
  `facebook/dinov3-vits16-pretrain-lvd1689m`.
- Added `scripts/build_unlabeled_manifest.py` with dataset-count reconciliation,
  val-duplicate documentation, missing-image checks, and trusted-ID exclusion.
- Added `scripts/extract_embeddings.py` with offline cached model loading,
  frozen/eval guards, CLS pooling, float32 NPZ output, preprocessing metadata,
  smoke budgets, and an explicit conditional ViT-B opt-in guard.
- Added `scripts/build_trusted_splits.py` for Option-A 60/20/20 splits and
  Option-B repeated stratified folds. Official writes require explicit
  `--supervisor-approved`.
- Added fail-closed `scripts/adapt_dinov3.py`. It locks the unambiguous ViT-S,
  short-schedule, low-LR, EMA, multi-crop, AMP, gradient-checkpointing, and
  smoke-budget settings but refuses training until the probe policy is resolved.
- Added 14 unit tests across the four required public seams.
- Hardened official trusted-split writes after spec review: canonical input
  path, source size/class count, exact Option-A tail exclusions, 19 retained
  classes, fixed seed 20260716, fixed Option-B 5-fold x 10-repeat settings,
  and all mandatory
  disclosures are now fail-closed guards.
- Generated the sanctioned 51,906-row unlabeled manifest and a permitted
  50-image Arm-F smoke artifact with 384-d float32 embeddings.

No official run was launched by the agent.

## 2026-07-19 - Phase 4 continual-learning machinery

- Added a torchhd-backed 10,000-dimensional bipolar HDC learner with fixed
  seeded projection, additive prototype accumulators, documented stable-tie
  re-binarization, cosine-margin confidence, and confirmed-label new-class
  registration. It has no backpropagation or epochs.
- Added naive incremental, online-EWC, and deterministic reservoir-replay
  neural-head baselines on the identical frozen embeddings. All share the same
  head family, confidence trigger, confirmed-label interface, and budget
  accounting; a non-positive EWC coefficient and zero replay capacity are
  rejected.
- Added the deterministic spec-driven stream simulator, strict label-free NPZ
  guard, supervisor split provenance guard, per-step history, tagged summaries,
  and all four learner-by-threshold runs in one invocation.
- Added continual metrics for forgetting, new-class uptake, update compute,
  label budget, macro-F1, macro-recall, per-class/per-stage recall, and ECE.
- Added the strict v1 JSON schema and protocol documentation. The canonical
  `stream_spec.json` remains intentionally absent pending preregistration; no
  official protocol value was invented.
- Added thirteen synthetic Phase 4 tests. Full pilot verification is 46 passed
  plus 6 subtests; all modified Python files compile.
- Appended user-launch-only official blocks, blocked on approved splits and the
  canonical stream spec. Arm D is additionally blocked on its probe gate and
  selected trusted embedding NPZ.
- Post-review fixes made `seeds.stream_order` an explicit preregistered sweep,
  allowed multiple class activations at one step, aligned neural training and
  confidence on identical cosine scores, and split cumulative prequential
  metrics from a post-stream no-update final evaluation.
- Extended `common/thesis_eval.py` with shared string-label classification and
  confidence/correctness ECE helpers; continual metrics now reuse the common
  evaluator instead of maintaining parallel metric formulas.
- Marked the historical Phase 4 deferral as superseded and refreshed the main
  context date.

No official run was launched by the agent.

## 2026-07-19

- Replaced the Arm-D fail-closed placeholder with real label-free DINO
  self-distillation on the sanctioned 51,906-row manifest only.
- Added two-global/four-local multi-crop augmentation, EMA teacher updates,
  AMP, gradient checkpointing, low-LR and smoke-budget guards, and atomic
  epoch checkpoints including the frozen `checkpoint_epoch0` baseline.
- Added resume-safe optimizer, scaler, teacher, projection-head, center, and
  student-encoder restoration. Resume config mismatches hard-error.
- Added `scripts/probe_gate.py` with the amendment-fixed
  `LogisticRegression(max_iter=2000, class_weight="balanced",
  random_state=20260716)`, validation macro-F1, 0.005 tolerance, and
  highest-score checkpoint selection.
- Added immutable supervisor-approved provenance columns to official split
  outputs; the probe gate fails closed when those fields or values are absent.
- Added resume/config, checkpoint-state, provenance-refusal, and synthetic
  better/worse gate tests. The full pilot suite contains 21 passing tests.
- Installed `pytest==9.1.1` into `.venv-e1` for the requested unit-test run.
- Completed an eight-image/one-step Arm-D CUDA smoke and verified identical
  command resume behavior; temporary smoke artifacts were deleted after
  logging.
- Post-review hardening enforces loadable NPZ checkpoint completeness, exact
  Option-A train/validation sizes and 19-class coverage, canonical trusted
  image/label agreement, exact tail-class exclusion, raw 384-d probe features,
  full remainder batches, verifiable checkpoint state schemas, and package-safe
  `python -m` official commands.

No official run was launched by the agent.

## 2026-07-19 - Round 3 launch hardening

- Changed the adapted inference artifact from the online student to the EMA
  teacher; resumable checkpoints now store the student encoder separately.
- Added fail-fast rejection of NaN/Inf losses before any parameter, EMA, or
  center update.
- Made the probe require every checkpoint from epoch 0 through the configured
  final epoch and reject checkpoints that do not export the EMA teacher.
- Added a smoke-only early-stop control and a strict ten-total-step ceiling so
  a two-process epoch-1 to epoch-2 CUDA continuation can be verified without
  weakening official configuration identity.
- Resume now fails closed if the complete checkpoint epochs are not contiguous;
  exact optimizer and both-head restoration are asserted tensor-for-tensor.
- Added six Round-3 guard tests; the full suite now contains 27 passing tests.
- Verified real-model CUDA continuation, EMA/student separation, finite loss
  history, full cadence, and completed-run no-op behavior.

No official run was launched by the agent.

## 2026-07-19 - Arm-D prelaunch throughput selection

- Completed the permitted eight-setting CUDA smoke sweep for batch sizes
  2/4/8/16 and DataLoader worker counts 0/2, with ten steps per setting.
- Selected batch size 16 and workers 0 from measured throughput and the
  predeclared peak-VRAM rule; learning rate and all other recipe parameters
  remain unchanged.
- Appended a new authoritative user-launch block and replaced the former
  65-hour tiny-step planning basis with a measured 3.95-hour training-loop
  estimate and 5-hour wall-clock allowance.

No official run was launched by the agent.

## 2026-07-19 - Arm-D extended batch-size selection

- Extended the bounded CUDA sweep to batch sizes 32 and 64 with DataLoader
  worker counts 0 and 2, staying within 512 images and ten steps per setting.
- Selected batch size 64 and workers 0 under the unchanged peak-VRAM rule.
  Learning rate and every other recipe parameter remain unchanged.
- Appended a superseding authoritative user-launch block with a measured
  3.19-hour training-loop estimate and 4-hour wall-clock allowance.

No official run was launched by the agent.

## 2026-07-19 - Authoritative Arm-D throughput selection

- Added measurement-only CUDA peak-allocation telemetry to `history.csv`
  using `torch.cuda.max_memory_allocated`, with a guard test. Training loss,
  optimizer, EMA, crops, LR, epochs, and update order are unchanged.
- Re-ran all 12 batch/worker settings on an otherwise idle GPU with both
  PyTorch allocator peak and whole-device `nvidia-smi` peak measurement.
- Applied the fixed headroom and 8,000-update schedule rules mechanically:
  selected batch size 32 and workers 0; batch 64 is information-only.
- Appended the final user-launch command with the measured 2.97-hour
  training-loop estimate and approximately 4-hour wall-clock allowance.
- Added one telemetry schema test; the full pilot suite now has 28 passing
  tests.
- Post-review launch hardening marks and comments every historical Arm-D
  command as superseded, leaving only the final batch-32 block runnable.

No official run was launched by the agent.

## 2026-07-19 - Encoder-size LADDER implementation

- Replaced the ViT-S-only Arm-D guard with the exact preregistered
  ViT-S/16, ViT-B/16, and ViT-L/16 whitelist.
- Bound every model ID and immutable resume config to its required
  `arm_d_vits16`, `arm_d_vitb16`, or `arm_d_vitl16` output directory.
- Locked non-smoke runs to the shared five-epoch ladder recipe: LR 1e-5, EMA
  0.996, two global plus four local crops, AMP, gradient checkpointing, batch
  32, and workers 0.
- Made frozen extraction validate and record the rung model ID and exact
  384/768/1,024-d feature width.
- Made the probe gate derive its model and dimension from the checkpoint run,
  reject cross-rung feature widths, and write the rung into its verdict.
- Added the single authoritative S-to-B-to-L user-launch ladder section,
  including required Arm-F extraction, measured wall-time arithmetic, the
  ViT-L idle-GPU warning, and supervisor-split-blocked per-rung probe commands.
- Added guard tests and completed one permitted CUDA optimization step for each
  rung. All three size-specific smoke directories checkpointed successfully and
  were deleted after logging.
- Full `ssl_hdc_pilot` suite: 33 tests passed; modified Python files compiled.

No official run was launched by the agent.

## 2026-07-19 - Section 6a generator and selected-checkpoint extraction

- Added `scripts/build_stream_spec.py`. It accepts only the approved Option-A
  directory and applies the locked four-rarest-disease selection, alphabetical
  tie break, 25% per-known-class holdback, descending-count 20/40/60/80%
  introductions, seeds 0/1/2, thresholds 0.05/0.10/0.20, and confirmed-event
  accounting without exposing those values as flags.
- Added per-seed holdback manifests to the runner schema so the generated
  initial/stream partition is auditable and deterministic. Legacy synthetic
  `confirmed_per_class` specs remain smoke-only; official mode now rejects
  them and enforces the complete Section 6a seed/threshold/class/accounting
  contract. A novel class's first sample is forced at its exact entry step.
- Factored encoder identities and the EMA-teacher checkpoint validator/strict
  loader into shared modules used by adaptation, the probe gate, and embedding
  extraction.
- Added `extract_embeddings.py --checkpoint-dir`, model/dimension guards, and
  nested checkpoint provenance containing rung, epoch, EMA role, and immutable
  run-config SHA-256.
- Added the single authoritative continual-study user-launch section with the
  runnable generator, selected ViT-B epoch-3 and ViT-L epoch-4 extraction
  commands, and the blocked 72-run headline grid.
- Retained a permitted 50-image ViT-B epoch-3 smoke artifact. It contains 50
  label-free float32 vectors at 768 dimensions; extraction itself took 1.084
  seconds after model loading.
- A synthetic 768-d one-seed/one-threshold timing smoke across all four
  learners took 3.865 seconds for 44 stream examples. No official continual
  setting was launched.
- Hardened official execution to recompute the locked rare-class selection,
  novel entry order, stream length, and deterministic 25%-per-class manifests
  directly from the canonical provenance-stamped train/validation/test CSVs
  before accepting the generated specification.
- Full `ssl_hdc_pilot` suite: 58 passed plus 6 subtests; modified Python files
  compiled and the JSON schema parsed.

No official run was launched by the agent.

## 2026-07-20 - EWC task-boundary consolidation correction

- Fixed the online-EWC anchor lifecycle. The previous implementation updated
  the anchor to the current parameters after every confirmed example, so the
  next update began with parameter equal to anchor and update_steps=1 made the
  regularization penalty identically zero.
- Kept per-update online Fisher accumulation and the existing ewc_decay
  recurrence unchanged, but separated it from the frozen Fisher used by the
  penalty.
- Consolidated both the parameter anchor and a frozen Fisher snapshot only
  when register_class accepts a genuinely novel class. Before the first such
  boundary the penalty is intentionally zero.
- Added a production-recipe regression with update_steps=1 that proves a
  positive post-boundary penalty, a different parameter trajectory, and at
  least one different prediction versus naive fine-tuning. The existing EWC
  constraint test now includes a real novel-class boundary.
- Added the optional stream-runner learner selector used by the correction
  runbook. The default remains the complete HDC/naive/EWC/replay grid.
- Added an EWC-only user-launch block for the frozen-B and adapted-B grids. It
  replaces exactly the 18 stale EWC files per output directory and hash-checks
  all non-EWC outputs before and after.
- Full ssl_hdc_pilot suite: 60 passed plus 6 subtests in 9.71 seconds.

No official EWC rerun was launched by the agent.

## 2026-07-20 - EWC bounded-decay amendment implementation

- Changed the canonical Section 6a stream-spec generator from ewc_decay=1.0
  to the binding amended value 0.9. ewc_lambda remains 10.0 and no other
  baseline or stream-protocol setting changed.
- Regenerated the canonical non-smoke stream_spec.json from the approved
  Option-A split. Its new SHA-256 is
  62b545d31d592735be4404451bddd7f603576d127351f337e1e34c8fd04604cc.
- Kept the d46b3ea task-boundary consolidation logic unchanged. Added an
  EWC-specific non-finite-loss message that reports both decay and lambda;
  the guard still aborts instead of normalizing or clamping the penalty.
- Added regression coverage showing identical observations accumulate without
  bound at decay 1.0, while decay 0.9 stays below the geometric-series ceiling.
  The generator test locks decay 0.9 and lambda 10.0.
- Replaced the EWC-only user-launch block to support the current partial state:
  it snapshots every non-EWC file in both arms before deletion, removes any
  number of existing EWC files, reruns frozen-B then adapted-B, requires nine
  fresh summaries plus nine histories per arm, validates the new stream-spec
  hash in every summary, and rechecks all non-EWC hashes.
- Full ssl_hdc_pilot suite: 62 passed plus 6 subtests in 10.01 seconds.

No official EWC rerun was launched by the agent.

## 2026-07-20 — Frozen 7B ceiling rung (arm_f) enabled in the encoder ladder

- `encoder_models.py`: added `VIT7B16_MODEL_ID`
  (`facebook/dinov3-vit7b16-pretrain-lvd1689m`, 4096-d CLS) to
  `LADDER_MODEL_SPECS` with a new `adaptable` flag set False. The 6.7B backbone
  is a FROZEN-ONLY ceiling reference: label-free DINO adaptation of a 7B model
  does not fit the 24 GB lab 4090, so it is extraction-only (arm_f). The
  adaptation ladder (arm_d) remains exactly ViT-S/B/L.
- `adapt_dinov3.py::validate_config`: rejects non-adaptable rungs with a clear
  error, so the 7B can never be sent down the arm_d training path.
- `extract_embeddings.py`: added `--dtype {float32,bfloat16,float16}` for the
  frozen path (loads directly in dtype via `torch_dtype`, casts float inputs to
  the model dtype). bfloat16 puts the 7B weights at ~13.4 GB, within the 4090's
  24 GB; float32 (~27 GB) would not fit. Checkpoint (arm_d) extraction stays
  float32. `encoder_dtype` recorded in metadata.
- Tests: adaptation-guard test now asserts the ADAPTABLE set is exactly the
  three S/B/L rungs and that the 7B is rejected for adaptation; embeddings
  ladder test expects the four-rung set. Full pilot suite: 63 passed, 6
  subtests.
- The official 7B trusted extraction is user-launched on the confirmed 4090
  (see RUN_COMMANDS.md). No official run was launched by the agent. The 7B
  weights are not downloaded; the gated-download command is provided
  separately.


## 2026-08-02 - Decisive prototype baselines and replay audit

- Added `continual_prototypes.py` with online L2-normalized-embedding NCM and a
  RanPAC-style fixed Gaussian projection, ReLU, running class prototypes, and
  cosine-margin inference. RanPAC uses the same projection seed and
  10,000-dimensional width as HDC; no decorrelation or ridge was added.
- Added guarded `--learners` selection while preserving `--ewc-only`.
  Default complete-grid membership is now HDC, NCM, RanPAC, naive, EWC, replay.
  Historical four-learner launch blocks are pinned to their original set.
- Recorded learner-specific configuration in every newly written NCM and
  RanPAC summary.
- Audited replay end to end. Reservoir Algorithm R, embedding/label alignment,
  current-sample inclusion, add-after-update ordering, dynamic class indices,
  sampling, and seeding are correct.
- The ordinary mean loss gives the current event 1/33 of a full 32-item replay
  update. That is the registered, conventional current-plus-replay batch, not a
  demonstrated implementation defect. It remains unchanged to avoid
  outcome-driven protocol modification. The weak replay result is therefore
  retained as real under the registered configuration.
- Added a hash-protected user-launch block for exactly 36 new NCM/RanPAC cells.
- Verification: focused tests 19 passed; full SSL-HDC suite 73 passed plus
  6 subtests in 11.53 seconds; prototype and runner Python modules compiled; the
  new PowerShell block parsed.
- Measured prototype smoke: 0.514 seconds for both learners on 44 synthetic
  768-D examples using CUDA. The runbook records an approximately 11-minute
  projection-count estimate and a 25-minute allowance for the 36-cell grid.
- `results/SSL_HDC_REPORT.md` was not modified.

No official run was launched by the agent.

## 2026-09-07 — Documentation reconciliation

- Replaced obsolete scaffold/pending-run handoff text with the verified 108-run state.
- Updated the report and audit for NCM/RanPAC, prequential scores, stage recall,
  ECE, retention, uptake, and saved update timing; retained PROVISIONAL status.
- Corrected final-evaluation population, query-versus-calibration confidence,
  bootstrap accounting, replay's adapted-arm exception, and unsupported
  universal HDC retention/speed claims.
- Marked completed runbook blocks and the old summary figure as historical;
  preserved commands, protocol parameters, and scientific artifacts.
- Refreshed root/context/exploration entry pages and recorded the checkpoint.
  No code change, test-suite run, or evidence-bearing experiment.

## 2026-09-09 — Core evaluation protocol draft

- Added a local [protocol agreement draft](../docs/TODO/HDC_CORE_EVALUATION_PROTOCOL_DRAFT_2026-09-09.md)
  for user review, based on TODO v2; linked it from README and the handoff.
- Proposed separate shared-label and equal-allowance controls, state/replay
  acceptance criteria, fixed-population metrics, paired statistics, and a locked
  test-opening gate. Quantum remains an optional extension.
- Flagged the existing approved 25-class CV obligation and longitudinal-test
  requirement for explicit reconciliation; no binding protocol was amended.
- Preserved the user's TODO edits and completed artifacts. Documentation only;
  no code changes, test-suite runs, or test-set evaluation.

No official run was launched by the agent.

## 2026-09-10 — Protocol draft review amendments

- Incorporated the user's five review recommendations into the existing draft:
  explicit endpoint-hierarchy amendment/RQs, claim-to-evidence rules, upstream
  supervision accounting, latency plus state/replay storage scope, and A1–A4
  freeze decisions with proposed resolutions and required sign-off evidence.
- Distinguished an observed advantage from equivalence or mechanism attribution;
  per-run labels from project supervision; classifier payload from total memory;
  research-machine measurements from optional CPU/deployment benchmarking.
- Updated the handoff. The draft is still unfrozen; the binding specification,
  TODO, code, runbook, and scientific results remain unchanged in this checkpoint.
  No test-suite run, benchmark, or test-set evaluation occurred.

No official run was launched by the agent.


## 2026-09-10 — supervisor forgetting/throughput priorities (documentation revision)

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

## 2026-09-10 — Authorized development steps 1–2: learner checkpoints

- Aligned the TODO with primary forgetting/throughput and required predictive,
  uptake, label, and memory evidence; added explicit benchmark implementation
  tasks without authorizing or running them. A1–A4 remain open.
- Added versioned NPZ save/load for all six learners, preserving projections,
  registry/order, HDC ties, prototype counts, neural state/config/modes/gradients,
  EWC anchors/Fishers, replay buffer/seen/private RNG, and optional global RNG
  restoration. Loading is non-pickle; saves refuse overwrite.
- Added opt-in `--save-final-state` / `save_final_state=True` to the runner,
  recording progress and input/source hashes before the historical training-pool
  final pass. Default histories/summaries/CLI output stay unchanged.
- Added a checkpoint-only neural-construction flag to avoid reseeding deferred
  CUDA RNG during CPU load. Ordinary construction and update formulas remain
  unchanged. A regression test reproduced and now guards the review finding.
- Added LEARNER_CHECKPOINTS.md and refreshed the draft/handoff. Artifacts are
  learner-resumable, not whole-stream resume; no historical grid was reconstructed.
- Verification: 36 focused tests passed; full CPU suite 89 passed plus six
  subtests in 10.84 seconds. Tests use synthetic data. Initial sandbox pytest
  attempts hit temporary-directory ACL failures; approved outside-sandbox runs
  completed. No GPU benchmark or held-out evaluation occurred.

No official run was launched by the agent.

## 2026-09-11 — Reconstruction infrastructure and manual-run handoff

- Added `reconstruct_run.py`: read-only default planning, explicit one-cell
  execution, exact registered source-spec recovery, immutable-by-convention
  fresh outputs, state preservation and fidelity comparisons. No learner formula
  or canonical stream specification changed.
- Official reconstruction checks canonical train paths before reading; it uses
  approved training provenance and training-only holdback derivation, not the
  generator's validation/test rarity check. Added a filesystem regression.
- Added an optional final-prediction callback to the runner, preserving existing
  summaries and default behavior. Reconstruction saves ID-aligned final training
  predictions, environment/configuration and code/input provenance, plus a linked
  fidelity report. State preservation is not whole-stream resume.
- Added RECONSTRUCTION.md, appended a guarded HDC-local manual first-cell block,
  and updated TODO/handoff/migration wording. Historical command blocks remain
  unchanged. All experiments remain manually user-launched.
- Verification: full CPU suite 114 passed plus six subtests in 31.53 seconds;
  manual PowerShell syntax and diff whitespace checks passed. Both code-review
  axes cleared their blocking findings. First-cell metadata/hash planning passed;
  actual historical fidelity remains unverified. No official artifacts changed.

No official run was launched by the agent.

## 2026-09-11 — First user-run reconstruction verified

- Recorded frozen-B/NCM/seed-0/threshold-0.10 PASS: 976 stream steps, no
  history/summary mismatches. Verified saved artifact and current input/code
  hashes read-only; final ledger contains 3,396 training-pool predictions.
- Updated TODO/handoff/reconstruction guide; appended guarded manual commands
  for the other 11 representative cells, excluding the completed NCM cell.
- No algorithm changes, benchmark claims, held-out evaluation, or automatic
  experiment launch. Full historical fidelity and A1-A4 remain pending.

The user launched the first cell. No official run was launched by the agent.

## 2026-09-11 — Twelve representative reconstructions verified

- Recorded all 12 representative cells PASS; independently hash-verified the
  newest 66 artifacts and current input/code provenance across all 12 cells.
- Preserved the prematurely ended batch transcript and documented its coverage
  gap. The final Stop-Transcript exception is not a failed learner run; the
  cause of early transcript termination remains unresolved.
- Appended guarded manual commands for the remaining 96 cells with direct logs,
  prior-artifact checks, duplicate exclusion and a free-space guard. Syntax and
  read-only planning guards passed. No experiments were launched or overwritten.
- Updated TODO and handoff. No algorithm change, new benchmark or held-out score.

The user launched all experiments. No official run was launched by the agent.

## 2026-09-11 — All historical final states reconstructed and verified

- Recorded 108/108 user-run fidelity PASS with exact original-grid coverage;
  independently verified 648 artifact hashes, current input/code provenance,
  and 108 final training ledgers of 3,396 unique IDs each.
- Added the full reconstruction audit and marked TODO 1.3/1.4/C/R2 complete.
  Updated handoff, checkpoint guide, README and draft progress; marked runbook
  reconstruction blocks completed. No scientific protocol choice was frozen.
- Next input gate is Experiment A's proposed supervision policy. A1-A4 remain
  open. No learner code change, held-out evaluation or benchmark was performed.

The user launched all experiments. No official run was launched by the agent.

## 2026-09-11 — Approved Experiment-A acquisition development

- Recorded subsequent user approval of budgets 0/25/50/100/200/400 and shared
  label-blind nested schedules with paid-label-only novel registration.
- Added experiment_a.py and an opt-in shared-label runner path, fresh-directory
  outputs, exact event/accounting guards, training-only input checks and manifests.
- Fixed the review-discovered preparation-time input hashing race. Added 41
  synthetic tests; full wrapper suite passed 155 tests plus six subtests (54.80s).
  Standards/spec review: no remaining blockers in this bounded scope.
- Marked TODO 2A.1-2A.2 complete, added EXPERIMENT_A.md, updated protocol/handoff,
  and supplied one direct-log manual development-cell command. Fixed evaluation,
  the full study and A1-A4 are not complete or approved by this checkpoint.
- Preserved completed historical artifacts and original manifest code hashes;
  the runner's new code revision does not retroactively alter their provenance.

No official run was launched by the agent. Changes remain uncommitted.

## 2026-09-11 — Saved-state validation development

- Independently verified the completed user-run A25 artifacts/accounting and
  marked its manual training check complete without altering or rerunning it.
- Added validation-only saved Experiment-A state evaluation, fixed 19-class
  metrics, disjoint population checks, source provenance and state/RNG purity.
  Default planning does not load a learner or score predictions.
- Added 23 synthetic tests; full wrapper passed 178 tests plus six subtests in
  42.10 seconds. Standards/Spec reviews found no blockers in this bounded scope.
- Updated TODO/handoff and appended the manual first-validation command. Actual
  validation, budget-wide acceptance, longitudinal metrics and test remain pending.

No official run was launched by the agent. Changes remain uncommitted.

## 2026-09-11 — A25 validation verified and longitudinal arithmetic added

- Audited the completed user-run validation's three artifact hashes and 1,132
  unique prediction rows; reconstructed classification metrics/ECE exactly.
- Added pure fixed-population forgetting, uptake, AULC and paired descriptive
  summary helpers with 44 synthetic tests. Explicit caller-supplied grids and
  targets avoid silently approving scientific choices; no run integration yet.
- Full wrapper passed 222 tests and six subtests (46.26s). Standards/Spec reviews
  found no blockers. Preserved existing learner/metric code and all artifacts.
- Updated TODO, handoff and completion markers; documented the next development
  approval gate for Experiment B and longitudinal snapshot settings.
- Recorded that the agent reads/analyses saved files directly; user-launch
  commands are for new experiments, not report display.

No official run was launched by the agent. Changes remain uncommitted.

## 2026-09-12 - A/B budget and approved snapshot capture

- Recorded explicit bounded approval of B acquisition and retention/uptake grids
  and targets. Added budget_experiment.py, passive snapshot capture, event
  accounting and capture-time artifact checks; no learner formula changes.
- Full wrapper passed 281 tests plus six subtests in 89.66s, including all six
  learners, zero/exhausted budgets, underspending and exact state/RNG checks.
- Updated TODO 2B.1-2B.3, protocol approval wording, handoff and linked capture
  guide. New-format validation integration and real longitudinal evidence remain
  pending; legacy A25 validation is not a loader for new snapshot artifacts.
- Added guarded user-launch B400/NCM/frozen-B/seed-0/.10 development block with
  direct logs. Read-only real-input plan passed; existing results stay preserved.

No official run was launched by the agent. Changes remain uncommitted.

## 2026-09-12 - B400 verified; longitudinal validation and one-command launcher

- Verified user-run B400: 27 hashes, 976 events, 400 purchases/450 requests,
  50 denials, all novel registrations and 20 states. Marked training complete.
- Preserved split preflight/recovery logs after the detached -FilePath paste
  error. Added a script launcher with explicitly bound logging parameters.
- Added strict new-format snapshot validation and fixed-population metric
  integration without altering learner/runner/common metric code or artifacts.
- Full wrapper passed 324 tests plus six subtests (98.46s). Standards review
  found no issues; fixed Spec partition-filter finding with a regression test.
- Real-input plan passed for 20 states x 1,132 validation examples. Updated
  handoff/TODO/runbook and SNAPSHOT_VALIDATION.md; actual scoring remains manual.

No real experiment was launched by the agent. Changes remain uncommitted.

## 2026-09-13 - B400 longitudinal validation accepted

- Verified user-run validation: 22 artifact hashes, 48 pins, all 20 ledgers and
  all derived metrics. Added the audit with explicit single-cell claim limits.
- Updated TODO 3.1-3.5 and bounded dry runs 4.2-4.4; marked both B400 commands
  completed provenance. Preserved source files, reports and logs unchanged.
- Added a proposed benchmark development policy requiring explicit approval;
  no timing policy, CPU deployment scope or final A4 freeze silently adopted.
- Documentation-only checkpoint; no training, model evaluation or benchmark
  launched by the agent. Prior test result: 324 tests plus six subtests.

Changes remain uncommitted; unrelated work is preserved.

## 2026-09-13 - Approved bounded benchmark implementation

- Added separate benchmark timing, environment and study modules plus a guarded
  one-command A25 six-learner pilot launcher; real execution remains user-owned.
- Added state/RNG reference fidelity, five paired blocks, separate latency,
  peak-memory sampling and retention-grid prediction/resume/replay storage.
- Recorded explicit experiment/budget/seed provenance; rejected timing inventory
  hooks and empty/malformed Windows process lists with regression coverage.
- Updated BENCHMARKS.md, bounded policy, protocol checkpoint, TODO, README,
  HANDOFF and runbook. Implementation is complete; real acceptance remains open.
- Prior independent Standards/Spec review findings were addressed. Final
  synthetic verification is recorded in RESULTS_LOG.md; no real run launched.

Completed artifacts and unrelated changes are preserved. No commit or push.

## 2026-09-13 - Approved desktop guard and native logging recovery

- Preserved user-run attempt01 failure/log; it stopped before learner setup.
- Added approved exact executable-path desktop baseline with PID/path records,
  explicit Overwolf blocking and strict malformed GPU-inventory rejection.
- Fixed native stderr truncation and exit-code capture, tested with real harmless
  stderr-emitting processes. Added read-only workload preflight before RUN.
- Prepared fresh attempt02 with explicit -Attempt selection and overwrite refusal;
  updated policy, TODO, handoff, benchmark guide and absolute-path runbook command.
- Standards review found no issues; Spec malformed-row finding was fixed with
  six regression cases. Final suite recorded in RESULTS_LOG.md.

No real experiment launched, app killed, original artifact changed or push made.

- Recovery verification: 394 tests plus six subtests passed. Final narrow
  Windows-Search-owned WebView exception passed 31 focused regression tests;
  no full-suite rerun after that refinement at the user's immediate-finish request.
  Browser/unverified WebView remains blocked; accepted parent chains are recorded.

## 2026-09-13 - A25 benchmark attempt02 independently accepted

- Verified 251 artifact hashes, 21 pins, 205 checks and all 60 timed-pass ledgers
  and saved predictions. Added bounded audit; marked attempt02 completed.
- HDC throughput was below all five controls in every paired block. No HDC
  throughput advantage established at this condition; report stays provisional.
- Updated TODO/handoff/guide/runbook. Feasibility and final release remain open.
- Saved-file audit only; no experiment, learner loading or artifact alteration.

## 2026-09-13 - Preliminary feasibility and resource gate

- Added measured-input worksheet separating the pilot from conditional full-grid
  scaling and excluded costs. Current free space does not fit the unchanged
  benchmark layout scenario. User storage/runtime envelope required; A3 stays open.
- Updated TODO/handoff/log; read-only measurements, no runs or artifact changes.

## 2026-09-14 - Shared-projection storage test

- Verified71.10GiB free; isolated deduplication test preserved46 checkpoints,
  1,066 arrays/metadata entries and all252 original file hashes.
- Equivalent pilot footprint reduced90.92%; conditional benchmark-only storage
  scenario now17.14GiB. No production format change or all-study capacity claim.
- Recorded durable audit/feasibility/TODO/handoff; retained scratch evidence and
  removed throwaway test code. No training, scoring or original artifact change.

## 2026-09-14 - Approved shared-projection production integration

- Added independently versioned NPZ storage envelopes with verified run-local
  content-addressed projection blobs; legacy standalone saves/loads remain.
- Wired future budget snapshots and benchmark resumable/prediction exports to
  one run-local store; receipt closure includes required blobs, and benchmark
  storage distinguishes per-file bytes from dependency-inclusive bytes.
- Atomic exclusive publication refuses overwrite, handles concurrent blob reuse,
  and normalizes Windows extended-path spelling without waiving root boundaries.
- Synthetic round trips cover all six learners and linear/MLP heads; negative
  tests cover corrupt/missing blobs, receipt omissions/extras, unsafe envelopes,
  failed writes, concurrent publication and portable copied bundles.
- Updated checkpoint/benchmark guides, TODO, feasibility and handoff. Completed
  artifacts remain untouched; no scientific formulas, budgets or timing-boundary
  changes. Production-size acceptance remains a fresh user-launched pilot.
- Final full CPU suite passed434 tests plus6 subtests in133.93s; diff whitespace
  check and attempt03 read-only input plan passed. No real run launched.

## 2026-09-14 - Approved practical shared-desktop benchmark policy

- Preserved incomplete user-run attempt03 after Settings reappeared at block4.
- User-approved exact-path Windows Settings baseline, independent of PID changes;
  ordinary accepted OS activity is logged, not treated as measured interference.
- Existing competitor/malformed-inventory guards, timing and learner semantics
  unchanged. No process termination, automatic retry or retrospective PASS.
- Added12 synthetic regression cases; updated bounded/core protocol, launcher
  messaging, benchmark guide, TODO/handoff and fresh attempt04 manual runbook.
- Full CPU suite446 tests plus6 subtests passed; standards/spec reviews found no
  issues. Read-only workload preflight and attempt04 input plan passed. No run.

## 2026-09-14 - Compact benchmark attempt04 acceptance

- Independently verified user-run PASS:252 artifacts,22 pins,60 timed ledgers,
  46 compact archives sharing one projection,78 resumable states/1,128 array hashes.
- Measured production directory121.87MiB,90.92% smaller than attempt02; no originals
  removed. Conditional benchmark scaling17.14GiB; high-budget/B/all-study costs open.
- HDC slower than all five comparators in all paired blocks; no speed claim.
- Added audit; updated TODO/handoff/guides/runbook/feasibility. Attempt04 complete,
  do not rerun. No real experiment or learner execution by the agent.

## 2026-09-14 - Runtime decision and B400 feasibility launcher

- Recorded no user-imposed runtime cap; manual-only execution, resource guards
  and full-grid/test-release gates remain unchanged.
- Added fixed benchmark-b400.ps1 launcher for frozen-B/seed0/B400/threshold0.10,
  six heads/five paired blocks; preserves completed NCM B400 and A25 artifacts.
- Both launchers share synthetic guard/argument/log-binding coverage:14 targeted
  tests passed. Read-only real-input plan passed; no benchmark launched.
- Added full guarded runbook command and updated feasibility/TODO/handoff/guide.
- Final full CPU suite453 tests plus6 subtests passed in208.14s; reviews found
  no blocking issues. Initial approval-service limitation resolved on retry.

## 2026-09-14 - Recurring background GPU entries warn instead of aborting

- User-requested practical guard fix after B400 preflight blocked on GitHub
  Desktop, Spotify Game Bar WebView and Edge Game Assist; no attempt was created.
- Unknown GPU entries remain unclassified warnings, not automatic baseline
  approval. Known workload exclusions and primary inventory failures still stop.
- Preflight PASS_WITH_WARNINGS exits0; summary/receipt retain transient warning
  boundaries and shared-desktop qualification. No timing/learner changes.
- Exact-app regression reproduced the old failure, then63 targeted checks passed.
  Independent Standards/Spec reviews found no blocking issues; see
  results/BACKGROUND_WARNING_REVIEW_2026-09-14.md for full verification status.
- Updated both launcher messages, benchmark policy/guide, core protocol, TODO,
  runbook and handoff. No process termination, artifact overwrite or experiment run.
- Full CPU suite475 tests plus6 subtests passed in144.61s; live read-only preflight
  PASS_WITH_WARNINGS/exit0 with PhoneExperienceHost retained as unclassified.

## 2026-09-14 - B400 benchmark acceptance and resource observation

- Independently accepted user-run B400 attempt01:251 artifact hashes,22 pins,
  60 timed passes,78 resumable archives/1,182 array hashes verified.
- Retained all205 background-warning boundaries; no isolated-machine claim.
  HDC slower than all five controls in every paired block; label purchases differ.
- Recorded123.95MiB and10.44minute operational duration; added detailed audit,
  updated report/README/TODO/handoff/runbook/guide/feasibility. Broader A3 remains open.
- Marked B400 command completed, not a next launch. No official run, inference,
  learner update, process termination or artifact rewrite by the agent.

## 2026-09-15 - Comparative forgetting grid preparation

- Added a one-cell-at-a-time, user-confirmed B400 capture plus validation launcher.
  It reads registered learners from the project, protects frozen/NCM/seed0, refuses
  existing targets, uses the workload policy, and never batches cells.
- Added post-hoc no-model uptake arithmetic with original/repaired window, terminal
  accuracy and explicit null results. It pins saved validation ledgers while using
  authenticated chronological history for actual examples-seen windows.
- Added measured A3 feasibility, a 35-cell manifest and synthetic acceptance tests.
  No completed artifact, protected evaluator, learner, runner, stream spec or test
  split changed.

## 2026-09-15 - Forgetting-grid provenance follow-up

- Expanded the manifest from 35 to 36 new cells by adding a fresh frozen/NCM/seed0
  recapture under current code provenance; the existing audited cell is untouched.
- Moved future capture/validation outputs to their standard longitudinal result parents,
  retaining path-level no-overwrite protection.
- Required frozen/HDC/seed0 as cell 1 and added a measured-storage addendum gate before
  any remaining cell. Added a regression test for the 36-command manifest and HDC-first order.

## 2026-09-15 - Unattended power-cut-tolerant batch runner

- Added `run-forgetting-grid-all.ps1`; the original single-cell launcher remains unchanged.
- Added receipt-based skip/evaluate/capture routing, fresh-attempt escalation capped at three,
  durable progress JSONL and cell-1 storage addendum writing.

## 2026-09-15 - Comparative forgetting grid completed and analysed

- User completed all 36 grid cells; 72/72 integrity receipts PASS, all attempt01.
- Ran scripts/uptake_repair.py across all 36 evaluations into
  results/forgetting_grid_development/uptake_repair/; original metrics untouched.
- Established: label-matched Macro-F1 advantage for HDC on adapted-B, 3/3 seeds,
  non-overlapping ranges. Not established: reduced forgetting (2/3 seeds, encoder
  dependent). Null: novel-class uptake across all four criteria.
- Corrected the A3 runtime projection; actual grid span was 26.5 minutes.
- No test partition access; no protected file or completed artifact modified.
## 2026-09-15 - Mechanism-ablation preparation (120 cells; prepared only)

- Added `aggregation_prototypes.py`, `ablation_mode`, and
  `results/mechanism_ablation/ABLATION_PLAN.md`: 120 cells across three
  phases (Phase 1: HDC/NCM/RanPAC-style x 5 arms x 3 seeds = 45; Phase 2:
  Sign-NCM/Median-NCM x 5 arms x 3 seeds = 30; Phase 3: HDC at D in {1000,
  2000, 10000} x 5 arms x 3 seeds = 45). Prepared only; no official run was
  launched by the agent at this point.
- Added a draft locked test-release protocol for the sealed trusted-test
  partition, `docs/TODO/HDC_TEST_RELEASE_PROTOCOL_2026-09-15.md` - status
  DRAFT, not approved; defines gate A2 (referenced but never defined
  elsewhere in `HANDOFF.md`).

## 2026-09-16 - Mechanism ablation unblocked and run to completion (120/120 cells)

- Merged from `PENDING_CHECKPOINT_runner.md`
  (`results/mechanism_ablation/`) and `PENDING_CHECKPOINT_aggregator.md`
  (`results/forgetting_grid_development/`), both left in place per
  instruction (not deleted by this entry).
- Fixed three defects blocking the ablation, all verified against real
  artifacts: (1) `evaluate_snapshots.py` now threads
  `ablation_mode=bool(manifest.get("ablation_mode", False))` into
  `capture._prepare(...)` (previously always `False`, causing `ValueError:
  unknown learner` on every ablation evaluation); reproduced before the fix,
  confirmed `status: PASS` / `snapshots_scored: 20` after, against a real
  `sign_ncm`/`b_frozen`/seed-0 PASS capture. Not in the capture's pinned
  `code_hashes`, so the archived 36-cell grid is unaffected. (2) Added
  `--hypervector-dim` to `budget_experiment.py` (requires `--ablation-mode`),
  threaded through `experiment_a.py` and `stream_runner.py`, locked to
  `{1000, 2000, 10000}`, `hdc` learner only, recorded as
  `ablation_hypervector_dim` only in ablation mode; `policy_sha256` and
  canonical manifests unaffected. (3) `aggregate_grid.py` rejected Phase-3
  directory names; fixed in commit `126ea06`.
- Added `scripts/run-mechanism-ablation-all.ps1` (120-cell batch launcher,
  receipt-based completion, 3-attempt escalation, per-cell 8 GiB and
  workload-preflight guards, single `RUN ABLATION` confirmation) and three
  new test files (`test_aggregation_prototypes.py`,
  `test_hypervector_dim_ablation.py`, `test_mechanism_ablation_batch_runner.py`).
- User completed all 120 cells; 240/240 capture+validation integrity receipts
  PASS, all attempt01, 400 labels consumed in every cell for every
  label-matched learner; run order Phase 2 -> Phase 1 -> Phase 3.
- Established via the locked decision rule: Sign-NCM's `3_00_0` F1 on adapted
  ViT-B (0.2918) falls inside the pre-declared indeterminate band
  (0.10-0.30); Median-NCM clears 0.30 (0.4080); Sign-NCM clears 0.30 on
  adapted ViT-L (0.5094); nothing fell below 0.10. Phase 1 reproduces the
  36-cell grid exactly on shared arms. Phase 3 shows HDC does not collapse at
  any tested D but its macro-F1 rises with D (D=10000 reproduces Phase-1
  exactly). Adapted ViT-L NCM does not collapse and beats HDC on macro-F1.
  Best macro-F1 across all 120 cells: Median-NCM on frozen ViT-B, 0.7420.
- Conclusion recorded: collapse resistance tracks bounded-influence
  aggregation, not the hyperdimensional projection; the projection still
  improves HDC's own accuracy with dimension. Labelled note only, not a
  decision: this points toward a possible reframing around bounded-influence
  aggregation. No gate closed, no thesis-title change, no test-split access.
- Full test suite at last verified run: 559 passed, 6 subtests passed.
- No capture, evaluation, training, or GPU work was launched by this
  record-refresh pass; no code, test, or result artifact was modified by it.
