# Continual-head benchmark development

## Completed feasibility pilot: Experiment B, allowance400

COMPLETE2026-09-14: user-run attempt01 independently accepted; do not rerun the
historical launcher command below. See
[B400 audit](results/B400_BENCHMARK_AUDIT_2026-09-14.md):251 hashes/22 pins,
all60 timed passes verified,123.95MiB,about10.44minutes. All205 checks retain
background warnings. HDC slower than all five controls in every paired block;
actual purchases differ by learner. Whole-study feasibility remains open.

The user removed the runtime cap ("as long as is needed"). Prepared one fixed
frozen-B/seed0/Experiment-B/allowance400/threshold0.10 condition covering all six
learners and five paired timing blocks. This uses the existing benchmark engine;
it does not repeat or overwrite the completed NCM B400 capture/validation.
Method-specific acquisition can consume fewer than400 labels. One such condition
is informative for feasibility, not proof of a worst-case storage/runtime bound.

```powershell
rtk proxy powershell -NoProfile -File "G:\FOR THESIS\HDC\scripts\benchmark-b400.ps1" -Attempt 1
```

The launcher uses current G: artifacts, an8GiB prelaunch headroom check, the
background-warning shared-desktop policy, complete native logs and explicit
RUN BENCH B400 confirmation. Existing attempts are refused; no auto-retry/resume.
Output:results/benchmark_development/2026-09-14_b_frozen_s0_b400_t010_attempt01.
Add -PlanOnly for a read-only input plan (no learner constructed, no workload
audit or run). The engine/measurement/learner settings are unchanged. Full guarded
paste block is in RUN_COMMANDS.md; no agent-run experiment or full-grid approval.

## Completed A25 storage acceptance

Latest status2026-09-14: attempt04 PASSED and was independently audited. The
production shared-projection layout totals121.87MiB (90.9245% smaller than the
attempt02 directory). HDC still has lower full-stream throughput than every
comparator in every paired block. See
[compact benchmark audit](results/A25_COMPACT_BENCHMARK_AUDIT_2026-09-14.md).
Attempt04 is complete; do not execute its command again.

Shared-desktop amendment approved2026-09-14: Windows Settings is now accepted
and logged at its identified system path even if it restarts. Existing accepted
desktop activity need not be absent. Games, recording/overlays, browser activity,
other training workers still stop the run. Under the later background-warning
amendment, unclassified GPU entries warn instead of aborting. These
inventories do not measure interference; all five timing blocks and variability
remain required, with no isolated-machine claim. See BENCHMARK_DEVELOPMENT_PLAN.md.

Attempt03 stopped after the HDC latency pass in block4 when Settings reappeared
under the former rule. Preserve the incomplete attempt and its log; no resume,
retroactive PASS, or pooling its partial blocks into the next attempt.

Update (2026-09-14): future attempts use run-local shared projection blobs for
HDC/RanPAC resumable and prediction exports. Copy the whole attempt directory,
including projection_blobs. Per-file NPZ sizes exclude shared blobs; dependency-
closure fields include them, and whole-run storage counts each blob once. Live
tensor/peak memory and timing boundaries are unchanged. See LEARNER_CHECKPOINTS.md.
The production layout is now accepted at this bounded A25 condition. Its observed
90.92% directory reduction agrees with the earlier lossless prototype; broader
high-budget/B storage and runtime still need feasibility estimates.

Completed attempt04 command (historical provenance; DO NOT RERUN):

```powershell
rtk proxy powershell -NoProfile -File "G:\FOR THESIS\HDC\scripts\benchmark-a25.ps1" -Attempt 4
```

Attempt04 is separate from completed attempt02 and failed attempt03. The retained
launcher prefix is `2026-09-13_a_frozen_s0_b25_attempt04`; manifests/logs record the actual launch.
This is one development condition with real updates, not permission for a grid.

Current status (2026-09-13): attempt02 completed and independently audited.
Do not rerun the command below; it is completed-run provenance. See
[A25 benchmark audit](results/A25_BENCHMARK_AUDIT_2026-09-13.md).
HDC throughput was lower than all five comparators in all five paired blocks.
First-pilot acceptance is complete; full-grid feasibility and claims remain open.

The user approved [the bounded policy](BENCHMARK_DEVELOPMENT_PLAN.md) on
2026-09-13. Implementation is separate from existing scientific learner/runner
code. No completed A25/B400 or historical artifact is rewritten. No real
benchmark has been launched by the agent.

## First manual pilot

ONE frozen-B / seed-0 / Experiment-A / allowance25 condition, all six learners.
This is not the full grid. Its purpose is acceptance of the timing/fidelity and
storage workflow, not selection of favorable outcomes. It constructs new
references and performs real updates, so execution is user-owned.

```powershell
rtk proxy powershell -NoProfile -File "G:\FOR THESIS\HDC\scripts\benchmark-a25.ps1" -Attempt 2
```

The script runs a read-only plan, checks 8 GiB operational disk headroom, records
environment/GPU/process preflight, checks blockers, then asks for `RUN BENCH A25`.
Avoid deliberate competing workloads. Explicit splatted logging parameters avoid detached
FilePath prompts. All outputs go to the fresh
`results/benchmark_development/2026-09-13_a_frozen_s0_b25_attempt02` directory;
the adjacent log has a unique name. Existing attempts are refused, not overwritten
or resumed. Optional `-PlanOnly` never constructs a learner or performs timing.
After completion, tell the agent done; it reads saved files directly.

Attempt01 is failed preflight provenance, not a completed timing run. It contains
only failure.json; the original log is preserved. The user approved the exact-
path desktop-baseline amendment after that failure. Do not delete/reuse attempt01.
The launcher defaults to attempt02, accepts an explicit positive -Attempt number,
and never auto-increments or retries. Unready workload preflight stops before
confirmation/attempt creation. Native stdout/stderr are drained into the log,
then the real exit code is checked; warnings do not truncate later tracebacks.

The 8-GiB guard is operational headroom, not a measured pilot or 972-condition
storage estimate. Runtime is not yet measured; no speed estimate is promised.

## Measurement design

- Untimed references use the unchanged existing runner and approved A/B input
  preparation, including original learner formulas and request semantics.
- Each learner has an initial pre-bootstrap checkpoint and a bootstrap checkpoint,
  including RNG. One full-stream warm-up runs on a disposable restored copy.
- Five paired blocks rotate the declared six-learner order. Each block runs a
  full-stream throughput pass and a separate per-update latency pass per learner.
  Bootstrap latency is separately measured from the initial state, excluding
  construction/load time. Every repetition restores the same starting state/RNG.
- Throughput is stream events / synchronized full-stream elapsed seconds. Host
  embeddings enter the original predict/update methods, so their transfers are
  included. Minimal decision trace bookkeeping is included consistently. Rolling
  metrics, output encoding, snapshots, final scoring and other file I/O are outside
  timing. There are no added per-event timing synchronization calls in throughput.
  Inherent synchronization inside original learner methods is not removed.
- Latency samples synchronize around updates; zero purchases give N/A, not zero
  latency or infinite rate. Update throughput is never substituted for full-stream
  throughput. Report five raw blocks, medians/ranges and within-block HDC/comparator
  ratios; these are technical repetitions, not independent streams.
- Every pass must match untimed event decisions, registrations and final training
  predictions. Discrete values match exactly, floating scores use absolute tolerance
  1e-6. Bootstrap state/RNG fingerprints must also match before timing repetitions.

## Memory and storage boundaries

A distinct peak-memory pass samples the process working set with a requested
5-ms polling interval and records CUDA allocator peaks. Polling/query overhead
exists only in that pass; sampled process peaks can miss brief transients.
The process includes shared embeddings, interpreter and native runtime; CUDA
counters do not include every driver allocation. Unused CUDA cache is cleared
before restoring the peak-pass model and again after restoration temporaries
are freed. Initial and peak allocated/reserved bytes are recorded. Do not infer
per-head native/Python object overhead from these shared-process readings.

A second untimed inventory pass records bootstrap, each fixed retention time
and final payload, including the sampled maximum. Prediction tensors/registry,
extra resumable metadata/tensors, global RNG and replay occupancy/seen/label/
embedding payload are separately reported. Serialized metadata bytes are not
runtime object sizes. Inventory hooks are prohibited in throughput/latency modes.

At each inventory time, both full-resume NPZ and prediction-only payload NPZ
exports are measured. Prediction-only exports are cost artifacts, not inputs to
the existing resumable checkpoint loader. No new deployment interface is claimed.
Logical file bytes, actual filesystem allocation (hard-link identities counted
once), and hypothetical identical-content deduplication are distinct quantities.
No actual deduplication or destructive cleanup is performed. The storage snapshot
excludes its subsequently written summary and receipts, as its fields state.

## Guardrails and provenance

Real execution requires canonical training inputs, CUDA, explicit preflight and
a fresh output directory. Torch CPU intra/inter-op threads are one; the launcher
also fixes OMP/MKL/OpenBLAS thread environment settings. Hardware inventory includes
CPU model, physical RAM, GPU/driver and dtype. Input/code/protocol/policy hashes
are pinned and rechecked before final PASS. All raw blocks and failures remain.

Read-only process/GPU checks bracket passes. They reject listed competing browser,
game, recording/overlay (including Overwolf) and Python workloads. Other
GPU-listed processes produce unclassified warnings, not automatic aborts.
The approved exact-path Windows desktop/Terminal/VS Code
baseline is recorded with PID/path; it is not a blanket C+G exception. Windows
Settings is explicitly included by the September14 amendment. Ordinary accepted
OS activity may recur. Phone Link and Sticky Notes remain outside that identified
baseline. See the dated policy amendment for the exact scope.
Exception: the pinned WebView version is accepted only through a verified,
recorded parent chain to the pinned Windows Search host. Browser-owned or
unverified WebView is unclassified, not promoted to baseline; do not kill Windows Search.
An unreadable primary inventory fails closed; optional ownership failures warn.
The waiting Windows venv Python parent
is excluded only from the Python-name guard, not from GPU-compute detection.
Boundary checks cannot guarantee absence of transient/unlisted interference or
zero shared-desktop overhead;
human preflight remains necessary, but perfect OS inactivity is not required.
Nothing is killed automatically.

Preflight reports PASS_WITH_WARNINGS with exit0 when only background warnings
occur. Every boundary retains PID/path entries. Both summary.json and
integrity_report.json contain environment_qualification and all warning boundaries,
including transient ones. An integrity PASS is not a clean-timing certificate.
Warning-bearing results are shared-desktop observations with unclassified activity;
keep all five blocks and their variability, not just warning-free repetitions.

## Verification and review

Synthetic tests cover all six learners, A/B zero/limited budgets, original-runner
decision/prediction fidelity, exact timing synchronization boundaries, state/RNG
restore, five paired blocks, memory/storage distinctions, tampering failure and
six fully mocked launcher paths. Read-only real-input planning passed without
constructing a learner. Final full-suite results are recorded in RESULTS_LOG.md.
Final wrapper result: 371 tests plus six subtests passed in 119.72 seconds,
including 47 benchmark tests. These use synthetic inputs with CUDA disabled;
they do not validate real CUDA timing or establish a performance advantage.
That result predates the desktop-guard recovery patch; see the latest
RESULTS_LOG.md checkpoint for its final regression/full-suite results.

### Standards

Initial review found two nonblocking observations: allocator carry-over attribution
and repeated literal block counts. Both were addressed; follow-up reports zero
unresolved findings. Launcher/allocation APIs were also reviewed read-only.

### Spec

Three findings were addressed: intermediate memory checkpoint coverage;
prediction/full-resume and RNG/registry accounting; RAM/dtype provenance.
Follow-up reports no remaining static-review blockers.

The first real pilot must still pass before benchmark acceptance or feasibility
extrapolation. Comparative predictive/retention grids, paired image uncertainty,
resource ceilings, appendix/A1-A4 and final test release remain open. The overall
scientific report remains PROVISIONAL. Changes remain uncommitted in the mixed
worktree; no unrelated work is staged.
