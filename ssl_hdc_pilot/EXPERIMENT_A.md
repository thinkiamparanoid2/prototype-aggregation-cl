# Experiment A — shared-label development implementation

User approved these settings for implementation on 2026-09-11:
additional budgets **0, 25, 50, 100, 200, 400**, identical label-blind nested
selections and delivery times across all six learners and both ViT-B arms,
and novel registration only when a scheduled label is purchased.
This is not approval of held-out access, the full 216-cell grid, or A1-A4 freeze.

## Implemented scope

`scripts/experiment_a.py` builds schedules and executes one explicitly selected
development cell. It defaults to a read-only plan: input/schedule preparation,
without learner construction, updates, predictions, or output files.
Execution requires `--execute --preflight-confirmed` and a fresh output directory.
Every experiment remains manually user-launched.

The schedule implements the draft's deterministic recipe:

1. Take the already registered chronological stream IDs, not their labels.
2. SHA-256 hash UTF-8 `hdc-shared-v1|20260909|<seed>|<image_id>`.
3. Sort by digest, then literal ID; select the first B IDs.
4. Deliver labels at those IDs' original chronological stream positions.

IDs are nonempty unique strings, preserved literally: no integer conversion,
Unicode normalization or silent trimming. Edge whitespace is rejected. The
schedule contains full stream IDs, selected step/ID pairs, a ranking hash and
a canonical JSON schedule hash. The rank is independent of budget, learner,
embedding values, confidence and class labels; lower budgets nest in higher ones.
Small budgets may miss a novel class; the schedule is not repaired using labels.

The new opt-in branch in `stream_runner._run_one` purchases exactly scheduled
events and bypasses uncertainty/oracle acquisition. Historical callers retain
their prior branch and output schema. Learner update formulas are unchanged.
Each cell starts from scratch, receives the same bootstrap, traverses the whole
stream, and only then saves the final state. B=0 gives no stream updates or
novel registrations. Neural replay/EWC retain their registered update rules.

For real inputs, the bootstrap has 2,420 labels and the stream 976 events.
Bootstrap labels are outside B. Report `2420 + consumed` as labels supplied to
the continual learner, not the entire project's labeling requirement. Encoder
selection previously used trusted training/validation labels; this budget study
is conditional on those already selected representations.

## Data and provenance guards

Only canonical approved training labels and the two existing ViT-B embedding
arms are accepted for non-smoke use. Non-training paths are rejected before
opening input files. Canonical base-spec bytes are pinned; training-only
holdback verification does not reopen validation/test partitions. Only training
IDs enter learning and the final diagnostic prediction pass.

Inputs are hashed before parsing/loading and rechecked after preparation;
execution rechecks them and code at completion. Source files and completed
reconstructions are not overwritten. The effective acquisition policy is stored
separately from the unchanged base-spec bytes/hash, not misrepresented as the
original historical policy. The current stream-runner file hash changes because
of the new branch; earlier reconstruction manifests retain their original code
hashes and remain historical evidence. Do not rewrite them or rerun those cells.

Fresh per-cell outputs: `manifest.json`, `base_stream_spec.json`,
`shared_label_schedule.json`, `final_state.npz`, `history.csv`, `summary.json`,
`final_training_predictions.json`, and `integrity_report.json`. Manifests record
code/Git provenance, input hashes, environment, policy, schedule hash and the
unfrozen draft protocol hash. States are learner-resumable, not stream resume.
NPZ states remain Git-ignored and need separate backup.

`integrity_report.json` PASS means observed label events equal the schedule,
exact additional-label accounting, no denied scheduled purchases or unpurchased
registration, and stable input/code hashes. It is NOT a fidelity comparison to
historical runs, predictive success criterion, or test-release approval.
Exceptions/missing reports/FAIL block acceptance; preserve partial outputs.

## Still pending

TODO 2A.3 fixed validation evaluation, longitudinal snapshots, fixed-population
forgetting/uptake, benchmark/storage instrumentation, and final grid/release
remain unfinished. Current summaries report only prequential and final
bootstrap-plus-stream training-pool diagnostics. Their rolling metrics and host
update timers are not the proposed primary thesis endpoints.

Completed user-run checkpoint: **frozen-B / NCM / seed 0 / B=25** training-side
development cell passed: 25 stream labels, 2,445 supplied learner labels and
976 events. Independent audit verified all seven artifact hashes and 3,396
unique training-prediction IDs. Do not rerun it. Next: evaluate its saved state
on fixed validation using [VALIDATION_EVALUATION.md](VALIDATION_EVALUATION.md)
and the new manual block in RUN_COMMANDS.md. This is not the 216-cell study or
a frozen scientific result.

## Review

### Standards

One provenance gap was found and fixed: hash inputs before parsing and recheck
after preparation. A filesystem-boundary regression covers replacement after
CSV parsing. Recheck found no blockers. Shared training-validation helper
duplication is a maintenance observation, not an unresolved correctness issue.

### Spec

No blockers found in the approved shared-label development scope. Fixed
validation evaluation and longitudinal analysis remain explicitly deferred.

Review summary: Standards 0 remaining blockers; Spec 0 blockers in approved scope.
Full verification: `scripts/hdc.ps1 -Action test` passed 155 tests and six
subtests in 54.80 seconds, including 41 new synthetic Experiment-A tests.
The manual PowerShell block passed parser-only syntax validation without
execution; `git diff --check` passed after documentation updates.
No official run was launched by the agent.
