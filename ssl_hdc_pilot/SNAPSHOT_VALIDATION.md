# Fixed-population A/B snapshot validation - development only

## Completion checkpoint - 2026-09-13

B400 validation is COMPLETED user-run PASS. All 22 output hashes and 48 input
pins matched; all metrics reconstructed from 20 x 1,132 saved predictions.
See [audit and results](results/B400_VALIDATION_AUDIT_2026-09-12.md). Do not rerun
the historical launch below. Next input: [benchmark policy](BENCHMARK_DEVELOPMENT_PLAN.md).

## Completed source acceptance

The user completed B400/NCM/frozen-B/seed-0/threshold .10 on 2026-09-12 in
`results/budget_longitudinal_development/2026-09-12_b_frozen_ncm_s0_b400_t010_attempt01`.
Independent read-only audit verified all 27 report-listed artifact hashes,
unchanged recorded code/input hashes, 976 history/event rows and 3,396 unique
training prediction IDs. Actual purchases: 400 of 450 potential requests;
50 denied, zero unused allowance, 2,820 supplied learner labels with bootstrap.
All four novel classes registered. There are 20 physical snapshots, including
every approved retention checkpoint. These are capture/accounting facts, not
validation performance or a benchmark result.

The copied shell pipeline broke before `-FilePath $bLog -Append`. Preflight and
confirmation are preserved in `preflight_run_6c7795364b0b4fe3b4471dc604ce66ef.log`;
training output is in `b400_recovery_01.log` beside the source attempt. A later
standalone -FilePath command failed after the PASS report. No artifact rerun or
rewrite is required. Future launches use a script with splatted logging parameters.

## Implemented evaluator

`scripts/evaluate_snapshots.py` accepts only the new A/B snapshot format. It does
not weaken the legacy A25 evaluator or modify any learner/runner/metric file.
Preparation rejects noncanonical real validation paths before reading inputs,
checks exact artifact/code provenance and paths, reconstructs the approved
training stream and policy without constructing a learner, and audits event
identities, single charging, paid registration, exhaustion and snapshot times.
The reconstruction here is input ordering only, not learner replay/training.

The evaluator uses the same 1,132 validation IDs and 19-class universe at every
state. Only validation embeddings reach the scorer even though the stored cache
contains all partitions. Each checkpoint's full context, dimensions and paid
class registry must match the capture ledger. Predictions preserve learner/RNG
state, reuse common classification/stage/ECE semantics, and are saved per state.
No updates, calibration fitting, model selection or test scoring occurs.

Forgetting uses only the ten approved retention times and the 15 initial classes;
extra uptake states do not affect peaks. Novel uptake uses each class's actual
purchased-label milestones plus final, with .80-relative and .80-absolute targets.
Missing, never-registered/reached, undefined and later-drop cases stay explicit.
No AULC or paired comparison is inferred from one B400 cell.

Outputs in a fresh, separate directory: `predictions/step_*.json`, `metrics.json`
(fixed metrics, forgetting and uptake), `manifest.json`, `integrity_report.json`.
Input/source/code hashes are checked again before PASS. Existing directories and
outputs inside the source are refused. Missing/failed receipts are not acceptance.

## Manual launch

From the repository terminal:

```powershell
rtk proxy powershell -NoProfile -File .\scripts\validate-b400-snapshots.ps1
```

The script runs a read-only input plan, records environment/GPU/process preflight,
asks for `RUN SNAPSHOT VAL`, and scores all 20 saved states. Logs have an explicit
unique filename; no manual FilePath prompt is intended. It writes only to the
new `results/longitudinal_validation_development/2026-09-12_b_frozen_ncm_s0_b400_t010_attempt01`
attempt and its adjacent log. On completion tell the agent "done"; the agent
reads saved files directly. No need to paste metrics or rerun B400 training.
For preparation only, add `-PlanOnly`; that option never constructs a learner.

The real-input plan passed for 20 states x 1,132 samples with no learner loading
or scoring. Real validation was subsequently completed by the user, not the agent.

## Review and verification

Standards: independent review found no hard violations or blocking smells.
Spec: one boundary finding (all-partition embeddings reaching the scorer) was
fixed by filtering to validation IDs and covered by a regression test. The
review otherwise found approved snapshot/metric semantics and guards aligned.

Synthetic verification covers all six learners, A/B zero/full budgets, every
saved state, fixed IDs/classes, source immutability, tampering, missing/mislabelled
validation inputs, forbidden partitions, no-model planning and non-overwrite.
Five launcher tests mock every external command and cover explicit log binding,
success, plan-only, cancellation, failed native command and existing output.
No real experiment is executed by those tests. Full wrapper: **324 tests plus
six subtests passed in 98.46 seconds**. No commit or staging mixes these changes
into unrelated work.

After manual validation acceptance, comparative grid coverage, timing/storage
instrumentation, paired uncertainty, feasibility and final A1-A4/test-release
decisions remain open. The scientific report remains PROVISIONAL.
