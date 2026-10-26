# A/B acquisition and longitudinal snapshot development

Implemented 2026-09-12 following explicit bounded approval on 2026-09-11.
Entry point: `scripts/budget_experiment.py`. Default CLI action validates training
inputs and prints a plan without constructing a learner. Execution is manual,
requires preflight confirmation and refuses an existing output directory.

## Scientific scope

- A: additional budgets 0/25/50/100/200/400, nested label-blind schedules shared
  across learners/arms, registration only on purchased scheduled examples.
- B: same allowances; thresholds .05/.10/.20; uncertainty plus disclosed
  ground-truth-assisted novel requests. Potential requests include denied events;
  a double-trigger purchase is charged once. Underspending is permitted. No
  purchase, registration or update follows exhaustion. This is not a deployable
  novelty detector.
- Both use the existing strict training-input preparation. For B that helper
  computes an unused A schedule, but it is neither applied nor persisted.
  Bootstrap supervision is 2,420 plus actual additional purchases, conditional
  on preselected representations, not the entire project's label requirement.
- Retention states: processed-event counts 0/195/196/390/391/585/586/780/781/976.
  Bootstrap is t=0 and zero-based history event i is observed at t=i+1.
- Novel states: immediately after 1/2/5/10/20/50/100 purchased labels per class
  where available, plus final. Development uptake targets are .80 of final
  recall and absolute recall .80; no target was chosen from the A25 result.

The internal observer saves one physical state per selected event count, with
overlapping reasons combined. Final state is `final_state.npz`, not a duplicate
snapshot. Only the declared retention times will enter forgetting peaks.

## Outputs and integrity

Each fresh attempt contains a manifest with input/code/policy provenance, an
unchanged copy of the base spec, event/accounting ledger, snapshot index, NPZ
states, history, final training predictions, summary and integrity report.
A additionally stores its shared schedule. Snapshot index records capture-time
hashes, purchase counts and reasons. Integrity checks verify accounting,
exhaustion, snapshot schedule/counts, capture-to-finish snapshot hashes and
unchanged inputs/code/protocol. A missing or FAIL report is not acceptance.

PASS means training acquisition/snapshot integrity, NOT held-out prediction,
forgetting, uptake or throughput success. Serialization/synchronization affects
execution cost; this run is not a benchmark. The canonical spec, common metrics
and learner update formulas are unchanged.

## Verification and review

Full local test wrapper: **281 tests plus six subtests passed in 89.66 seconds**.
New synthetic checks cover all six learners, A/B zero and exhausted budgets,
B underspending, double-trigger charging, exact history/metric agreement with
an unserialized reference, and saved state/RNG equality at every capture.
They also cover preflight, non-overwrite, forbidden split, read-only planning
and post-capture corruption rejection. No real learner run was performed.

The debugging workflow identified a test-fixture threshold issue and unequal
caller RNG starting states between compared passes. Tests now restore identical
initial RNG; no scientific threshold guard or learner algorithm was relaxed.

### Standards

Completed independent review found no hard documented violation. Manual launch,
non-overwrite, canonical inputs, provenance and metric semantics are preserved.
Two optional maintainability findings: replace the eight-position preparation
tuple with a named result, and extract the combined execution function's post-run
audit into a pure helper. Neither blocks bounded manual development acceptance.

### Spec

Review reported no blocking finding in the approved bounded scope and identified
stale approval wording, now reconciled. Snapshot timing, single charging,
exhaustion, A schedules and passive state/RNG checks align with the approval.
Real evaluation/curve integration remains intentionally unfinished.

Review summary: Standards two optional findings, no hard violation; Spec stale
documentation resolved, no remaining blocking finding. Runbook syntax and
`git diff --check` both passed; the runbook was parsed, not executed.

## Current boundary and next manual cell

Update 2026-09-12: the B400 cell below is now COMPLETED and independently
verified: 27 hashes, 976 events, 400 purchases from 450 requests, 50 denials,
all four novel registrations and 20 snapshots. The split preflight/recovery
logs are preserved. Do not rerun the training command. Snapshot validation is
also completed and verified; see [audit](results/B400_VALIDATION_AUDIT_2026-09-12.md).
Next input is approval of [benchmark development policy](BENCHMARK_DEVELOPMENT_PLAN.md).

The real-input plan passed for frozen-B/NCM/seed-0, B=400, threshold .10.
This was the first proposed manual development acceptance run, selected to
exercise acquisition and novel-state capture, not from predictive rankings.
Use the final block in RUN_COMMANDS.md, fresh directory
`results/budget_longitudinal_development/2026-09-12_b_frozen_ncm_s0_b400_t010_attempt01`.
Expect 976 events, at most 400 additional labels, ten retention states plus
available uptake states (overlapping states deduplicated).

The legacy `evaluate_validation.py` accepts the earlier A final-state format,
NOT this new A/B snapshot artifact format. Separate strict validation integration
exists in `evaluate_snapshots.py`; first B400 real curves are now verified.
Pure metric arithmetic exists in
`longitudinal_metrics.py`. Do not run the old A25 evaluation block on this output.

The observer changes the runner's code hash. Preserve original historical/A25
manifests and completed results; never rewrite provenance or waive compatibility
guards. Old completed blocks are not restart instructions. No full-grid, test
release, benchmark policy freeze or A1-A4 decision is implied. Changes remain
uncommitted in the preexisting mixed worktree.
