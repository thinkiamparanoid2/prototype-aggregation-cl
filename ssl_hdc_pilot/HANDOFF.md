# SSL-HDC Pilot Handoff

Last updated: 2026-09-16.

## Latest checkpoint: Mechanism ablation COMPLETE (120/120 cells) - collapse resistance traced to bounded-influence aggregation, not the hyperdimensional projection

User-run mechanism ablation completed all 120 planned cells (Experiment B,
threshold 0.10, budget 400), run order Phase 2 -> Phase 1 -> Phase 3. 240/240
capture+validation integrity receipts PASS, all attempt01, 400 labels consumed
in every cell for every label-matched learner. Source: `results/mechanism_ablation/
aggregate_all_120/cells.csv`, `summary.json`, `PROVENANCE.md`.

**Phase 1** (HDC, NCM, RanPAC-style x five arms) exactly reproduces the archived
36-cell grid on the shared frozen-B/adapted-B arms: frozen ViT-B HDC 0.7220 / NCM
0.7313 / RanPAC-style 0.7406; adapted ViT-B HDC 0.6001 / NCM 0.5275 / RanPAC-style
0.5243 (median macro-F1 over seeds 0-2).

**Phase 2** (Sign-NCM, Median-NCM x five arms) is the decisive phase: it asks
whether an aggregation rule without any hyperdimensional projection reproduces
HDC's collapse resistance on adapted ViT-B. Adapted ViT-B medians over seeds 0-2
(macro-F1 / per-class F1 `3_00_0` / per-class F1 `2_00_0`): HDC 0.6001/0.4238/0.7407;
Median-NCM 0.5925/0.4080/0.7407; Sign-NCM 0.5350/0.2918/0.7407; NCM 0.5275/0.0168/
0.0625; RanPAC-style 0.5243/0.0168/0.0588.

Under the pre-declared locked decision rule (indeterminate band 0.10-0.30 on
`3_00_0`, adapted ViT-B): Sign-NCM scores 0.2918 - **inside the indeterminate
band**, recorded as indeterminate for that cell, not as a pass or a collapse.
Median-NCM clears the 0.30 threshold (0.4080). Sign-NCM does clear 0.30 on
adapted ViT-L (0.5094). Nothing tested fell below the 0.10 floor. Taken together,
the result points toward collapse resistance coming from bounded-influence
aggregation (median/sign accumulation resists a few damaged classes dominating
the prototype), not from the hyperdimensional projection itself. Phase 3 (next
paragraph) shows the projection is not inert - it still improves HDC's own
accuracy with dimension - but dimension is not what prevents the adapted-ViT-B
collapse.

Adapted ViT-L is a second, independent check: NCM does **not** collapse there
(`3_00_0` = 0.5521) and NCM's macro-F1 (0.6579) beats HDC's (0.6308) on that arm.
The adapted-ViT-B collapse in NCM/RanPAC-style is therefore specific to that one
encoder arm, not a general prototype-learner failure mode.

**Phase 3** (HDC at hypervector dimension D in {1000, 2000, 10000} x five arms):
adapted ViT-B, macro-F1 / `3_00_0` / `2_00_0`: D=1000 0.5374/0.3505/0.7692; D=2000
0.5720/0.3933/0.7407; D=10000 0.6001/0.4238/0.7407. HDC does not collapse at any
tested D, but its macro-F1 rises with D. D=10000 reproduces the Phase-1 HDC
numbers exactly on all five arms (it is the canonical stream-spec dimension).
**Do not read this as "the projection contributes nothing"** - dimension clearly
helps HDC's own accuracy; what dimension does not supply is the collapse
resistance itself, which Phase 2 isolates to the aggregation rule.

Best macro-F1 across all 120 cells: **Median-NCM on frozen ViT-B, 0.7420**
(above every HDC cell, including frozen-B HDC's own 0.7220). Median-NCM's
implementation stores every confirmed example (an exact running median); its
accuracy figures above are valid, but **no throughput or memory figure may be
claimed for it** - that cost was never measured.

### Defects found and fixed during ablation unblocking (2026-09-15/16)

Three defects blocked the ablation from running at all; all three are now
**fixed**, verified against real artifacts, not open:

1. `evaluate_snapshots.py` did not thread `ablation_mode` into `capture._prepare`
   (line ~167), so it always hit the canonical (non-ablation) gates and any
   ablation capture evaluation failed with `ValueError: unknown learner`.
   Reproduced against a real PASS capture (`sign_ncm`/`b_frozen`/seed 0) before
   the fix. Fixed with a one-line change reading
   `ablation_mode=bool(manifest.get("ablation_mode", False))` back from the
   capture's own authenticated manifest. After the fix, `evaluate_snapshots
   --execute` against the same capture produced `status: PASS`,
   `snapshots_scored: 20`. Not part of the capture's pinned `code_hashes`, so
   the archived 36-cell grid is unaffected.
2. No CLI path existed for the Phase-3 hypervector-dimension sweep. Fixed by
   adding `--hypervector-dim` to `budget_experiment.py` (requires
   `--ablation-mode`, `parser.error` otherwise), threaded through
   `experiment_a.py` and `stream_runner.py`, locked to `{1000, 2000, 10000}`,
   applied to the `hdc` learner only. Recorded as `ablation_hypervector_dim` in
   `manifest.json`/`integrity_report.json` only when `ablation_mode` is true -
   never in `policy` (so `policy_sha256` is unaffected by canonical runs) and
   never added to the checkpoint-context whitelist.
3. `aggregate_grid.py` rejected Phase-3 directory names (e.g.
   `p3_b_adapted_hdc_d10000`). Fixed in commit `126ea06` ("aggregate_grid:
   accept Phase-3 hypervector-dimension cell names").

Full test suite at last verified run: **559 passed, 6 subtests passed**.

A draft release protocol for the sealed trusted-test partition now exists at
`docs/TODO/HDC_TEST_RELEASE_PROTOCOL_2026-09-15.md` - status **DRAFT, not
approved**. It proposes closing gate A2 (see next paragraph) but requires
explicit user and supervisor sign-off first; `trusted_splits/test.csv` remains
sealed and unopened. This checkpoint does not approve it and does not close
any gate.

Gate A2 - the test-release gate this file refers to as open throughout - is
defined in `docs/TODO/HDC_CORE_EVALUATION_PROTOCOL_DRAFT_2026-09-09.md:88-90`.
This is a pointer only; no existing gate wording elsewhere in this file is
changed or closed by adding it.

**Note on framing (labelled, not a decision):** this ablation's result -
collapse resistance tracking the aggregation rule rather than the
hyperdimensional projection - points toward a possible reframing of the
thesis around bounded-influence aggregation rather than around HDC/hyper-
dimensional computing specifically. This is not a decision; closing any gate
or changing the thesis framing is the user's and supervisor's call, not
made here.

Supervisor briefing updated: https://claude.ai/artifact/36wbcTZRNCayj22t6M619q
(Version 4).

All standing qualifiers carry forward unchanged: `trusted_val` is a
development split and this remains development-only evidence, not held-out
evaluation; RanPAC-style is an ablation of the aggregation rule, not
published RanPAC; naive/EWC/replay consumed 17-79 labels against the
prototype heads' fixed 400 in every cell and must never share an unqualified
comparison table with hdc/ncm/ranpac-style/sign_ncm/median_ncm; three seeds
support a consistency statement, not a significance test. Report remains
PROVISIONAL.

## Prior checkpoint: Comparative forgetting grid (36 cells) COMPLETE - encoder-dependent result (backfilled 2026-09-16)

This checkpoint was completed 2026-09-15 and is recorded in full in
`RESULTS_LOG.md:1408-1485` and `CHANGELOG.md:702-738`; it had not previously
been carried into this file. Numbers below are copied from those entries, not
recomputed.

User-run 36-cell Experiment-B/threshold 0.10/budget 400 grid completed across
both ViT-B arms (frozen, adapted), all six learners, and stream seeds 0-2.
All 72 capture and validation integrity receipts are PASS, all attempt01.
Supervision is matched at 400 stream labels for HDC, NCM and RanPAC-style in
every cell; naive/EWC consumed 17-53 and replay 37-79 labels and are not
label-matched.

Final Macro-F1 on `trusted_val`, medians over three seeds: frozen-B HDC
0.7220 / NCM 0.7313 / RanPAC-style 0.7406 (HDC best in 0/3 seeds); adapted-B
HDC 0.6001 / NCM 0.5275 / RanPAC-style 0.5243 (HDC best in 3/3 seeds, non-
overlapping per-seed ranges). The best cell in the whole 36-cell grid is
RanPAC-style on frozen at 0.7406, above HDC's own best of 0.7220.

Mean fixed-population forgetting, medians: frozen-B HDC 0.0497 / NCM 0.0435 /
RanPAC-style 0.0451; adapted-B HDC 0.0395 / NCM 0.0533 / RanPAC-style 0.0651,
but the adapted margin holds in only 2/3 seeds - seed 2 reverses at 0.0640
against NCM's 0.0627. **Reduced forgetting is NOT established.**

Novel-class uptake is a **null result** across all four repaired criteria.

Per-class concentration finding: the adapted-B advantage sits in two of
nineteen classes (`2_00_0`, `3_00_0`); the macro delta falls from +0.0635 to
+0.0069 once those two classes are excluded. Any aggregate macro-F1 figure
quoted without the per-class breakdown overstates the result.

The mechanism-ablation preparation referenced by earlier checkpoints
(`aggregation_prototypes.py`, `ablation_mode`, `ABLATION_PLAN.md`, 120 cells
across three phases) has since been unblocked and run to completion - see the
checkpoint above this one.

## Latest checkpoint: B400 six-head benchmark accepted - COMPLETE

User-run benchmark_development/2026-09-14_b_frozen_s0_b400_t010_attempt01 passed
independent saved-file audit:251 artifact hashes,22 pins,60 timed passes,
58,560 event rows,203,760 training predictions,78 resumable states/1,182 arrays,
46 compact archives/one shared blob. All205 boundaries retained PhoneExperienceHost
warnings; summary and receipt agree. This is qualified shared-desktop timing,
not a clean-GPU certificate. No rerun or warning-free repeat is required.

Total123.95MiB; operational elapsed about10.44minutes. HDC428.58events/s versus
NCM1206.89/RanPAC1370.35; slower than all five controls in every paired block.
Purchases:400 for each prototype,53 naive/EWC,37 replay. Equal allowance is not
identical supervision, even where counts match. No HDC throughput win follows.
See results/B400_BENCHMARK_AUDIT_2026-09-14.md. The agent performed saved-file
checks only; no official run was launched or completed output modified by agent.

Next is whole-study resource/backup planning and analysis templates, then the
remaining A1-A4 decisions, not another execution of this pilot. A3 remains open:
benchmark-only scenarios of roughly17.4GiB exclude capture/evaluation/backup and
appendix. User runtime cap remains removed; no full-grid/test release is implied.

## Prior checkpoint: background warnings instead of unknown-app aborts

User-run B400 preflight blocked on GitHub Desktop, Spotify's Game Bar WebView and
Edge Game Assist before creating an attempt. Preserve log
results/benchmark_development/preflight_run_d2ad9f4836c24975b2793339284290fa.log.
The user requested a practical fix. Current September14 policy supersedes the
unknown-GPU abort rules in older checkpoints below: unknown entries now remain
logged warnings, not silently accepted baseline or evidence of idle activity.
Known browser/game/recording/overlay/Python names and unreadable primary inventories
still block. Optional Search ownership failures warn rather than certify ownership.
Preflight PASS_WITH_WARNINGS exits0. Each in-run boundary retains entries; summary
and integrity receipt include environment_qualification and all warning boundaries.
All five blocks remain; integrity PASS is not a clean-timing or isolated-machine claim.
No processes killed, completed artifacts changed, or experiments launched by agent.
B400 attempt01 remains the next manual target if absent; no automatic retry/resume.
Verification details: results/BACKGROUND_WARNING_REVIEW_2026-09-14.md.
Final verification:475 CPU tests plus6 subtests passed in144.61s; live read-only
preflight PASS_WITH_WARNINGS/exit0 with PhoneExperienceHost recorded, no known
blocked workloads. Original three apps covered synthetically, absent live.
Read-only B400 plan and diff checks passed. No B400 benchmark executed.

## Prior checkpoint: runtime cap removed; B400 feasibility launcher prepared

User answered remaining machine-time question "as long as is needed". No numerical
runtime cap is required; do not repeatedly ask for one. This does not authorize
agent-launched training/benchmarking, unlimited storage or full-grid/test release.
Prepared scripts/benchmark-b400.ps1:one frozen-B/seed0/Experiment-B/allowance400/
threshold0.10 six-head condition with five paired blocks, fresh attempt01. Read-only
input plan passed. It uses existing measurement/acquisition/shared-storage policy;
actual purchases may differ by learner and do not necessarily exhaust400.

The script preserves the8GiB prelaunch headroom check, Settings-aware desktop
baseline, complete native logs and no-overwrite/no-auto-retry behavior. Confirmation
is RUN BENCH B400. Completed NCM B400 capture/validation and A25 attempt04 remain
untouched. No official run was launched by the agent. Broader storage/backup
forecasting, validation templates and A1-A4 still remain; one B400 pilot is not
a guaranteed worst-case bound. Verification complete:453 CPU tests plus6 subtests
passed in208.14s, including14 launcher cases. Read-only plan and diff checks pass;
reviews found no blocking issues (one nonblocking duplication note). See
results/B400_BENCHMARK_LAUNCHER_REVIEW_2026-09-14.md. The initial approval-service
limit resolved on retry; no remaining full-suite verification blocker.

## Prior checkpoint: compact benchmark attempt04 accepted

User completed attempt04. Independent saved-file audit verified252 artifact hashes,
22 pins,205 workload checks,60 timed ledgers,58,560 event rows and203,760 training
predictions. All46 compact archives share one verified projection blob; all78
resumable archives and1,128 array hashes/dependency sizes verified. See
results/A25_COMPACT_BENCHMARK_AUDIT_2026-09-14.md. No official run was launched by
the agent; no learner construction or scoring was part of this saved-file audit.

Production attempt size121.87MiB versus attempt02's1.3114GiB:90.9245% smaller.
Operational elapsed roughly10.2minutes; conditional144-group scenario17.14GiB /
24.5hours, not a complete-study forecast or upper bound. Original artifacts intact.
HDC median553.06events/s versus NCM1507.45/RanPAC1624.39; slower than all five
controls in all paired blocks. No throughput advantage or held-out claim follows.
Settings was absent at all205 recorded boundaries; do not claim real recurrence
was exercised in attempt04. Logs retain the approved shared-desktop limitations.

Attempt04 is COMPLETE: do not rerun. Broader high-budget/B and analysis/backup
feasibility, runtime/storage ceilings, templates and A1-A4 remain open. No new
experiment or full-grid launch is implied by this acceptance.

## Prior checkpoint: approved practical shared-desktop policy

On2026-09-14 the user confirmed Settings reappeared automatically during manual
benchmark attempt03. Saved failure:142 successful boundary inventories, last
before/3/hdc/latency; failure after that pass in paired block4/5, no final PASS.
Preserve attempt03 and preflight_run_655e0b085a5d4566b39606f9653180dd.log; never
resume, delete or relabel its incomplete timing evidence.

User explicitly approved logging/accepting ordinary identified desktop activity.
Windows Settings now joins the exact-path baseline regardless of PID changes.
Other accepted desktop apps remain; games, recording/overlays, browser activity,
other training workers and unapproved GPU entries remain blocked. No arbitrary
Windows-directory exemption. Logs explicitly say interference_measured=false;
presence is not measured interference. Same policy for all six heads, all five
blocks retained, report variability and shared-desktop limitations.

Verification:446 CPU tests plus6 subtests passed in128.75s; both independent
reviews found no issues. Current read-only workload preflight and attempt04 input
plan passed; launch-time preflight still applies. git diff --check clean.
Next is fresh user-owned attempt04 via scripts/benchmark-a25.ps1 -Attempt 4;
launcher defaults remain unchanged and existing attempts are refused. No official
run was launched by the agent. A1-A4/full-grid/test-release gates remain open.

## Prior checkpoint: shared-projection save/load integrated

User approved production integration on 2026-09-14. New longitudinal budget
captures and benchmark runs share content-addressed projection blobs within each
run directory, including prediction-only exports. Logical learner schema1 and
legacy standalone loads remain supported. Compact loads require an explicit run
root; copy/back up the whole run, including projection_blobs. No old artifacts
were migrated or overwritten. See LEARNER_CHECKPOINTS.md for the physical format,
atomic exclusive publication, corruption checks and dependency-size accounting.

Verification: full CPU suite passed434 tests plus6 subtests in133.93s; git diff
--check clean. Attempt03 read-only input plan passed; no real experiment launched.
See results/SHARED_PROJECTION_INTEGRATION_2026-09-14.md for review scope. A3 broader
resource estimates/runtime ceiling, A1-A4 freeze and untouched-test release remain
open. Next evidence-bearing step is a fresh manual storage-acceptance pilot, not
a repeat/overwrite of completed attempt02. Old result/receipt hashes remain valid
for their original files; current source hashes are a new implementation version.

## Prior checkpoint: shared-projection storage test passed

User authorized testing deduplication and freed disk space (verified 71.10 GiB
available before the scratch test). One shared projection replaces 46 duplicate
HDC/RanPAC exports; 1,066 arrays/metadata entries reconstructed byte-for-byte,
92 negative integrity cases rejected, all 252 original source files unchanged.
Equivalent whole-pilot footprint: 1.3114 -> 0.1190 GiB (90.9245% reduction).
Conditional A/B benchmark scaling: ~17.14 GiB, not a complete-study guarantee.
See results/PROJECTION_STORAGE_TEST_2026-09-14.md; scratch test evidence retained.

At the prototype checkpoint, production save/load was NOT changed. Next was shared-blob
integration with legacy/relocation/atomicity/integrity and synthetic learner
round-trip checks before any new manual pilot. No original result deleted,
checkpoint rewritten, learner executed or official run launched by the agent.
A3/runtime ceilings and final release remain open. Quantum remains out of scope.

## Prior checkpoint: feasibility worksheet prepared; resource input needed

See FEASIBILITY_WORKSHEET_2026-09-13.md. Read-only measurement found ~30.74 GiB
free on G:. The accepted six-learner pilot used 1.3114 GiB and about 11.3 minutes
after confirmation. Scaling its unchanged layout across proposed A/B benchmark
groups gives ~188.84 GiB / ~27.1 hours as a conditional scenario, not a validated
forecast; separate capture/evaluation/analysis/backup costs are excluded.
No unchanged full grid fits the observed disk envelope. Need user-specified
storage capacity/location and runtime ceiling before choosing additional storage,
verified future checkpoint sharing or an explicit scope amendment. No conditions
silently omitted, artifact deleted/relocated, or experiment launched. A3 stays open.

## Prior checkpoint: A25 six-learner benchmark attempt02 accepted

User completed attempt02; saved-file audit matched all 251 artifact hashes and
21 pins, 60 timed-pass ledgers, 58,560 event rows and 203,760 training predictions.
All 205 workload checks passed under the recorded shared-desktop baseline.
Median stream events/s: HDC532.34, NCM1628.22, RanPAC1183.00, naive1313.13,
EWC1473.56, replay1386.09. HDC was slower than every comparator in every paired
block; this condition does not support an HDC throughput advantage.
See results/A25_BENCHMARK_AUDIT_2026-09-13.md for accounting and claim limits.

Attempt02 is complete; do not rerun the command. Attempt01 remains preserved.
First benchmark acceptance is complete, not 972-condition feasibility or final
release. Next: feasibility and validation analysis/templates, then A1-A4 freeze.
The agent read/reduced saved files only; no official run or learner execution.

## Prior checkpoint: benchmark attempt01 blocked; approved guard recovery

User-run attempt01 stopped at the initial workload guard before learner setup
or timing. Only failure.json exists; its original preflight log is preserved.
The driver listed Windows desktop apps in the compute query, and the initial
guard rejected all of them. PowerShell Stop handling also truncated stderr.

The user approved an exact-executable-path Windows desktop/Terminal/VS Code
baseline. Accepted PID/path lists are recorded; no blanket C+G exception.
Overwolf, browsers/WebView, games, other Python and unknown GPU processes remain
blocked. Native stderr logging now drains output and checks the real exit code.
The workload check runs before the RUN prompt or attempt-directory creation.
Next user command is the absolute-path benchmark-a25.ps1 launcher with -Attempt 2;
existing attempts are refused, with no automatic retry, cleanup or app termination.
See BENCHMARKS.md and the latest RESULTS_LOG.md verification checkpoint.

Latest read-only preflight still reported WebView, Sticky Notes and Settings;
machine state must be checked again at launch. The agent has not run a benchmark.
A1-A4, grids, benchmark acceptance/feasibility and untouched-test release stay open.
Recovery verification: full suite passed 394 tests plus six subtests before the
final Search-owned WebView refinement; 31 focused tests passed after it. Windows
Search's pinned WebView is now allowed only with a verified recorded parent
chain; other WebView remains blocked. No final full-suite rerun after that narrow
refinement (user requested immediate finish). Attempt02 remains user-owned.

## Prior checkpoint: approved benchmark development implemented

The user approved the bounded benchmark policy on 2026-09-13. Added separate
timing/environment/study modules without editing the existing learner, runner,
checkpoint or common metric code. The implementation restores states/RNG,
checks every measured pass against untimed references, separates throughput,
update latency, peak sampling and retention-grid payload/storage inventories.
See [BENCHMARKS.md](BENCHMARKS.md) for scope, review and verification.
Final synthetic wrapper verification: 371 tests plus six subtests passed in
119.72 seconds. Explicit experiment/budget/seed provenance and fail-closed
empty/malformed Windows inventory handling are regression-tested.

Next user-owned pilot: ONE frozen-B/seed-0/Experiment-A/B=25 condition, all six
learners, one warm-up and five paired timing blocks. The one-command launcher
is scripts/benchmark-a25.ps1; the read-only input plan passed. This new timing
workload does not overwrite or replace completed A25/B400 predictive evidence.
No real benchmark has been launched by the agent. Existing attempt directories
are refused; preserve failures and logs. Full-grid/test release and A1-A4 remain
open. After the user says done, inspect saved reports directly.

## Prior checkpoint: B400 longitudinal validation verified

User completed all 20 states on the fixed 1,132-image/19-class validation set.
Audit matched 22 output hashes and 48 input/provenance pins, checked 22,640
ID-aligned predictions and reconstructed all saved metrics/forgetting/uptake
exactly. Final Macro-F1 .7260553028, macro recall .8108204853, accuracy
.8162544170, ECE .7359850955. Mean old-class forgetting .0438711956 (4.3871
percentage points), not a comparator improvement. Two novel classes never
reached .80 recall at observed uptake milestones. See the
[audit](results/B400_VALIDATION_AUDIT_2026-09-12.md).

B400 training AND validation are complete; do not rerun either launch block.
Only saved artifacts were read/reduced by the agent. No test or benchmark run.
The then-next decision was [bounded benchmark policy](BENCHMARK_DEVELOPMENT_PLAN.md), proposing
common CUDA placement, host-resident embeddings with transfers included,
one-thread CPU settings, one disposable warm-up, five paired rotating blocks,
separate latency measurement and memory/storage accounting. These settings are
subsequently approved separately on 2026-09-13. Full grids, uncertainty,
feasibility ceilings, appendix and A1-A4/test release remain open.

## Prior checkpoint: B400 capture verified; longitudinal validation prepared

User-run frozen-B/NCM/seed-0/Experiment-B allowance 400/threshold .10 capture
PASSED. Independent audit matched all 27 artifact hashes, unchanged source code
and input hashes, 976 history/event rows, and 3,396 unique final training IDs.
There were 450 potential requests, 400 purchases, 50 denied requests and zero
unused allowance (2,820 supplied learner labels including bootstrap). All four
novel classes registered. Twenty physical snapshots cover all ten retention
times plus available uptake milestones. Do not rerun this completed cell.

The manual paste separated `-FilePath $bLog -Append` from Tee-Object. Preflight
and RUN confirmation are in preflight_run_6c7795364b0b4fe3b4471dc604ce66ef.log;
run output is in b400_recovery_01.log, both beside the capture directory. The
later standalone -FilePath error was a shell logging error, not a learner failure.
Preserve both logs. Do not infer benchmark cleanliness from this integrity PASS.

Added a strict A/B snapshot validation evaluator and a one-command guarded launch
script. See [SNAPSHOT_VALIDATION.md](SNAPSHOT_VALIDATION.md). The real-input plan
passed for 20 states x 1,132 fixed validation examples, without loading/scoring
a learner. Next user action is the guarded validation command; no retraining.
The agent reads resulting reports and ledgers directly. Actual longitudinal
validation evidence, comparative grids, benchmarks, final protocol A1-A4 and
test release remain pending. Existing learner/runner/metric code is unchanged.
Full verification: 324 tests plus six subtests passed in 98.46 seconds.
Standards review found no issues; the Spec partition-filter finding was fixed
and regression-tested. No real scoring was performed by the agent.

## Prior checkpoint: approved A/B acquisition and snapshots implemented

The user explicitly approved Experiment-B acquisition, retention/uptake snapshot
times and uptake targets for bounded development on 2026-09-11. Implementation
and synthetic verification are complete: **281 tests plus six subtests passed
in 89.66 seconds**. See [BUDGET_SNAPSHOTS.md](BUDGET_SNAPSHOTS.md).

The then-next user-owned launch was one frozen-B/NCM/seed-0 Experiment-B development cell,
allowance 400, threshold .10, capturing retention and available uptake states.
The final RUN_COMMANDS.md block gives the guarded command. Its read-only real-
input plan passed; the agent has not launched training, scoring or benchmarking.
After the user says done, inspect saved reports/hashes directly, not by asking
them to paste accessible files. Do not rerun the completed A25 or 108 historical
cells. Keep all existing artifacts and their original code hashes unchanged.

The new snapshot format is NOT accepted by the legacy A25 validation evaluator.
New-format integration is now separate in evaluate_snapshots.py; real
forgetting/uptake curves, benchmarks, full-grid/test release and A1-A4 remain open.
The runner observer changes its code hash; legacy compatibility checks are not
waived. These are development infrastructure results, not predictive evidence.

## Prior checkpoint: A25 validation verified; metric arithmetic implemented

The user completed saved-state A25 validation. All three report-listed hashes
match; 1,132 unique prediction rows reproduce the saved common classification
metrics and ECE exactly. Macro-F1 0.6600447531, macro recall 0.7225543250,
accuracy 0.8356890459, ECE 0.7468861113. Two classes remain unregistered:
3_a9_3 and 6_a11_1. This is NCM/frozen-B/seed-0/B=25 development evidence, not
an HDC comparison, untouched test result, forgetting curve or benchmark.
Do not rerun the completed training or validation block.

Added [pure longitudinal metrics](LONGITUDINAL_METRICS.md): fixed-population
forgetting, novel-class uptake summaries, normalized budget AULC and paired
descriptive seed contrasts. Synthetic tests only; real longitudinal states and
new predictions were not generated. Phase 3 and budget-wide 2A.3 remain open.

The then-next input gate was approval of the draft Experiment-B acquisition settings and the
retention/uptake snapshot schedule for bounded development (see the linked
decision checklist). Full-grid/test/benchmark release and A1-A4 remain separate.
That bounded approval has since been received and implemented as recorded above.

Workflow clarification: the agent reads saved reports, checks hashes and analyzes
existing results directly. Give the user commands for new training/evaluation/
benchmark launches, not for displaying files already accessible in this repo.
Continue unblocked development until an actual decision or manual run is needed.
Full verification: 222 tests plus six subtests passed in 46.26 seconds.
Standards/Spec review: no blocking findings in the new arithmetic-only scope.
No official run was launched by the agent.

## Prior checkpoint: A25 verified; saved-state validation implemented

The user completed frozen-B/NCM/seed-0/B=25. Independent read-only audit verified
all seven artifact hashes, 976 history rows, 25 purchased/scheduled labels,
2,445 total supplied learner labels, and 3,396 unique training prediction IDs.
Preserve that completed attempt; do not rerun it.

Added [saved-state validation evaluation](VALIDATION_EVALUATION.md): fixed full
19-class common metrics, strict validation-only paths, source/state provenance,
and learner/RNG purity checks. The real-input plan passed for 1,132 validation
examples without learner construction or scoring. Next user action is the new
manual validation block at the end of RUN_COMMANDS.md, using the existing A25
state and a fresh output directory. Return integrity_report.json and metrics.json.
No official run was launched by the agent; no trusted-test labels were opened.
Full test wrapper: 178 tests plus six subtests passed in 42.10 seconds. Standards
and Spec reviews found no blocking issues within this bounded implementation.

2A.3 implementation is partial: final-state evaluator exists, but actual
validation acceptance across budgets remains pending. Longitudinal snapshots,
forgetting/uptake, benchmarks, A1-A4 and full-grid/test release are still open.

## Prior checkpoint: Experiment-A acquisition implemented

User approval on 2026-09-11 covers additional budgets 0/25/50/100/200/400,
label-blind nested schedules shared across all six learners and both arms,
and novel registration only on purchased scheduled labels. Implementation and
synthetic verification complete TODO 2A.1-2A.2, not the whole Experiment A.
See [EXPERIMENT_A.md](EXPERIMENT_A.md). Full wrapper verification: 155 tests
and six subtests passed in 54.80 seconds. Two-axis review has no remaining
blockers; a preparation-time input provenance race was fixed and regression-tested.

The then-next user action was the first frozen-B/NCM/seed-0/B=25 training-side development
cell, using the new direct-log block at the end of RUN_COMMANDS.md. Expected:
25 additional labels, 2,445 supplied learner labels, 976 events. Stop after that
cell and return its integrity report. A read-only real-input plan passed; no
Experiment-A learner execution has been performed by the agent.

Fixed validation evaluation (2A.3), longitudinal snapshots, primary metrics,
benchmarks, full 216-cell execution, A1-A4 and test release remain pending.
Changes are uncommitted; unrelated edits and all historical artifacts remain
preserved. The new opt-in runner branch changes its code hash: historical
manifests retain their original hashes; the earlier audit describes that revision.

## Prior checkpoint: all 108 historical reconstructions verified

Completed 2026-09-11: all 108 user-run historical cells PASS. Read-only audit
verified exact original-grid coverage, all 648 artifact hashes, current source/
input/code provenance, and 108 final ledgers of 3,396 unique training IDs each.
All reports have zero history/summary mismatches over 976 events. See the
[reconstruction audit](results/RECONSTRUCTION_AUDIT_2026-09-11.md).

TODO 1.3, 1.4, C and R2 are complete for training-side observed-output fidelity.
Do not rerun the completed reconstruction blocks. The subsequent Experiment-A
development approval and implementation are recorded above. A1-A4 and held-out
release remain open.
All experiments remain user-launched. No official run was launched by the agent.

## Earlier reconstruction milestones (completed; not launch instructions)

Latest update 2026-09-11: all 12 representative cells now passed (six learners,
both ViT-B arms, seed 0, threshold 0.10). All 66 artifacts in the latest 11-cell
batch were hash-verified; all 12 cells' current code/input hashes also match.
Next manual stage: the remaining 96 historical cells, excluding these 12, using
the new direct-log block at the end of RUN_COMMANDS.md. Stop at any failure.
The agent ran only read-only guards for that block: 96 distinct pending cells,
required input files present, and about 33 GiB free. No experiment was launched.

The latest batch's transcript exists but ended before the confirmation prompt.
The final Stop-Transcript error means no transcript was active at cleanup; why
it ended early is not established. The saved reconstruction reports/states are
intact. New commands use direct preflight/per-cell logging instead. Do not rerun
completed cells to repair a transcript or use these timings as benchmark evidence.

Earlier milestones below are chronological context, superseded by this update.

Update 2026-09-11: the user manually completed frozen-B/NCM/seed-0/threshold-0.10.
Its report is PASS: all 976 history rows and the summary match, with zero reported
input/code changes. Read-only checks verified all six artifact hashes, current
source/input/code hashes, and 3,396 final prediction rows. This is 1/12
representative checks, not completion of 1.3 or the 108 historical cells.
Next: manually run the remaining 11 representative cells using the new block
at the end of RUN_COMMANDS.md, which excludes the completed cell. Return their
reports before expanding further. The first transcript shows browser/overlay
activity before confirmation, so no clean timing or speed claim is made.
The earlier first-cell launch instruction below is now completed.

The user authorized continuing the TODO until input is needed, while explicitly
retaining manual ownership of every experiment. Reconstruction infrastructure
is implemented; see [RECONSTRUCTION.md](RECONSTRUCTION.md) and the new first-cell
block in [RUN_COMMANDS.md](RUN_COMMANDS.md). The original next user action was
frozen-B/NCM/seed-0/threshold-0.10 reconstruction and returning its fidelity report
plus preflight transcript. This is one cell, not permission to start the grid.

Read-only planning passed for that real cell without constructing a learner.
The tool preserves final state, ID-aligned final training predictions, source
provenance and exact/tolerant fidelity comparisons. Review fixed an indirect
held-out-label read by using locked-spec and training-only validation instead;
non-training paths are rejected before opening inputs. Synthetic tests cover
these guards and all six learners. No actual historical fidelity is claimed.

TODO 1.2/B infrastructure is complete; 1.3/C, 1.4 and A1-A4 remain open.
Final verification on 2026-09-11: `scripts/hdc.ps1 -Action test` completed with
114 passed and six subtests passed in 31.53 seconds (CPU). This includes the
training-only filesystem guard and final-prediction summary reconstruction.
The new manual PowerShell block passed syntax validation without execution;
`git diff --check` passed. Changes remain uncommitted; existing unrelated edits
are preserved. The earlier approval/usage-limit interruption is resolved for
this test invocation. No historical experiment has been run.
No official run was launched by the agent. No experimental results were changed.

## Prior checkpoint: authorized development steps 1–2

The user authorized TODO alignment and learner-state preservation on 2026-09-10.
The [checkpoint implementation](LEARNER_CHECKPOINTS.md) now saves/loads all six
learners with opt-in runner final-state exports. Synthetic CPU tests verify exact
predictions and continued updates, HDC ties, EWC state, replay RNG/buffer state,
and unchanged runner CSV/JSON outputs under a deterministic clock. A checkpoint-
only constructor path avoids changing deferred CUDA seeds during neural reload;
ordinary seeded initialization and scientific update rules are unchanged.

Verification: focused checkpoint/baseline/runner suite 36 passed; full local
CPU suite 89 passed plus six subtests in 10.84 seconds. The sandbox's temporary-
directory permissions required an approved out-of-sandbox test invocation.
No trusted dataset was used for the new checks; no benchmark or held-out
evaluation was run. Cross-device fidelity remains unverified.

The TODO now includes explicit throughput and state/storage tasks; only learner
preservation and synthetic checkpoint verification are marked complete. Historical
reconstruction/fidelity, both budget policies, metrics, benchmarks, and test access
remain pending. A1–A4 are not closed by this development authorization.
No official run was launched by the agent.

## Prior checkpoint: protocol draft and research priorities

The [core evaluation protocol draft](../docs/TODO/HDC_CORE_EVALUATION_PROTOCOL_DRAFT_2026-09-09.md)
now proposes the next classical HDC/NCM/RanPAC evaluation, following the user's
TODO v2. It separates identical-example and identical-allowance experiments,
defines proposed state/replay and fixed-population evaluation rules, and keeps
quantum optional. It is not approved, frozen, or implemented.

The latest 2026-09-10 clarification supersedes the earlier same-day AULC-only
primary hierarchy: the supervisor wants clear evidence of reduced forgetting
and higher throughput. The user authorized this documentation update. The draft
now makes fixed-population forgetting and cached-embedding continual-head
throughput primary thesis outcomes, explicitly comparing HDC against NCM and
RanPAC-style as well as naive/EWC/replay. Predictive performance, novel-class
uptake, labels, and memory are required joint evidence, not disposable checks.

The desired joint claim is not established: rolling forgetting against prototype
controls depends on encoder arm, and recorded update durations favor NCM/RanPAC-style.
Matched-supervision held-out retention and validated throughput remain pending.
The draft adds paired forgetting contrasts, longitudinal resampling, throughput
ratios, repeat/state-restoration rules, and timing boundaries. A1–A4 remain open
for appendix, release, feasibility, and exact endpoint/claim/benchmark decisions.
The binding specification has an appended scope note; historical registrations
and results are preserved. Documentation authorization is not experiment approval.

Review must reconcile the binding 25-class repeated-CV appendix obligation,
which TODO v2 omits, and the older longitudinal-test requirement. The user-edited
TODO is preserved unchanged; no checkboxes are marked complete. No test-set
evaluation, code changes, or test-suite runs occurred in this checkpoint.
No official run was launched by the agent. That documentation-only checkpoint
preceded the bounded implementation authorization recorded above.

## Current state

The encoder ladder, approved trusted splits, probe gates, and **108 continual
runs** are complete in this checkout. There are 108 summary JSONs and 108
976-row histories: six learners × frozen/adapted ViT-B × three stream seeds
× three confidence thresholds. The 36 NCM/RanPAC runs are in commit
`9cfd20a`. The [scientific report](results/SSL_HDC_REPORT.md) remains
**PROVISIONAL**.

Read the [binding specification](../docs/still_exploring/self_supervised_true_label_path/DINOV3_HDC_IMPLEMENTATION_HANDOFF.md),
[runbook](RUN_COMMANDS.md), [results log](RESULTS_LOG.md), and
[current audit](results/VERIFICATION_AUDIT.md). Earlier log entries describe
their state at that date; they do not make completed runs pending again.

## Completed artifacts and implementation

- Sanctioned unlabeled pool: 51,906 unique images; the 5,193 validation IDs
  are already included, and trusted IDs are excluded. Labels are unused.
- Frozen/adapted DINOv3 small/base/large ladder and all three probe gates.
  Selected adapted base: epoch 3; adapted large: epoch 4; small: frozen.
- Approved Option-A splits: 3,396/1,132/1,132 train/val/test, 19 classes.
  Option-B repeated-CV assignments exist; its 25-class analysis is unrun.
- Canonical stream spec: `stream_spec.json`, SHA-256
  `62b545d31d592735be4404451bddd7f603576d127351f337e1e34c8fd04604cc`.
- HDC, L2-normalized online NCM, RanPAC-style ReLU prototypes, naive, EWC,
  and reservoir replay. RanPAC omits ridge/decorrelation; it is a controlled
  prototype ablation, not the full published algorithm.
- EWC consolidates at novel-class boundaries and uses the bounded-decay
  amendment 0.9. The replay audit found no implementation defect.
- Earlier checkpoint/reconstruction verification: September 11, 114 CPU tests
  plus six subtests; later development verification is recorded above and in
  VALIDATION_EVALUATION.md, not new scientific results.

## Evaluation and claim boundaries

At threshold 0.10, frozen-B median final Macro-F1 is HDC 0.7720, RanPAC
0.7602, NCM 0.7489. The final pass scores all 3,396 bootstrap+stream
**training-pool** samples, not trusted_test. Prequential scores are HDC
0.7451, RanPAC 0.7566, NCM 0.7504. Labels charged during the stream differ;
the shared 2,420 bootstrap labels are excluded from that charge.

Current evidence supports prototype-method strength under the registered
protocol, not unique HDC superiority at equal annotation budgets. Query
confidence is a cosine-score margin for every learner; ECE instead uses
softmax-derived top probability. The forgetting score is peak-minus-final
rolling known-class accuracy, not standard fixed-task held-out forgetting.
Timing covers classifier updates only and does not establish hardware speedup.

## Remaining work

Equal-label-budget evaluation, fixed-population held-out forgetting, and validated
continual-head throughput remain unresolved. Reconcile the proposed benchmark
repetitions/device policy and practical performance/uptake criteria before freeze.
Encoder-geometry Study 2 requires a dated protocol reconciliation before
execution; its original four-learner specification is retained as history.
Frozen 7B extraction and extra continual encoder rungs have no results here.
The standalone embedding-VQC pilot is a separate, implemented but unrun study.

Official runs remain user-owned. The runbook labels completed blocks as
reproduction history; do not restart them automatically. Preserve historical
hashes for unaffected HDC/naive/replay runs and all scientific artifacts.

Added 2026-09-16: the mechanism ablation (120 cells) is complete and reported
above; it is no longer remaining work. What remains from it: Phase 4 (seed
extension of surviving conditions) has not been selected or run. The draft
release protocol at `docs/TODO/HDC_TEST_RELEASE_PROTOCOL_2026-09-15.md` is
unapproved - it requires explicit user and supervisor sign-off before the
sealed `trusted_splits/test.csv` may be used, and this checkpoint does not
provide that sign-off.
