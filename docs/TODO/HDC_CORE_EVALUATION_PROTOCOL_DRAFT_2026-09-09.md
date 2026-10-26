# HDC core evaluation protocol agreement — draft

Created: 2026-09-09. Revised: 2026-09-10 (supervisor priorities and comparator reconciliation).
Status: **DRAFT FOR REVIEW — COMPLETE PROTOCOL NOT APPROVED OR FROZEN.**

Subsequent bounded approval 2026-09-11: the user explicitly approved Experiment B
at budgets 0/25/50/100/200/400 and thresholds .05/.10/.20, retaining uncertainty
and ground-truth-assisted novel requests, single purchase charging, underspending,
and no update/registration after exhaustion. The user also approved development
retention snapshots at processed-event counts 0/195/196/390/391/585/586/780/781/976,
novel uptake snapshots after purchases 1/2/5/10/20/50/100 plus final, and uptake
targets 80% of final recall and absolute recall .80. This authorizes implementation
and synthetic verification, not agent-launched real experiments, full-grid/test
release, benchmark policy freeze, or closure of A1-A4.

Bounded approval 2026-09-11: the user approved Experiment-A budgets
0/25/50/100/200/400 and label-blind nested schedules shared across all six learners
and both arms, with novel registration only on purchased scheduled labels.
The acquisition implementation and synthetic tests now complete TODO 2A.1-2A.2.
See [implementation](../../ssl_hdc_pilot/EXPERIMENT_A.md). This does not approve
the full 216-cell grid, held-out scoring, or resolve A1-A4. Experiment B was
subsequently approved for development as recorded above.

Progress update 2026-09-11: under subsequent user authorization and manual run
ownership, all 108 historical cells were reconstructed and passed observed-output
fidelity checks. The [audit](../../ssl_hdc_pilot/results/RECONSTRUCTION_AUDIT_2026-09-11.md)
records independent artifact verification. This completes historical state/fidelity
work only; it does not approve the proposed budget policies or close A1-A4.
The earlier development-authorization note below is chronological provenance.

Development authorization — 2026-09-10: the user approved next-steps 1–2 only:
align the TODO with forgetting/throughput priorities and implement learner-state
preservation with non-test prediction/continued-update checks. This supersedes
the earlier documentation-only restriction for that bounded implementation, not
for experiments, benchmarks, historical grids, or test access. A1–A4 remain open.

This document proposes the first step of [TODO v2](HDC_THESIS_NEXT_STEPS_TODO.txt).
It does not itself authorize implementation, replay, official experiments, or test-set
access. The [binding specification](../still_exploring/self_supervised_true_label_path/DINOV3_HDC_IMPLEMENTATION_HANDOFF.md) records the dated scope clarification; this draft does not freeze new experimental rules or retrospectively amend historical runs.
No TODO checkbox is completed merely by creating this draft.

## 1. Agreement to review first

The core question is whether classical HDC reduces forgetting and increases
continual-head throughput relative to NCM, RanPAC-style, and the tested gradient
learners under identical supervision, while maintaining useful held-out predictive
performance and novel-class learning. These are hypotheses, not established wins. A generic
prototype advantage, or a failure of the earlier HDC advantage to generalize, is
also an admissible finding. Quantum is an optional subsequent extension, not a
required contribution. Thesis novelty and acceptance still require literature
positioning and supervisor judgment; this protocol cannot guarantee either.

Choices below remain proposals except the explicitly dated bounded approval.
None is a retrospective registration of historical experiments.

| Decision | Proposed agreement | Approval status |
| --- | --- | --- |
| Core scope | Six existing learners, two existing ViT-B arms, existing three stream seeds; both Experiments A and B | Pending review |
| Supervision | Additional-label budgets 0, 25, 50, 100, 200, 400; common 2,420-label bootstrap | Experiments A/B development approved by user 2026-09-11; final freeze pending |
| Shared-label control | Label-blind, nested selection of stream events; identical labels and timing across methods | Experiment-A development approved by user 2026-09-11 |
| Test boundary | Validation-only development, then one locked analysis release covering all predeclared final and longitudinal states | Pending review; reconciles older longitudinal-test requirement |
| Previously approved appendix | Retain the 25-class repeated-CV obligation; agree its separate plan or obtain an explicit supervisor-approved amendment | Unresolved scope gate |
| Endpoint hierarchy and claims | Primary thesis outcomes: fixed-population forgetting and continual-head throughput, with explicit HDC-versus-NCM and HDC-versus-RanPAC-style contrasts; predictive performance and uptake are required joint evidence (§5) | Supervisor priorities relayed by user on 2026-09-10; measurement details and claim criteria pending freeze |
| Cost environment | Cached-embedding continual-head throughput, separate update throughput/latency, and state/storage footprint on the current research machine | Proposed measurement scope; device placement, repetitions, and feasibility pending freeze |

The binding specification §4 records supervisor approval of **Option A+B**:
19-class headline evaluation AND a 25-class, 5-fold × 10-repeat appendix, both
reported regardless of outcome. TODO v2 does not cover that appendix. This draft
addresses the 19-class core, but cannot silently cancel that obligation or claim
whole-thesis completion. Do not launch the CV appendix during core development;
its overlap with the headline test population also requires an explicit data-use
and sequencing plan before any CV outputs are inspected.

### Amendment checklist — required before freeze

These are approval gates, not completed amendments to the binding specification.
For each closure, record the decision, approver, date, and linked evidence in this
document. Drafting a preferred resolution does not close the gate.

- [ ] **A1 — Appendix scope and sequencing.** Proposed resolution: retain the
  existing 25-class, 5-fold × 10-repeat appendix and freeze its learner/encoder,
  fold-wise training/selection, metrics, and CI plan before the headline test
  release. Execute and inspect it only after that locked headline release, with
  no feedback into the headline protocol. Account for overlap with Option-A
  test IDs and upstream checkpoint-selection labels; do not call those CV scores
  untouched generalization. Obtain supervisor sign-off on the concrete appendix
  plan or an explicit scope amendment; no appendix experiment is authorized here.
- [ ] **A2 — Test release.** Explicitly approve one locked release covering final
  AND longitudinal states (§7), with an immutable state list and no intermediate
  test-guided decisions. Record this interpretation against the older test rule.
- [ ] **A3 — Feasibility.** Before approving 972 logical conditions, review a
  validation/synthetic-only runtime and storage worksheet including snapshots,
  predictions, timing instrumentation, and the bootstrap analysis. Record usable
  disk space, peak working storage, expected wall time, approved ceilings, and
  any verified duplicate reuse. If infeasible, approve a dated scope revision
  before freezing; do not silently omit cells. The appendix needs its own estimate.
- [ ] **A4 — Endpoints and claim rules.** Formally reconcile the original §9
  primary contribution outcomes with the supervisor priority of forgetting
  reduction and higher throughput in §5. Freeze budget-wise contrasts, benchmark
  device/workload/repetitions, forgetting uncertainty, and the joint evidence
  criteria against each comparator. Approve its claim-to-evidence table and cost
  environment. Any numerical practical-benefit threshold or deployment
  requirement must be specified before test access; otherwise report measured
  tradeoffs without an unqualified practical-superiority claim.

2026-09-10 amendment record: the earlier five review topics remain incorporated.
The subsequent supervisor priorities relayed by the user supersede the earlier
AULC-only primary hierarchy: forgetting and throughput are now primary thesis
outcomes, explicitly tested against NCM and RanPAC-style as well as gradient
controls. This authorizes the documentation revision, not experiments or test
access. A1–A4 remain open for concrete protocol freeze; historical registrations
and numerical artifacts are unchanged.

## 2. Inherited controls and present evidence

Preserve the approved Option-A train/validation/test split: 3,396/1,132/1,132
images, 19 classes. The training pool supplies 2,420 bootstrap examples and a
976-event stream, initially 15 known classes with four novel classes. Preserve
the canonical stream ordering construction, introduction rules, and split IDs.
Pair methods on identical bootstrap IDs/order and stream IDs/order within each
stream seed; bootstrap membership need not be identical between stream seeds.

Primary learners are HDC, L2-normalized online NCM, and RanPAC-style ReLU
prototypes. Naive incremental, EWC, and reservoir replay remain required gradient
comparators. Report HDC against each of the five alternatives on both primary
outcomes; NCM/RanPAC-style cannot be relegated to accuracy-only comparisons.
RanPAC-style here is not the full published ridge/decorrelation algorithm.
Use frozen ViT-B and the already selected adapted ViT-B epoch 3 embeddings. No
backbone retraining, new architectures, or outcome-driven learner retuning.

Preserve existing learner configurations, including 10,000-dimensional prototype
projections, learner/projection seed 0, registered neural updates, bounded-decay
EWC, and Algorithm-R replay with capacity 512 and current-plus-32 replay updates.
Use stream seeds 0, 1, 2. Existing confidence thresholds 0.05, 0.10, 0.20 remain
complete reporting strata for historical reconstruction and Experiment B, not a
menu from which to select a test winner. Experiment A has no acquisition threshold.

The [current report](../../ssl_hdc_pilot/results/SSL_HDC_REPORT.md) remains
PROVISIONAL. Its 108 completed cells evaluate final states on the training pool,
not held-out generalization; their label charges differ. Existing histories and
results remain immutable. The runner now optionally saves a learner-resumable
final state; CPU synthetic prediction/update round trips are verified. Full
historical reconstruction and observed-output fidelity passed for all 108 cells
on 2026-09-11; original input identity and held-out evidence are separate. See the
[checkpoint implementation note](../../ssl_hdc_pilot/LEARNER_CHECKPOINTS.md).

Current evidence does not establish the desired joint claim. At threshold 0.10,
frozen-B median rolling forgetting is HDC 0.1748, NCM 0.1677, and RanPAC-style
0.1639; adapted-B values are 0.2425, 0.2977, and 0.3003 respectively. Recorded
update durations are shorter for NCM/RanPAC-style on both arms. These are rolling
stream proxies and unsynchronized host timers under unequal label expenditure,
not fixed-population forgetting or validated throughput. HDC's final training-pool
Macro-F1 lead is a different result and cannot substitute for either primary claim.

## 3. Data roles and state selection

- Bootstrap labels initialize every learner; only purchased stream labels may
  update it afterwards. Validation examples never become bootstrap/stream data.
- Validation supports implementation debugging, analysis design, and any
  explicitly documented pre-freeze protocol amendments. Keep a decision log;
  report validation as development evidence, not independent confirmation.
- Trusted test supports only the final locked analysis. Do not generate or inspect
  test metrics, predictions, class-wise errors, or test-derived selection advice
  during development. Cached all-partition embeddings must be partition-filtered
  before development evaluators receive them.
- Every budget condition starts from its paired bootstrap and processes the
  **entire 976-event stream**. The primary final state is after event 976, not the
  instant the last affordable label was purchased. No best intermediate state is
  selected. Predictions are always made before the current event's update.
- Intermediate states exist only for the predeclared retention/uptake analysis.
  Validation or test evaluation must not change model mode permanently, RNG state,
  class registry, replay memory, counters, or future updates.

Before calling test results untouched, audit local split access and prior
evaluation provenance. Local trusted-data exclusion from encoder adaptation is
not proof of absence from the foundation model's external pretraining data.

### Upstream supervision is separate from continual-run charges

This budget study is **conditional on already selected representations**. The
binding specification's 2026-07-18 probe-gate amendment (§5.2) permits fitting
probes on trusted_train labels and choosing the adapted checkpoint using
trusted_val Macro-F1. Thus label-free encoder adaptation is not label-free
checkpoint selection: the upstream gate uses the 3,396 training and 1,132
validation labels, including labels withheld from low-budget continual heads.
The selected adapted ViT-B epoch 3 is fixed input, not reselected at each budget.

Report separate supervision ledgers for (i) upstream probe/development/selection
uses, (ii) labels actually delivered to each continual learner, and (iii) final
evaluation labels. Identify their populations, roles, unique IDs/counts, reuse,
and overlap. Reusing a label across probes, seeds, or budgets is not a newly
acquired annotation. Do not sum overlapping ledgers into a project total or
present per-run charges as the entire project's labeling requirement. The
conditional budget comparison does not establish an end-to-end low-label
representation-selection pipeline.

## 4. Two supervision experiments — both required

Shared grid: `B = {0, 25, 50, 100, 200, 400}` additional confirmed stream labels.
The common 2,420 confirmed bootstrap labels are excluded from B; tables also show
**continual-learner labels supplied** `2420 + consumed`, not total project labels
(see the separate upstream/evaluation ledgers in §3). A label is charged once even if several request
reasons apply. B=0 means bootstrap-only learning followed by prediction through
the full stream, with no additional updates or class registration.

### Experiment A: identical labeled examples and timing

Build a label-blind nested schedule per stream seed before running any learner.
Implemented recipe under the 2026-09-11 development approval: for each unique stream image ID, hash the UTF-8
string `hdc-shared-v1|20260909|<stream_seed>|<image_id>` with SHA-256; sort by
digest, then image ID to break ties. Use the first B ranked IDs, but deliver their
labels at their original chronological stream positions. Implementation preserves
literal string IDs (no numeric conversion, Unicode normalization or trimming),
rejects empty/duplicate/edge-whitespace IDs, and records ranking/schedule hashes.
The complete evaluation protocol still requires its separate final freeze.

All learners and encoder arms receive those exact B events, including the same
novel-class information. Disable uncertainty acquisition and privileged novel
registration requests. A novel class registers only when one of its scheduled
labels arrives. Do not stratify the schedule using unqueried ground-truth labels.
Small budgets may miss a novel class; disclose this rather than repairing the
schedule after seeing results. This is an offline controlled-supervision design,
not a proposed online acquisition algorithm.

Question answered: given identical supervision, how do the learner configurations
compare? This controls acquisition, but does not alone identify an individual HDC
operation as the causal source of any difference.

### Experiment B: identical allowance, method-specific requests

Retain the existing cosine-margin uncertainty rule at each registered threshold.
Also retain the current ground-truth-assisted request for an unconfirmed novel
class. A successful requested label consumes one unit, updates the learner, and
registers the class when needed. Simultaneous uncertainty and novel reasons still
cost one label. After exhaustion, deny purchases, updates, and registrations;
continue predictions and log potential requests separately from purchases.

Allow underspending. Record allowance, potential requests, successful purchases,
denied requests, unused allowance, both reason flags, purchased class counts,
and registration timing. Ground-truth class counts for unpurchased requests, if
reported retrospectively, must never feed back into the learner or policy.

The oracle can trigger a request using hidden class identity. That is an explicit
simulation assumption, not solved open-set detection. Shared oracle rules do not
guarantee identical registration times when budgets run out differently.

Question answered: under the same allowance and oracle assumptions, how does the
complete acquisition-plus-learning system perform, and what does it spend?

## 5. Metrics and planned comparisons

### Supervisor priorities, endpoint hierarchy, and research questions

The user relayed on 2026-09-10 that the supervisor wants clear indicators of
lower catastrophic forgetting through HDC and higher throughput. This clarification
supersedes the earlier same-day AULC-only primary preference. It is consistent
with the original binding specification's emphasis on retention and update cost,
but the precise new measurements remain proposals until A4 is closed. No historical
result is retrospectively registered or relabeled as confirming these hypotheses.

| Research question | Outcome and hierarchy |
| --- | --- |
| RQ1: Does HDC retain old-class knowledge better under identical supplied labels? | Primary: fixed-population mean old-class recall-based forgetting, paired against NCM, RanPAC-style, naive, EWC, and replay; show initial/final recall and retention curves |
| RQ2: Does HDC process the same continual workload faster? | Primary: cached-embedding continual-head throughput in stream events/second, paired against the same five comparators; update throughput/latency and memory are required supporting measurements |
| RQ3: Are retention and speed accompanied by useful predictive performance and novel-class learning? | Required joint evidence: final held-out Macro-F1, Experiment-A AULC, per-class/stage recall, absolute/relative uptake, final novel-class recall, and never-reached cases |
| RQ4: Under the same allowance, what performance, retention, and throughput does each complete request-and-learning system achieve, and what does it spend? | Required secondary system evidence: Experiment B at every registered threshold, with actual purchases, unused allowance, registration, and upstream-supervision disclosure |

Headline evidence is the paired **frozen-B Experiment-A budget profile** over the
fixed grid, not a test-selected budget. Report both primary outcomes at every
budget and each HDC-versus-comparator contrast; adapted-B remains a required
secondary arm with the same tables. B=0 is a bootstrap-only diagnostic, not evidence
of successful adaptation with no forgetting. No single favorable budget, arm, or
comparator can stand for superiority across the profile. If a scalar primary
summary or a single operating budget is wanted, define it explicitly at A4 before
test access; do not improvise an average or select a test winner.

Low forgetting from an initially weak or non-learning classifier is not success.
Useful retention/throughput claims require the accompanying predictive and uptake
results and pre-test practical criteria. If those criteria are not agreed, report
the numerical tradeoff without asserting acceptable performance or equivalence.

Preserve [common metric semantics](../../common/thesis_eval.py). On each fixed
19-class evaluation population report Macro-F1, macro recall, all per-class
recalls, stage recall, 15-bin ECE, and a fixed-order confusion matrix. Keep all
19 true classes in macro denominators even when some remain unregistered.
Undefined class precision/F1 uses the established zero-division behavior, not
silent omission. Stage recall measures correct stage, not necessarily correct
crop/disease. ECE uses softmax-derived top probability, not the query margin;
do not fit calibration on test. Report prequential and training-pool metrics in
separate tables with explicit population labels.

Required predictive summary: frozen-B Experiment-A normalized Macro-F1 AULC,
with signed pairwise differences HDC−NCM, HDC−RanPAC, NCM−RanPAC. For adjacent
budgets, sum `(B_next - B) * (F1_B + F1_next) / 2`, then divide by 400.
Compute within each paired stream seed before aggregating. Also report every
budget point and both encoder arms; adapted-arm results are a required secondary
comparison, not a substitute headline selected after results arrive.

Experiment B is required secondary system evidence at every threshold: show
performance versus allowance AND actual consumed labels. A trapezoidal area on
its allowance axis must be called **area over allowance**, not efficiency per
consumed annotation. Do not interpolate incomparable actual-spend ranges to
manufacture a common [0,400] label-efficiency score.

HDC-versus-NCM and HDC-versus-RanPAC-style contrasts are mandatory for forgetting,
throughput, and predictive performance. Report HDC versus each gradient comparator
alongside them; retain NCM-versus-RanPAC-style as a required supporting comparison.
Additional per-class/calibration metrics remain required secondary evidence. Do
not choose a favorable threshold, arm, budget, or metric after test inspection.

### Claim-to-evidence decision table

| Possible claim | Required evidence and permitted interpretation |
| --- | --- |
| HDC has higher predictive performance under identical supervision | Held-out Experiment-A AULC comparisons against BOTH NCM and RanPAC-style, with paired seed values, signed differences, uncertainty, and all budget points. A positive paired mean against each supports a scoped observed-average advantage, not significance or superiority at every budget; report inconsistent seeds or intervals spanning zero explicitly. Evidence against only one control cannot establish an advantage over both. |
| HDC has less forgetting than comparator C | Positive paired `100 * (F_C - F_HDC)` percentage-point differences on the fixed old-class population under Experiment A, with all budget points, seed values, conditional uncertainty, initial/final recall, and novel-class uptake. This supports the named comparison only; rolling historical forgetting cannot establish it. |
| HDC has higher throughput than comparator C | Paired `T_HDC / T_C` ratios above 1 for the declared workload/device, accompanied by absolute events/second, repetition variability, and predictive/uptake outcomes. Host update medians alone are insufficient. |
| HDC has both lower forgetting and higher throughput than comparator C | Both outcomes must favor HDC against the SAME comparator, arm, and budget, with uncertainty and the pre-test performance/uptake criteria satisfied. Separate an observed point-estimate win from a robust or practically useful advantage. Report failures and mixed outcomes; do not combine wins over different comparators or budgets. |
| HDC is superior to the prototype controls | The stated joint criteria must hold against BOTH NCM and RanPAC-style in the declared operating conditions. A gradient-only win cannot establish this claim. Even a win over both identifies a configuration advantage, not a causal HDC mechanism. |
| HDC offers a useful operational tradeoff | Joint evidence on predictive performance, actual labels consumed, retention/uptake, synchronized latency, model-state memory, and replay storage. State the supported operating condition and cost/performance compromise. A claim of practical usefulness requires a pre-test use-case criterion or constraint; absent one, describe the measured tradeoff rather than declare a universal winner. |
| Prototype methods broadly perform well | Results across HDC, NCM, and RanPAC-style versus all tested gradient heads (naive, EWC, replay), including matched-supervision evidence and required secondary outcomes. Limit the conclusion to the tested configurations, representation arms, and dataset; it is not a theorem about prototype versus gradient learning. |
| A particular HDC operation causes the benefit | An additional mechanism-specific, controlled ablation that isolates the operation while preserving relevant supervision and other controls. The core comparison alone is insufficient; such an ablation is not automatically added to this work package. |

If differences are uncertain, mixed, or small, say **no clear difference detected
under this protocol**, not **the methods are equivalent**. An equivalence claim
would need predeclared equivalence margins and a suitable analysis with adequate
precision; neither is supplied by a nonsignificant result or an interval spanning
zero. Preserve the modest three-stream interpretation below. Do not backfill
benefit thresholds after seeing results or turn a conditional CI into a universal
ranking. A failed HDC-specific claim may still leave a useful controlled finding.

### Fixed-population forgetting and uptake

Use the same evaluation IDs throughout each curve. During development these are
validation IDs; final locked curves use the test IDs without any model update.
Define t as the number of processed stream events: bootstrap is t=0. Development-
approved (2026-09-11) retention checkpoints are `{0,195,196,390,391,585,586,780,781,976}`, before and
after the canonical zero-based novel introductions at 195,390,585,780.

For the initial 15 known classes, report recall-based forgetting
`max_t recall(c,t) - recall(c,976)` over that fixed checkpoint grid, its mean,
known-class accuracy and macro recall on the fixed old-class subset, and
old-class Macro-F1 on that subset using the fixed 15-class averaging list.
Predictions of novel classes on old examples remain errors. Label this restricted
population clearly; it is not the full-population 19-class Macro-F1. Do not mix
extra uptake checkpoints into the peak used for forgetting. Existing rolling
stream forgetting remains a separate historical diagnostic.

Let `F_m(B,s)` be the mean of those 15 class-wise forgetting values for learner m,
budget B, and paired stream seed s. The reported forgetting-reduction contrast is
`100 * (F_C(B,s) - F_HDC(B,s))` percentage points for each comparator C. Positive
values favor HDC; report absolute forgetting alongside the contrast, not a relative
percentage reduction with an unstable near-zero denominator. Show initial, peak,
and final old-class recall plus full fixed-checkpoint curves. The maximum is over
the declared sampled grid; it is not a claim about every transient dip between
checkpoints. The stream includes old-class refresh labels, so this measures
retention under that protocol, not isolation of interference from new classes alone.

For each novel class, preserve states immediately after its 1st, 2nd, 5th, 10th,
20th, 50th, and 100th purchased labels, where available, plus the final state.
Evaluate recall on a fixed set of examples of that true class. Report the first
observed checkpoint reaching (a) 80% of its final recall and (b) absolute recall
0.80. These targets were approved for development on 2026-09-11, not selected
from validation outcomes; final claim-rule freeze remains A4. Final recall
zero makes the relative target undefined; never-registered and never-reached
cases remain explicit. Unavailable checkpoints are missing, not zero. Include
final recall alongside both targets so weak eventual performance cannot look
like strong uptake. First observed crossing is not exact crossing time or proof
of sustained retention; report any later drop separately.

### Statistics

Report all three paired stream-seed values, mean, sample SD, and median. The
fixed split and fixed learner seed do not create independent dataset or
projection replications. Form contrasts within matched arm/seed/budget/policy.

Propose 10,000 paired, true-class-stratified image bootstrap resamples, seed
20260909, for 95% percentile intervals on final Macro-F1 differences and AULC
differences, and extend the same paired resampling to forgetting differences.
For forgetting, reuse sampled old-class IDs across ALL checkpoints, methods,
budgets, and seed models; recompute each class peak-minus-final and then the
15-class mean inside each resample. Do not bootstrap precomputed checkpoint
means independently. Use identical sampled IDs across methods, budgets, and seed
models for predictive comparisons, then average the three seed-specific contrasts. These are
conditional test-sample intervals for these fitted models, not confidence about
variation across new training streams. Audit plant/session/group dependence
using non-test metadata before freeze; if grouping is required, register a
group-resampling amendment. Do not assert image independence without that audit.
No confirmatory significance claims, family-wise guarantees, or equivalence
claims from an interval containing zero; three streams are limited replication.

Timing uncertainty is separate from image bootstrap uncertainty. Development
approval on 2026-09-13 adopts the bounded policy in
[BENCHMARK_DEVELOPMENT_PLAN.md](../../ssl_hdc_pilot/BENCHMARK_DEVELOPMENT_PLAN.md):
common CUDA placement, host-resident embeddings with transfers included, one CPU
intra/inter-op thread, one disposable full-stream warm-up, separate throughput/
latency/memory passes and five paired rotating blocks for A and B. Final A4 freeze
and full-grid execution remain separate. The existing proposed final default is:
five paired measurement blocks per Experiment-A arm/seed/budget, including every
learner in each block and rotating execution order by a fixed predeclared schedule.
Freeze repetitions, order, warm-up, and failure/exclusion rules at A4 after non-test
feasibility checks. Restore each learner's identical paired bootstrap state and
RNG before each repeat; repeated timings are technical repetitions, not additional
independent training streams. Report every block, per-cell median and range, paired
throughput ratios within blocks, and the three stream-seed summaries. Do not use
test-image bootstrap intervals as throughput intervals or invent significance from
many repeated timers. Required Experiment-B reporting must also freeze its timing
coverage and repetitions before release; its workload includes method-specific spend.

### Throughput, update latency, and state/storage footprint

The proposed target is the **current research machine**, not a specified edge,
mobile, embedded, or CPU deployment environment. Record actual CPU/GPU, RAM,
software versions, dtype, device placement, and thread settings at measurement
time; do not infer current hardware from historical logs. A CPU benchmark is an
optional scope decision requiring separate approval, not an automatic requirement.
The primary throughput comparison uses the same machine, fixed thread settings,
and a common device/placement policy across learners where supported. Freeze the
exact policy before evidence collection; any separately reported mixed-device
best-configuration comparison is a system result, not an isolated learner effect.
No edge-deployment, energy-efficiency, or full image-pipeline throughput claim
follows from a cached-embedding head benchmark.

Measure and label distinct quantities:

- **Primary continual-head throughput:** `976 / elapsed_stream_seconds`, covering
  the complete chronological stream with cached embeddings, prediction, acquisition
  or shared-schedule decisions, registration, and all required learner work. Include
  replay maintenance, EWC consolidation, and transfers inside the declared boundary.
  Exclude bootstrap construction/loading, encoder inference, human response delays,
  evaluation, snapshot serialization, and diagnostic output. Report bootstrap cost
  separately. Specify embedding residency and included transfer costs before freeze.
- **Supporting update throughput:** purchased stream events divided by their total
  update-processing time. One new label event counts once even if replay performs
  32 additional rehearsals; include registration and required consolidation/buffer
  work in its denominator. At B=0, update throughput and per-update latency are N/A.
- **Supporting update latency:** median/p95 and total seconds, separately for
  bootstrap and stream updates. This is not full-stream throughput.
- **Full image-pipeline throughput:** outside the current benchmark. It would need
  a separately specified measurement including image loading/preprocessing and the
  shared encoder; do not infer it from cached-embedding results.

For the throughput pass, synchronize the active accelerator immediately before
and after the full timed region. Do not add per-event synchronization solely for
timing to this pass. Run detailed synchronized per-update latency instrumentation
as a separate pass restored from the same bootstrap state; verify identical event
decisions and final predictions before accepting it. Perform warm-up on disposable
copies, restore states/RNG, and exclude evaluation/snapshot hooks from the throughput
pass. Do not calculate throughput as `1 / median_update_latency`. This separation
follows [PyTorch's benchmarking guidance](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark).

For every named comparator C, report paired `T_HDC / T_C` together with both
absolute throughputs, labels consumed, device, budget, and repetition variability.
Ratios above 1 favor HDC. Present throughput-versus-budget and forgetting-versus-
throughput plots with predictive performance/uptake disclosed at the same conditions;
never join a retention win at one budget to a speed win at another.

Report update count, total update time, and per-update median/p95, separating
bootstrap from charged stream updates. New timing synchronizes CUDA before and
after the update and excludes prediction, encoder work, state saving, and
evaluation. Document transfer boundaries and warm-up treatment before freeze;
never perform extra warm-up updates on scientific states. If there are no stream
updates, report total stream-update time zero and per-update latency N/A. Keep new
measurements distinct from unsynchronized historical host timings.

Measure memory/storage at bootstrap, the fixed retention checkpoints, and final
state, reporting the sampled maximum explicitly (not an unmeasured runtime peak):

- **Prediction-state payload bytes:** projection/item memory, prototypes or
  means/counts, neural weights, and registry/output mapping as applicable.
- **Additional resumable-state bytes:** accumulators not needed for prediction,
  EWC/Fisher state, replay memory, RNG state, counters, and other saved update
  state. Give a total with overlapping tensors counted once, not twice.
- **Replay-buffer storage:** actual occupancy/capacity, embeddings, labels, and
  bookkeeping; disclose runtime object overhead separately from tensor/array
  payload. Count bootstrap-filled buffers even at B=0.
- **Serialized storage:** prediction-only and full-resume artifact bytes,
  including replay. Report logical per-model size and physical deduplicated
  storage separately; shared projection blobs are not free model state.

Keep the shared frozen encoder, cached embeddings, process/runtime overhead, and
temporary update allocations outside the classifier-payload total and explicitly
listed as exclusions. If measured, report process/device peaks separately with
their instrumentation; payload bytes and file bytes are not RAM/VRAM peaks.
Freeze the byte-accounting and instrumentation rules using non-test validation.
These measures support RQ2 and are required for any efficiency claim.
Conduct contamination preflight before authorized evidence runs; no
hardware/quantum speedup claim is established by adding measurement definitions.

Bounded development amendment approved 2026-09-14: measurements on the current
Windows research machine use a recorded shared-desktop baseline, including the
identified Windows Settings process even if it restarts. Ordinary accepted OS
activity is logged, not required to be absent. Process presence is not measured
interference. The subsequent September14 background-warning amendment retains
known browser/game/recording/overlay/Python workload exclusions, but records
other GPU-listed apps as unclassified warnings without aborting. They are not
promoted to the identified desktop baseline or certified idle. Unreadable primary
inventories still abort. Summary and integrity receipts retain all warning
boundaries and qualify warning-bearing results as shared-desktop measurements
with unclassified activity; integrity PASS is not evidence of clean timing.
Retain all five paired blocks and report their variability without selectively
dropping slow successful repetitions. These are not isolated-machine timings.
See ssl_hdc_pilot/BENCHMARK_DEVELOPMENT_PLAN.md for the exact dated policy; this
does not freeze A4 or revise old receipts, scientific endpoints or hypotheses.

## 6. State preservation and historical reconstruction contract

Prediction state must include the class registry/output order and all parameters:
HDC projection, real accumulators AND current bipolar prototypes (tie history
matters); NCM means and counts; RanPAC projection, means and counts; neural
weights and output mapping. A fully resumable snapshot also preserves EWC means
and Fisher state, replay items/order/seen count/private RNG, other RNG states,
stream cursor, query/registration counters, configuration, dtype, and device
metadata. Distinguish prediction-only from resumable artifacts explicitly.
Content-addressed immutable projection blobs may be shared between snapshots.

Historical reconstruction must use the original embeddings, stream events,
thresholds, hyperparameters, query logic, and update ordering. No DINO retraining.
Begin validation with one cell per learner/arm (seed 0, threshold 0.10), then
verify all 108 historical cells before their reconstructed states enter the
held-out release. This extends the TODO's selected-cell check as a proposal.

Require exact sample ordering, labels, prediction identities, query decisions,
charges, and registration events. Proposed numerical tolerance for saved scores
and confidence is `atol=1e-6, rtol=0`; exclude only explicitly listed non-scientific
fields such as timings, new paths, and new provenance. Validate this tolerance
before freeze. A mismatch blocks acceptance and requires investigation, not
quietly relaxed criteria or overwritten historical artifacts.

Historical HDC/naive/replay artifacts retain pre-amendment spec hash
`30447621e0d3701f36a152fb37716f54b6a31cab9f26b118a07c732a6e365812`.
Preserve that provenance separately from the current spec hash; document effective
configuration equivalence for unaffected methods. Recover missing source specs
if necessary rather than inventing a matching hash.

Save new artifacts only in dedicated versioned, non-overwriting directories.
The manifest must identify code revision and patch hash if dirty, protocol hash,
source/effective spec hashes, embeddings and split hashes, event-schedule hash,
learner configuration, environment, state hashes, evaluation roles, and fidelity
report. Save ID-aligned predictions and label-event ledgers sufficient to rebuild
tables; keep large states/embeddings outside Git under the existing ignore policy.

## 7. Development acceptance and test-opening gate

Before test access, implement and verify on synthetic/training/validation data:

- Snapshot/reload prediction equality for all learners; resumable replay equality
  including HDC ties, dynamic registries, EWC state, and replay RNG/buffer state.
- Split guards; no test-derived decisions; evaluator purity and unchanged RNG.
- Exact Experiment-A event equality/nesting; Experiment-B accounting, double
  triggers, underspending, exhaustion, and B=0 behavior.
- Correct fixed-population denominators and hand-checkable metric/AULC/forgetting
  contrasts; explicit missing/never-reached cases; paired longitudinal resampling.
- Throughput boundaries, state/RNG restoration, instrumented-versus-uninstrumented
  event/prediction equality, replay/EWC maintenance accounting, B=0 N/A behavior,
  and paired timing tables for HDC against all five comparators.
- Historical fidelity reports, fail-on-overwrite protection, and complete result
  table/figure templates with comparison directions, captions, and CI meanings.
- Runtime/storage estimates and an approved immutable run manifest. Proposed
  coverage is 108 historical reconstructions, 216 Experiment-A cells, and 648
  Experiment-B cells: 972 logical conditions before verified duplicate reuse.
  Do not confuse those conditions with independent statistical replications.
  Timing repeats and separate latency passes add work beyond these 972 conditions;
  budget them explicitly rather than treating them as free or independent streams.

The §1 amendment checklist is a mandatory part of this gate: A1 (appendix scope
and sequencing), A2 (final-plus-longitudinal release), A3 (972-condition feasibility),
and A4 (endpoint hierarchy, evidence rules, and cost scope) must each have a dated
resolution and approver. Do not treat incorporation of review comments as sign-off.
The feasibility worksheet must budget the new state/storage instrumentation and
the longitudinal/CI analysis, not just end-state learner runs.

Resolve the appendix obligation and longitudinal-test interpretation explicitly.
Then freeze a dated approved protocol, implementation revision, input/state
manifest, and validation/fidelity reports. Approval of this draft alone is not
that final freeze. Keep official commands guarded and user-launched.

The binding specification §6a asks for longitudinal trusted-test retention and
uptake, whereas TODO v2 emphasizes evaluating the final state once. Proposed
reconciliation: **one final analysis release** scores all predeclared immutable
end states and longitudinal snapshots, after every state and analysis rule is
locked. Intermediate test scores never select models or trigger further training.
This is not literally one model invocation; the final-state endpoint and the
longitudinal analyses are separately labeled. If only end-state test evaluation
is approved instead, record an explicit amendment to the earlier requirement.

Do not inspect an early test cell and then revise later conditions. No updates,
queries, tuning, metric redesign, or new checkpoint selection after release.
Technical failures and any rerun justification must be logged; any analysis
invented after test inspection is exploratory, not silently confirmatory.

## 8. Handoff and next authorized step

Current checkpoint: local protocol draft revised on 2026-09-10 to incorporate
the supervisor priorities relayed by the user: forgetting reduction and higher
throughput, explicitly compared against NCM/RanPAC-style and gradient learners.
The earlier AULC-only hierarchy is superseded; A1–A4 remain unresolved. The user
subsequently authorized development steps 1–2: TODO alignment and learner-state
preservation. Those are implemented with synthetic CPU round-trip and runner
compatibility checks, not historical-cell fidelity or held-out evidence. No
official run was launched by the agent. The user subsequently authorized TODO
development until input is needed and explicitly retained manual ownership of
ALL experiments. Historical reconstruction infrastructure and synthetic checks
are now implemented; all 108 historical cells were manually run and passed
observed-output fidelity checks on 2026-09-11.
See [the reconstruction handoff](../../ssl_hdc_pilot/RECONSTRUCTION.md).
Both budget acquisition policies and approved retention/uptake snapshot capture
are now implemented and synthetically verified (2026-09-12: 281 tests plus six
subtests). The user-run A25 training and final-state validation were independently
audited. The first B400/NCM/frozen-B/seed-0/.10 capture is also user-run PASS,
with 27 artifact hashes verified, 400 purchases and 20 states. Separate strict
validation is now user-run and verified: 22 output hashes, 48 input pins and all
20 fixed-population ledgers/derived metrics. This is one development cell, not
comparative evidence. Next decision is the bounded
[benchmark policy](../../ssl_hdc_pilot/BENCHMARK_DEVELOPMENT_PLAN.md); comparative
grids and real benchmarking remain pending. That bounded benchmark development
policy was explicitly approved on 2026-09-13; implementation and synthetic
verification proceed without authorizing agent-launched experiments or closing
A1-A4. See [audit](../../ssl_hdc_pilot/results/B400_VALIDATION_AUDIT_2026-09-12.md).
No real run was launched by the agent. A1-A4 are still unresolved. Quantum and
new datasets remain outside scope.

Draft provenance: repository HEAD `ed437f3f4d41ae228fed01a1003518d90341f5d0`;
current canonical stream-spec SHA-256
`62b545d31d592735be4404451bddd7f603576d127351f337e1e34c8fd04604cc`;
original pre-development user-edited TODO v2 SHA-256
`6c2e4ba0bac9e38893c35d865ab718eae0852af060af561491fd3349f36127ba`.
These identify original inputs to this draft, not a frozen scientific implementation
or the subsequently authorized TODO revision.
