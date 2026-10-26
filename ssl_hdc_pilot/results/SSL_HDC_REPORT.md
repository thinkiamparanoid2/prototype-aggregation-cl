# SSL-HDC Track Report — DINOv3 Features and Continual Prototype Learning

Status: **PROVISIONAL — refreshed 2026-09-07.** The original 72-run grid and
36 NCM/RanPAC runs are complete in this checkout. Prototype coverage and replay
implementation audit are resolved; equal-label-budget evidence and held-out
generalization remain unresolved. This refresh aggregates existing artifacts;
it does not change a run, protocol, or checkpoint.

## 2026-09-16 - Mechanism ablation (120 cells) COMPLETE - collapse resistance is bounded-influence aggregation, not the hyperdimensional projection

User-run mechanism ablation completed all 120 planned cells: Experiment B,
threshold 0.10, budget 400, `trusted_splits/train.csv` for training and
`trusted_splits/val.csv` for evaluation (`trusted_splits/test.csv` was never
opened). Run order Phase 2 -> Phase 1 -> Phase 3. 240/240 capture+validation
integrity receipts PASS, all attempt01, 400 labels consumed in every cell by
every label-matched learner (hdc, ncm, ranpac, sign_ncm, median_ncm). Source:
`results/mechanism_ablation/aggregate_all_120/cells.csv`, `summary.json`,
`PROVENANCE.md`.

**Critical distinction from the tables in §2-4 below**, which is easy to miss:
§2-4 report **training-pool** final scores at unmatched supervision (the
original 108-run grid). This checkpoint and the 2026-09-15 checkpoint
immediately below it report **fixed-population `trusted_val`** scores at
supervision matched to 400 labels for the label-matched learners. The two sets
of numbers are not comparable and must never appear in one table.

Phase 1 (HDC, NCM, RanPAC-style x five arms) exactly reproduces the archived
36-cell grid on the shared frozen-B/adapted-B arms: frozen ViT-B HDC 0.7220 /
NCM 0.7313 / RanPAC-style 0.7406; adapted ViT-B HDC 0.6001 / NCM 0.5275 /
RanPAC-style 0.5243 (median macro-F1, seeds 0-2).

Phase 2 (Sign-NCM, Median-NCM x five arms) is the decisive phase: does an
aggregation rule that never projects into a hyperdimensional space reproduce
HDC's collapse resistance on adapted ViT-B? Adapted ViT-B medians over seeds
0-2, macro-F1 / per-class F1 `3_00_0` / per-class F1 `2_00_0`:

| Learner | Macro-F1 | `3_00_0` F1 | `2_00_0` F1 |
| --- | ---: | ---: | ---: |
| HDC | 0.6001 | 0.4238 | 0.7407 |
| Median-NCM | 0.5925 | 0.4080 | 0.7407 |
| Sign-NCM | 0.5350 | 0.2918 | 0.7407 |
| NCM | 0.5275 | 0.0168 | 0.0625 |
| RanPAC-style | 0.5243 | 0.0168 | 0.0588 |

Under the pre-declared locked decision rule (indeterminate band 0.10-0.30 on
`3_00_0`, adapted ViT-B): Sign-NCM scores 0.2918 - inside the indeterminate
band, recorded as indeterminate for that cell, not a pass or a collapse.
Median-NCM clears 0.30 (0.4080). Sign-NCM clears 0.30 on adapted ViT-L
(0.5094). Nothing tested fell below the 0.10 floor. The pattern across both
of these bounded-influence aggregation rules (median, sign-majority), neither
of which uses a hyperdimensional projection, is consistent with collapse
resistance coming from the aggregation rule rather than from the projection.
This is **not** a claim that the projection "contributes nothing" - see Phase
3 immediately below.

Adapted ViT-L is an independent check on the same question from a different
angle: plain NCM does **not** collapse there (`3_00_0` F1 = 0.5521), and NCM's
macro-F1 (0.6579) beats HDC's (0.6308) on that arm. The adapted-ViT-B collapse
in NCM/RanPAC-style is therefore specific to that one encoder arm, not a
general failure mode of non-HDC prototype learners.

Phase 3 (HDC at hypervector dimension D in {1000, 2000, 10000} x five arms),
adapted ViT-B, macro-F1 / `3_00_0` F1 / `2_00_0` F1:

| D | Macro-F1 | `3_00_0` F1 | `2_00_0` F1 |
| --- | ---: | ---: | ---: |
| 1000 | 0.5374 | 0.3505 | 0.7692 |
| 2000 | 0.5720 | 0.3933 | 0.7407 |
| 10000 | 0.6001 | 0.4238 | 0.7407 |

HDC does not collapse at any tested D, but its macro-F1 rises with D; D=10000
reproduces the Phase-1 HDC numbers exactly on all five arms (the canonical
stream-spec dimension). The projection is doing real work for HDC's own
accuracy - dimension is not inert. What dimension does not supply is the
collapse resistance itself; Phase 2 isolates that to the aggregation rule, not
to the projection.

Best macro-F1 across all 120 cells: **Median-NCM on frozen ViT-B, 0.7420** -
above every HDC cell in the ablation, including frozen-B HDC's own 0.7220.
Median-NCM's implementation stores every confirmed example (an exact running
median); its accuracy figures above are valid evidence, but **no throughput or
memory figure may be claimed for it** - that cost was never measured for this
learner.

Three defects blocked this ablation and are now fixed (verified against real
artifacts, not open): (1) `evaluate_snapshots.py` did not thread
`ablation_mode` into `capture._prepare`, so ablation captures could be
captured but never evaluated (`ValueError: unknown learner`); fixed with a
one-line read of `manifest["ablation_mode"]`. (2) No CLI path existed for the
Phase-3 dimension sweep; fixed with `--hypervector-dim` (requires
`--ablation-mode`), locked to {1000, 2000, 10000}, recorded as
`ablation_hypervector_dim` in manifest/report only in ablation mode. (3)
`aggregate_grid.py` rejected Phase-3 directory names; fixed in commit
`126ea06`. Full test suite at last verified run: 559 passed, 6 subtests
passed.

A draft release protocol for the sealed trusted-test partition exists at
`../../docs/TODO/HDC_TEST_RELEASE_PROTOCOL_2026-09-15.md` - status **DRAFT,
not approved**. It proposes closing gate A2 but requires explicit user and
supervisor sign-off first; `trusted_splits/test.csv` remains sealed.

**Note on framing (labelled, not a decision):** this result - collapse
resistance tracking the aggregation rule rather than the hyperdimensional
projection - points toward a possible reframing of the thesis around
bounded-influence aggregation rather than around HDC/hyperdimensional
computing specifically. This is not a decision; changing the thesis framing
or closing any gate is the user's and supervisor's call, not made here.
Supervisor briefing updated: https://claude.ai/artifact/36wbcTZRNCayj22t6M619q
(Version 4).

All standing qualifiers below carry forward unchanged and apply to this
checkpoint too: `trusted_val` is a development split and this is
development-only evidence, not held-out evaluation, and no test partition was
ever accessed; RanPAC-style is the implemented prototype ablation, not
published RanPAC; throughput and memory remain measured losses unchanged by
this ablation; naive/EWC/replay consumed 17-79 labels against the prototype
heads' 400 and must never share an unqualified table with them; three seeds
support a consistency statement, not a significance test. Status remains
**PROVISIONAL**.

## 2026-09-15 - Comparative forgetting grid (36 cells) COMPLETE - encoder-dependent result

Backfilled 2026-09-16: this checkpoint completed 2026-09-15 and is recorded in
full in `../RESULTS_LOG.md:1408-1485` and `../CHANGELOG.md:702-738`; the
numbers below are copied from those entries, not recomputed, and this report
had not previously carried them.

User-run 36-cell Experiment-B/threshold 0.10/budget 400 grid completed across
both ViT-B arms (frozen, adapted), all six learners, and stream seeds 0-2. All
72 capture and validation integrity receipts are PASS, all attempt01.
Supervision is matched at 400 stream labels for HDC, NCM and RanPAC-style in
every cell; naive/EWC consumed 17-53 and replay 37-79 labels and are **not**
label-matched with the prototype heads.

**This table uses fixed-population `trusted_val` at matched 400-label
supervision. It is not comparable to the §2 training-pool table below**, whose
HDC 0.7720 / RanPAC-style 0.7602 / NCM 0.7489 frozen figures are final
training-pool scores at unmatched supervision. The two must never appear in
one table.

Final Macro-F1 on `trusted_val`, medians over three seeds: frozen-B HDC 0.7220
/ NCM 0.7313 / RanPAC-style 0.7406 (HDC best in 0/3 seeds); adapted-B HDC
0.6001 / NCM 0.5275 / RanPAC-style 0.5243 (HDC best in 3/3 seeds, non-
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

`trusted_val` is a development split and has now been used for development;
this is not held-out evidence and no test partition was accessed. Throughput
and memory remain measured losses, unchanged by this grid. Status remains
**PROVISIONAL**.
## 2026-09-14 - Six-head B400 benchmark development checkpoint

User-run frozen-B/seed0/Experiment-B/allowance400/threshold0.10 benchmark passed
independent artifact/accounting audit:251 artifact hashes,22 pins and all60
throughput/latency pass ledgers verified. All205 workload boundaries retained
unclassified PhoneExperienceHost activity. These are shared-desktop observations,
not measured zero-interference or isolated-machine timings. See
[B400 benchmark audit](B400_BENCHMARK_AUDIT_2026-09-14.md) for all values and limits.

Median full-stream events/s:HDC428.58,NCM1206.89,RanPAC-style1370.35,
naive1441.91,EWC1287.47,replay1430.51. HDC was slower than every comparator in
every paired block. Stream labels consumed:400 for each prototype,53 for
naive/EWC,37 for replay, plus the common2,420bootstrap labels. Equal allowance
does not mean identical acquired supervision. This condition does not support
HDC throughput superiority; faster update-only medians than gradient heads do
not establish faster whole-stream processing. No new held-out predictive or
fixed-population forgetting comparison is established by this benchmark.
The applied research question and report remain PROVISIONAL; no outcome tuning,
selective timing omission, scientific rerun or old-artifact edits by the agent.

## 2026-09-13 - B400 longitudinal validation development checkpoint

User-run NCM/frozen-B/seed-0/Experiment-B allowance400/threshold.10 validation is
verified: 20 states on the same 1,132 images, all 22 hashes and 48 provenance
pins match, all saved metrics reconstruct exactly. Final Macro-F1 .7260553028,
macro recall .8108204853, accuracy .8162544170 and ECE .7359850955. Mean old-class
recall forgetting over the approved ten checkpoint times is .0438711956, or
4.3871 percentage points. Initial/final old-class macro recall: .8468223144 /
.8164332207. Two novel classes did not reach .80 recall at observed uptake
milestones. See [full audit](B400_VALIDATION_AUDIT_2026-09-12.md).

This establishes an operational fixed-population evaluation path, not HDC
superiority, untouched-test generalization or throughput. Small novel validation
supports (8-22 images/class), one stream and one learner limit interpretation.
No comparative conclusions follow from comparing this B400 result with A25:
both label allowance and acquisition policy differ. Historical evidence below
remains unchanged; overall status is PROVISIONAL.

## 2026-09-11 — One Experiment-A validation development cell

The user completed frozen-B/NCM/seed-0 with 25 additional labels (2,445 supplied
learner labels including bootstrap), then evaluated its saved final state on
all 1,132 approved validation examples with the fixed 19-class denominator.
Artifact hashes and saved prediction-derived classification metrics/ECE were
independently verified. Macro-F1: 0.6600447531; macro recall: 0.7225543250;
accuracy: 0.8356890459; ECE: 0.7468861113. Two novel classes remain unregistered.
ECE uses the existing registered-class softmax, with no calibration fitting.

This is development evidence conditional on previously selected representations,
not untouched-test generalization, an HDC comparison, fixed-population forgetting
or validated throughput. No comparator ranking is inferred from one NCM cell.
Historical tables and their evaluation scopes below remain unchanged.

## 2026-09-10 claim assessment — forgetting and throughput priorities

The supervisor priorities relayed by the user are lower catastrophic forgetting
and higher throughput through HDC. These are research targets, not established
conclusions. This note interprets the existing results; it adds no new experiment.

| Comparison | Existing evidence | Permitted conclusion |
| --- | --- | --- |
| HDC versus naive/EWC/replay | Lower median rolling-forgetting proxy across the two arms and three thresholds; shorter recorded host update medians | Favorable observed retention/timing indicators under unequal label expenditure; not matched-supervision forgetting or validated throughput |
| HDC versus NCM/RanPAC-style | At threshold 0.10, frozen-B rolling forgetting is 0.1748 versus 0.1677/0.1639; adapted-B is 0.2425 versus 0.2977/0.3003. Recorded update durations are longer for HDC on both arms | Forgetting ranking depends on encoder arm; no joint retention/throughput superiority over the prototype controls is established |
| HDC final Macro-F1 lead | Final post-stream training-pool scores favor HDC at the displayed threshold | A separate predictive result; it cannot prove less forgetting or higher throughput |

NCM and RanPAC-style are mandatory primary prototype competitors for BOTH desired
outcomes. A win over gradient learners cannot be generalized to those controls.
The [revised protocol draft](../../docs/TODO/HDC_CORE_EVALUATION_PROTOCOL_DRAFT_2026-09-09.md)
proposes fixed held-out retention under identical labels and a paired cached-
embedding throughput benchmark, with predictive performance and uptake disclosed.
Both advantages must hold against the same named comparator at the same condition
to support a joint claim. The report remains PROVISIONAL; historical numerical
tables, rolling metrics, and unsynchronized timer caveats below are unchanged.

## 1. Scope, data, and evaluation

The study compares update mechanisms on the same frozen DINOv3 embeddings:
HDC, online NCM, RanPAC-style prototypes, naive incremental fine-tuning, online
EWC, and reservoir replay. There is no quantum component. The contribution
is an applied controlled evaluation, not a new continual-learning algorithm.

The trusted source contains 5,767 true-labelled images. Option A excludes
107 examples from six rare classes and splits the remaining 19-class set
into 3,396/1,132/1,132 training/validation/test examples. Option-B 25-class
repeated-CV assignments exist, but their analysis is not complete.
Legacy pseudo/generated labels do not supervise this track.

The adapted encoder uses the separate 51,906-image sanctioned pool without
labels, excluding trusted images. The frozen encoder does not use that pool.
This project-level separation is not an audit of foundation-model pretraining
corpora. Small/base/large encoder training and probe gates are complete;
the continual comparison uses frozen ViT-B and adapted ViT-B epoch 3.

The registered simulated class-incremental stream has 976 encounters and
four novel classes. All learners first receive the same 2,420 confirmed
bootstrap examples, which are **excluded from the reported stream-label
charge**. This is not a timestamped longitudinal disease stream.

Evaluation names have distinct meanings:

- **Final Macro-F1:** after streaming, predict on all 3,396 bootstrap+stream
  training-pool examples without further updates. This includes previously
  supervised examples and is **not held-out trusted-test Macro-F1**.
- **Prequential Macro-F1:** predict each stream encounter before deciding
  whether to request its label and update. This measures online behavior
  separately from the final training-pool pass.
- The trusted test partition is not consumed by these continual runs.
  Consequently these summaries do not establish untouched-test generalization.

## 2. Six-learner comparison

Medians over stream seeds 0/1/2 at confidence threshold **0.10**. The full grid
also contains thresholds 0.05 and 0.20; no threshold was reselected here.
ECE is for the final training-pool evaluation.

| Encoder | Learner | Final Macro-F1 | Prequential Macro-F1 | Stream labels charged | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen-B | HDC | 0.7720 | 0.7451 | 449 | 0.7797 |
| Frozen-B | RanPAC-style | 0.7602 | 0.7566 | 484 | 0.7775 |
| Frozen-B | NCM | 0.7489 | 0.7504 | 439 | 0.7595 |
| Frozen-B | Naive | 0.3696 | 0.3147 | 49 | 0.6223 |
| Frozen-B | EWC | 0.3689 | 0.3174 | 49 | 0.6217 |
| Frozen-B | Replay | 0.2344 | 0.2107 | 39 | 0.5541 |
| Adapted-B | HDC | 0.6549 | 0.6581 | 539 | 0.6418 |
| Adapted-B | RanPAC-style | 0.5516 | 0.6034 | 536 | 0.5755 |
| Adapted-B | NCM | 0.5455 | 0.5995 | 525 | 0.5542 |
| Adapted-B | Naive | 0.2571 | 0.3074 | 29 | 0.5534 |
| Adapted-B | EWC | 0.2571 | 0.3074 | 29 | 0.5548 |
| Adapted-B | Replay | 0.2787 | 0.2589 | 74 | 0.5363 |

HDC leads final Macro-F1 on both encoder arms. On frozen-B the lead over
RanPAC is only 0.0118 and over NCM 0.0231 at this threshold; NCM and RanPAC
instead lead HDC on prequential Macro-F1. Across the three registered
thresholds, the frozen-B HDC-minus-RanPAC median-final gap is 0.0038–0.0118.
On adapted-B HDC's final-score lead is larger, about 0.10.

The evidence supports strong prototype-method performance under this protocol.
It does not establish a universal or statistically significant HDC-specific
benefit from three seeds, unequal query budgets, and one representation family.
RanPAC here means fixed Gaussian projection, ReLU, and online class means.
It omits the published method's ridge/decorrelation machinery and must be
described as a **RanPAC-style prototype ablation**, not a full RanPAC benchmark.

All learners query from the same top-1-minus-top-2 cosine-score margin
definition. Their score distributions differ, so equal thresholds produce
different label counts. Query margin is distinct from calibration confidence:
the code computes ECE using softmax-derived `top_probability`, not the margin.
These scores have not been fitted as calibrated probabilities. HDC's high ECE
does not support a calibration advantage.

## 3. Retention, novel-class uptake, and recorded update timing

All values are medians over the same three seeds at threshold 0.10.

| Encoder | Learner | Final rolling known-class accuracy | Forgetting from peak | Stabilized novel classes | Recorded median update ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen-B | HDC | 0.8252 | 0.1748 | 2/4 | 0.988 |
| Frozen-B | RanPAC-style | 0.8361 | 0.1639 | 2/4 | 0.184 |
| Frozen-B | NCM | 0.8323 | 0.1677 | 2/4 | 0.155 |
| Frozen-B | Naive | 0.3459 | 0.6541 | 1/4 | 1.412 |
| Frozen-B | EWC | 0.3483 | 0.6517 | 1/4 | 2.815 |
| Frozen-B | Replay | 0.2944 | 0.6855 | 0/4 | 2.925 |
| Adapted-B | HDC | 0.7575 | 0.2425 | 2/4 | 0.997 |
| Adapted-B | RanPAC-style | 0.6997 | 0.3003 | 2/4 | 0.171 |
| Adapted-B | NCM | 0.7023 | 0.2977 | 2/4 | 0.158 |
| Adapted-B | Naive | 0.3740 | 0.6241 | 1/4 | 1.297 |
| Adapted-B | EWC | 0.3740 | 0.6241 | 1/4 | 2.901 |
| Adapted-B | Replay | 0.3735 | 0.6265 | 0/4 | 2.798 |

Forgetting is the peak minus final value of a curve that macro-averages each
initially known class's accuracy over its most recent 50 stream encounters,
including only classes observed so far. It is not a repeated fixed held-out
task evaluation or standard task-averaged forgetting. Early peaks can be based
on few observations. HDC is not immune to forgetting; on frozen-B the other
two prototype learners have slightly smaller values for this metric.

Stabilization requires three consecutive 20-example class-specific sliding
windows at accuracy at least 0.8. HDC, NCM, and RanPAC stabilize two of four
novel classes at the displayed setting; this is not an HDC-only finding.
Null per-class stabilization values in the raw artifacts mean the threshold
was not reached, not zero-shot uptake.

The timing column reports saved host-timer durations around
`learner.update`, including bootstrap and queried stream updates. It excludes
embedding extraction, prediction/query decisions, and final evaluation.
The runner has no explicit CUDA synchronization around this timer, and the
learner groups were collected in separate runs. These are instrumentation
observations, not validated GPU latency, end-to-end cost, or hardware-speedup
measurements. The former blanket claim that HDC is cheapest is withdrawn.

## 4. Final per-stage recall

These are training-pool post-stream recalls, median over seeds at threshold
0.10. The shared helper measures whether the predicted stage matches the true
stage, even if crop/disease is wrong. It averages samples within each stage,
not disease-class recalls within the stage.

| Encoder | Learner | Healthy (0) | Early (1) | Mid (2) | Severe (3) |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen-B | HDC | 0.8757 | 0.9477 | 0.8954 | 0.8077 |
| Frozen-B | RanPAC-style | 0.8677 | 0.9464 | 0.8661 | 0.8462 |
| Frozen-B | NCM | 0.8643 | 0.9386 | 0.8619 | 0.8462 |
| Frozen-B | Naive | 1.0000 | 0.0458 | 0.6946 | 0.0000 |
| Frozen-B | EWC | 1.0000 | 0.0458 | 0.6946 | 0.0000 |
| Frozen-B | Replay | 1.0000 | 0.0170 | 0.2469 | 0.0000 |
| Adapted-B | HDC | 0.7498 | 0.9477 | 0.7908 | 0.7692 |
| Adapted-B | RanPAC-style | 0.6644 | 0.9516 | 0.7741 | 0.7308 |
| Adapted-B | NCM | 0.6640 | 0.9503 | 0.7699 | 0.6923 |
| Adapted-B | Naive | 0.9577 | 0.0000 | 0.6402 | 0.0000 |
| Adapted-B | EWC | 0.9577 | 0.0000 | 0.6360 | 0.0000 |
| Adapted-B | Replay | 0.9421 | 0.1229 | 0.3891 | 0.0000 |

The class imbalance and restricted severe-stage support must accompany these
numbers. Per-class recall remains in every summary. No held-out clinical or
pre-symptomatic detection claim follows from these stage labels.

## 5. Encoder probe gates and interpretation

Fixed logistic-regression probes fit trusted_train and score trusted_val;
tolerance is 0.005.

| Rung | Frozen validation Macro-F1 | Best adapted validation Macro-F1 | Selected checkpoint |
| --- | ---: | ---: | --- |
| ViT-S | 0.8767 | 0.8719 | Frozen epoch 0 |
| ViT-B | 0.8870 | 0.8908 | Adapted epoch 3 |
| ViT-L | 0.8944 | 0.9044 | Adapted epoch 4 |

Static probe gains do not guarantee stronger continual results. Frozen-B
has higher final Macro-F1 for five of six learners at threshold 0.10;
**replay is the exception** (adapted 0.2787 versus frozen 0.2344).
A change in representation geometry is a hypothesis, not an identified
mechanism. Probe scores use trusted_val and must not be directly ranked
against the final training-pool scores or other tracks' different tasks.

## 6. Baseline corrections and artifact provenance

EWC's original every-update anchor reset made its regularizer ineffective.
The corrected implementation consolidates at genuine novel-class registration.
A dated amendment changes the cumulative Fisher decay from 1.0 to bounded
0.9 while retaining lambda 10.0. All 18 corrected EWC cells are present.

The August 2 replay audit verified Algorithm-R replacement, embedding/label
alignment, class registration, current-sample inclusion, and insertion after
the update. No implementation defect was found. Its registered mean loss over
one current plus up to 32 replay samples can dilute the current sample's
contribution; unequal label supply is another plausible factor. Their relative
effects are not identified by this grid, and no outcome-driven tuning was done.

NCM/RanPAC share the registered stream, seeds, query/registration rules,
embedding inputs, and accounting. They were implemented in `4308d43`; their
36-run artifacts were committed in `9cfd20a`.

Verified inventory: 108 summaries and 108 histories, 54 runs per encoder;
all histories contain 976 rows, all summaries are non-smoke, and all final
Macro-F1 values are finite. EWC/NCM/RanPAC carry current spec hash
`62b545d31d592735be4404451bddd7f603576d127351f337e1e34c8fd04604cc`.
Preserved HDC/naive/replay summaries retain pre-amendment hash
`30447621e0d3701f36a152fb37716f54b6a31cab9f26b118a07c732a6e365812`;
the run log documents preservation of unaffected outputs. Do not relabel
historical hashes or pretend all 108 files use the current hash.

## 7. What remains before final claims

1. **Equal annotation budgets:** compare online performance under a separately
   specified matched-label protocol, with a performance-versus-labels curve.
   The existing gradient gap does not prove the ranking survives alignment.
   Initial bootstrap supervision must be reported as well as stream requests.
2. **Generalization:** retain the evaluation distinction above. Any new
   held-out test use needs a defined evaluation procedure; the current final
   metric is not evidence that this has already been done.
3. **Inference and positioning:** report seed variability and the limits of the
   rolling forgetting proxy; complete literature positioning before submission.
   The applied contribution and RanPAC variant must remain explicit.
4. **Deferred work:** encoder-geometry Study 2 needs protocol reconciliation;
   extra continual encoder rungs, frozen 7B extraction, and the 25-class CV
   analysis have no completed results here.

The [historical summary image](ssl_hdc_summary.png) predates the prototype
controls and corrected interpretation. It is retained as history and should
not be used as the current six-learner summary.

## Sources and reproduction

- [Raw continual summaries and histories](continual/)
- [Results log](../RESULTS_LOG.md) and [runbook](../RUN_COMMANDS.md)
- [Current audit](VERIFICATION_AUDIT.md)
- [Stream runner](../scripts/stream_runner.py) and [metric definitions](../scripts/continual_metrics.py)
- [Prototype implementations](../scripts/continual_prototypes.py)
- [Binding specification and dated amendments](../../docs/still_exploring/self_supervised_true_label_path/DINOV3_HDC_IMPLEMENTATION_HANDOFF.md)

September 7 refresh: saved-artifact aggregation and source inspection only.
No new training, test evaluation, GPU benchmarking, or protocol mutation.
