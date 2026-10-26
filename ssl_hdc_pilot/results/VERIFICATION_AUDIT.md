# SSL-HDC Verification Audit

Status refreshed: **2026-09-07 — reporting remains PROVISIONAL.**

This supersedes the July 20 audit's open-item descriptions, not the historical
experiment record. Current numbers and definitions are in the
[scientific report](SSL_HDC_REPORT.md). Verification used saved summary JSONs,
history CSVs, and current source; no experiment was rerun.

## Resolved findings

| Earlier finding | Current evidence |
| --- | --- |
| No non-gradient comparator | NCM and RanPAC-style prototypes implemented August 2; all 36 runs are committed in `9cfd20a`. The grid now has 108 runs. |
| Replay may contain a defect | August 2 audit verified Algorithm R, alignment, registration, current-sample inclusion, and post-update insertion. No implementation defect was found; registered behavior retained. |
| EWC correction pending | All 18 EWC runs use the task-boundary consolidation fix and decay 0.9. |
| Missing prequential, calibration, and stage metrics | Both-arm, six-learner threshold-0.10 tables are now included in the report, with raw per-seed/per-threshold metrics retained. |
| HDC described as immune to forgetting | Removed. Frozen-B median rolling forgetting is HDC 0.1748, NCM 0.1677, RanPAC 0.1639. |
| HDC described as cheapest | Withdrawn. The report labels saved update durations as host-timer observations; it makes no validated GPU-latency or end-to-end speed claim. |
| All learners improve with frozen features | Corrected: replay's adapted final Macro-F1 exceeds its frozen result at threshold 0.10. |
| ECE described as query-margin calibration | Corrected: ECE consumes softmax-derived top probability. Querying uses the cosine margin. |

## Evaluation findings and remaining limits

**Final-score population.** `stream_runner.py` evaluates `bootstrap + stream`
after learning: 3,396 trusted-training examples, including previously supervised
ones. These are not held-out trusted-test results. Prequential predictions are
made before the current stream update and are reported separately.

**Ranking depends on the measure.** At threshold 0.10, frozen-B final Macro-F1
medians are HDC 0.7720, RanPAC 0.7602, NCM 0.7489. Prequential medians are
HDC 0.7451, RanPAC 0.7566, NCM 0.7504. This does not isolate a universal
HDC-specific advantage.

**Label budgets remain unequal.** Frozen-B stream-label charges are 449 HDC,
484 RanPAC, 439 NCM, 49 naive, 49 EWC, and 39 replay (medians). All methods
also receive 2,420 initial confirmed examples, excluded from those charges.
The earlier audit's suggestion that the neural heads received only roughly
50 updates in total was incorrect: it omitted bootstrap training. Unequal
stream supervision still confounds comparisons, and the existing threshold
sweep cannot establish equal-budget rankings.

**Forgetting definition.** The instrument records peak minus final rolling
accuracy over initially known classes, with per-class windows of 50 encounters.
The early peak can have little support and the evaluated examples change over
time. This is not standard task-averaged held-out forgetting. Both the metric
and its limits must be reported.

**Timing definition.** The host timer surrounds classifier updates, including
bootstrap updates. It excludes encoding, predictions, and final scoring.
There is no explicit CUDA synchronization in that timing path; different
learner groups were collected in separate runs. The old timing figures are
historical observations, not a controlled hardware benchmark.

**Attribution and positioning.** The current result supports prototype-method
strength under the registered protocol. RanPAC is a projection/ReLU/mean
ablation without ridge/decorrelation. Equal-label-budget evaluation,
held-out generalization, seed-level uncertainty, and fuller related-work
positioning remain work for final claims.

## Artifact checks

- 108 summaries and 108 histories, 54 runs per encoder arm.
- Nine runs per learner per arm (three seeds × three thresholds).
- All histories have 976 rows; summaries are non-smoke and final Macro-F1 is finite.
- All final-evaluation sample counts are 3,396.
- Corrected EWC and new NCM/RanPAC use current spec hash `62b545d3…`.
  Preserved HDC/naive/replay use historical hash `30447621…`, consistent
  with the documented EWC-only amendment and unaffected-output preservation.
- This is structural/aggregate verification, not full prediction reproduction,
  revalidation of every historical hash guarantee, or a new test-suite run.

## Continuation

The prototype implementation and replay audit are no longer blockers.
Before running encoder-geometry Study 2, reconcile its original four-learner
preregistration and the unresolved annotation/evaluation protocol in a dated
amendment. A documentation refresh does not silently authorize or redesign
that experiment.
