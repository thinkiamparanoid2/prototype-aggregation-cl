# A25 benchmark development audit - 2026-09-13

Status: PASS for bounded saved-artifact integrity and first-pilot acceptance.
Not a final scientific release or full-grid feasibility approval.

Source: benchmark_development/2026-09-13_a_frozen_s0_b25_attempt02.
User-run, frozen ViT-B / seed0 / Experiment A / 25 additional labels;
six learners, one discarded warm-up and five rotating paired blocks.
RTX 5070, Ryzen 5 3600, Windows shared-desktop baseline, host embeddings,
CUDA learner placement, Torch intra/inter-op and OMP/MKL/OpenBLAS threads1.
The agent audited saved files only; no model loaded or experiment rerun.

## Independent saved-file checks

- All 251 report-listed artifact hashes match; directory coverage is exact
  including its separate integrity receipt. All 21 current input/code/policy
  pins match. No failure.json in attempt02.
- All 205 recorded workload checks have empty blocked-process/other-GPU lists.
  Accepted desktop processes and verified Search-owned WebView ancestry are
  retained. Boundary inventories do not prove zero background interference.
- All 60 raw throughput/latency passes match the summary and untimed reference:
  58,560 event rows and 203,760 final training-prediction rows compared.
  Exact discrete fields, absolute score tolerance1e-6; no relative tolerance.
- Each reference contains 976 events, the same 25 scheduled purchases, and
  3,396 unique final training prediction IDs. Every pass reports 25 purchases.
- Recomputed full-stream rates, medians, paired ratios and per-block median
  update latency from raw durations. Throughput has no update-latency samples;
  latency passes have 25 samples and no full-stream elapsed metric.
- All 30 bootstrap-latency receipts contain 2,420 samples and report fidelity
  PASS. Saved fingerprints were produced by the run, not re-executed by audit.
- All six memory ledgers match the summary, cover ten approved retention times,
  have disjoint reported tensor accounting, and match actual prediction/resume
  NPZ sizes. Audit does not independently remeasure live memory peaks.
- Logical bytes before final receipts reconstruct exactly: 1,407,390,039.
  Whole attempt directory: 1,408,083,356 logical bytes. Reported filesystem
  allocation is 1,407,845,696 bytes before final receipts; it was not remeasured
  by this audit. Content deduplication is hypothetical, not performed.

## Observed timing

| Learner | Median stream events/s | Five-block range | Median of block median update latency (ms) |
| --- | ---: | ---: | ---: |
| HDC | 532.34 | 408.41-595.12 | 1.494 |
| NCM | 1628.22 | 1167.73-1633.46 | 0.208 |
| RanPAC-style | 1183.00 | 738.34-1632.74 | 0.205 |
| Naive | 1313.13 | 1056.04-1639.84 | 2.070 |
| EWC | 1473.56 | 1104.74-1515.73 | 3.235 |
| Replay | 1386.09 | 1078.53-1480.89 | 2.942 |

HDC full-stream throughput was below every comparator in all five paired
blocks. This condition does NOT support an HDC throughput advantage.
Its update latency was lower than the three neural controls, but higher than
NCM/RanPAC-style; that is not the same as better full-stream throughput.
Only 25 of 976 events trigger updates in this condition. Do not substitute
the update-only ranking for the preregistered whole-stream outcome.

Five timing blocks are technical repeats of one condition, not five independent
training streams. Dispersion is substantial; retain all successful blocks.
No condition, repeat or comparator is discarded because HDC ranks poorly.
No generalization, retention, hardware/deployment or universal ranking claim.

## Handoff

Attempt02 is completed provenance: do not rerun it. Failed attempt01 and logs
remain untouched. First benchmark acceptance is complete, but the 972-condition
runtime/storage/analysis feasibility decision is not. This one A25 condition
cannot establish total cost across budgets, arms and acquisition policies.
Next: feasibility planning and validation table/figure/statistical integration,
then resolve A1-A4 before full-grid/final-test release. Overall PROVISIONAL.
