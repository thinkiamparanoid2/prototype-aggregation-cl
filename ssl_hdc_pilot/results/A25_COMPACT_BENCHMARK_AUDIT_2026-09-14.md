# A25 compact benchmark attempt04 — saved-file audit

Status: PASS for bounded development acceptance. Source:
`benchmark_development/2026-09-13_a_frozen_s0_b25_attempt04`.
The user ran the benchmark; the agent read saved files only, without constructing
learners, training, scoring any population or rerunning timings.

Receipt SHA256:501fab5436a395535f1d763431ba9b98be0d52ddaf257d26b7aa543667b52054.
Audit helper `.cache/audit_benchmark_attempt04.py` SHA256:
e470dc4af5845c658d2532f3daeb7905795ffdc00e7c7558b1f24c45ed0f3a58.
The helper reuses the prior attempt02 ledger audit with only its source root
changed, then adds production storage resolution/dependency checks. All original
programs and attempts remain preserved.

## Verified evidence

- All252 artifact hashes and22 input/code/protocol/policy pins matched. Exact
  directory inventory matched the receipt plus the receipt itself; no failure file.
- All205 boundary checks passed the September14 shared-desktop policy. Every
  inventory declares interference_measured=false. Settings appeared at zero
  recorded boundaries in this attempt; real recurrence is not claimed here.
- All60 throughput/latency ledgers matched untimed references:58,560 event rows
  and203,760 final training-pool prediction rows. Every pass consumed the same25
  scheduled stream labels;976 stream events,2420 shared bootstrap labels.
- All30 separate bootstrap latency ledgers contain2420 updates. All six heads
  and five paired blocks are present; summary medians/paired ratios recomputed.
- Memory inventories match the ten declared retention checkpoints per head;
  disjoint tensor-category totals and serialized sizes reconciled.
- All46 compact HDC/RanPAC archives resolve to one verified projection blob.
  All78 resumable archives verified, including1,128 array shape/dtype/content
  hashes. Prediction exports resolved without constructing a learner.
- Required shared blobs are receipt-listed with exact dependency closure; each
  prediction/full-resume dependency-inclusive byte count reconciles. Artifact
  hashes were rechecked after array inspection.

## Timing results — shared desktop, one condition

Frozen-B,seed0,Experiment-A,additional allowance25. Five technical timing blocks
are not five independent training streams. Retain all blocks and their variability.

| Learner | Median stream events/s | Observed block range | Median of block median update latency (ms) |
| --- | ---: | ---: | ---: |
| HDC | 553.06 | 501.91–592.25 | 1.0485 |
| NCM | 1507.45 | 1445.80–1613.33 | 0.2096 |
| RanPAC-style | 1624.39 | 1238.21–1631.92 | 0.2120 |
| Naive | 1557.56 | 1400.67–1626.03 | 2.0131 |
| EWC | 1384.37 | 1301.39–1431.68 | 3.8212 |
| Replay | 1367.65 | 1187.25–1511.15 | 2.9909 |

HDC full-stream throughput is lower than every comparator in every paired block.
Lower HDC update-only latency than gradient heads does not establish higher
full-stream throughput; only25/976 stream events cause paid updates. This pilot
does not support an HDC throughput advantage. It does not measure held-out
predictive performance or comparative fixed-population forgetting.

## Production storage acceptance

Attempt04 totals127,790,310 logical bytes =121.8703MiB =0.119014GiB, excluding
the adjacent86,496-byte log. Attempt02 totals1,408,083,356 bytes: an observed
90.9245% directory-size reduction. This is now a real production-layout result,
not just the earlier isolated prototype. Metadata and logs also differ between
attempts; the directory comparison is not a controlled compression-only study.
No originals were deleted, so it describes avoided future duplication, not space
reclaimed from existing results. Sharing does not reduce live projection memory.

User confirmation14:40:49.7807934+06:00 to receipt/log last write about14:51:02
on2026-09-14 is approximately10.2minutes including guards, setup and untimed work.
This operational duration is not a throughput denominator or guaranteed forecast;
do not attribute differences from attempt02 to storage alone.

Conditional144 six-head benchmark groups at this footprint total17.1380GiB and
roughly24.5hours at this operational duration. This is NOT an all-study bound:
high-budget/B work, uptake snapshots, evaluations, analysis, failed attempts,
backups, staging and appendix costs remain additional/unbounded. A3 stays open.

## Handoff

Attempt04 is complete: do not rerun its command. Preserve failed attempt03 and
completed attempt02 independently; do not pool their timing blocks. Update
broader feasibility and resource ceilings before grid approval. A1–A4 and the
untouched-test release remain unresolved. No official run was launched by the agent.
