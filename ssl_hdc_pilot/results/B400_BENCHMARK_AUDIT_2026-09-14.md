# B400 benchmark attempt01 - saved-file acceptance

Status: PASS for development artifact integrity, with recorded shared-desktop
warnings. User-run frozen ViT-B / seed0 / Experiment B / allowance400 /
threshold0.10 / six learners / five paired blocks is COMPLETE. Do not rerun.

Source: benchmark_development/2026-09-14_b_frozen_s0_b400_t010_attempt01.
The agent read and reduced saved files only: no learner construction, inference,
training, held-out scoring or benchmark launch. Existing outputs were not changed.

## Independent checks

- All251 receipt-listed artifact SHA256 hashes matched, plus all22 current
  input/code/protocol/policy pins. Directory membership matched the receipt plus
  the receipt itself; no failure.json exists. Source hashes rechecked after audit.
- Six learners, five rotating paired blocks,60 throughput/latency passes;
  58,560 saved event rows and203,760 saved training-prediction rows matched each
  learner's untimed reference (discrete exact; floating values absolute1e-6).
  Each reference has976 events and3,396 unique training prediction IDs.
- Paid registrations, per-method purchases and unused allowances reconciled.
  All30 bootstrap timing ledgers contain2,420 updates. Timing summary fields,
  median rates/update latencies and paired throughput ratios recomputed from logs.
- All205 boundary inventories have no known blocked workloads. All205 recorded
  PhoneExperienceHost.exe from the installed YourPhone package as unclassified.
  Summary/receipt warning lists and counts exactly match boundary records.
  Qualification: shared_desktop_with_unclassified_activity. Neither actual
  interference nor its absence is established. All blocks retained.
- All78 resumable archives and1,182 array hashes/shapes/dtypes verified without
  constructing learners. All46 compact archives resolve to one verified shared
  projection blob. Ten retention states/head, payload sums, per-file/dependency
  closure sizes and whole-run logical byte accounting reconcile.

## Observed performance and supervision

| Learner | Stream labels consumed | Median events/s | Five-block range | Median of block update medians (ms) |
| --- | ---: | ---: | ---: | ---: |
| HDC | 400 | 428.58 | 395.14-461.70 | 1.0300 |
| NCM | 400 | 1206.89 | 1172.71-1400.51 | 0.2841 |
| RanPAC-style | 400 | 1370.35 | 1254.90-1440.07 | 0.2078 |
| Naive | 53 | 1441.91 | 1330.16-1493.58 | 1.6238 |
| EWC | 53 | 1287.47 | 1128.50-1359.40 | 2.8737 |
| Replay | 37 | 1430.51 | 1282.78-1524.73 | 2.5732 |

Add2,420 bootstrap labels: supplied learner totals are2,820 for HDC/NCM/RanPAC,
2,473 for naive/EWC and2,457 for replay. Upstream encoder development/selection
supervision is separate. Experiment B equalizes allowance, not purchased IDs:
even the three400-purchase methods need not receive identical examples.

HDC has lower full-stream throughput than every comparator in every paired
block. Its faster update-only median than gradient heads does not reverse that
full-stream finding. This condition does not support an HDC throughput advantage.
Technical repeats are not independent streams. These are cached-head timings,
not encoder-inclusive, isolated-GPU or deployment measurements. No new held-out
predictive, forgetting or uptake comparison is established by this benchmark.

## Resource observation and next gate

Whole attempt:129,973,339bytes =123.9522MiB. Adjacent user-launch log:
preflight_run_d82d139e78a242d4ae5f49bc934f03ee.log,229,122bytes.
Confirmation2026-09-14T22:47:16.0451394+06:00; receipt last-write
22:57:42.222709+06:00. Approximately626.18seconds (10.44minutes) includes setup,
guards, reference, memory, inventory and I/O; it is not the throughput denominator.

Read-only G: free-space observation:72,712,364,032bytes (67.72GiB).
Conditional144-group scaling at this pilot footprint is17.43GiB/~25.05hours;
36 groups at compact A25 plus108 at this B400 footprint give17.36GiB.
These are benchmark-only scenarios, not a full-study forecast or upper bound.
Capture/uptake checkpoints, evaluation predictions, analysis, preserved failures,
staging, backups and appendix costs remain additional. A3 stays open pending
whole-study storage planning; A1/A2/A4 and untouched-test release remain open.
No further experiment is requested merely to revalidate this completed pilot.

## Provenance

Receipt SHA256:404542877f019b6480bf78aff951e580153bec876c8f2e3533785dc7a224f888.
Launch log SHA256:1c99021046846294b9d2aaabc94e5f373980fb8684db6d3edddc1e128aaaa775.
Read-only audit helper:.cache/audit_benchmark_b400_attempt01.py;
SHA256:06539c82494a26f9e656b0ceb52ed13f282674eeada11355005e4af54ad6aeba.
The helper is local-only; all conclusions above derive from preserved source files.
