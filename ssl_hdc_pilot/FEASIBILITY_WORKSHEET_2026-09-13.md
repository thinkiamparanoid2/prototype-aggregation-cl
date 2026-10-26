# Core-study feasibility worksheet - 2026-09-13

Status: preliminary measured-input worksheet; A3 remains OPEN. No full grid
authorized. No runs, artifact deletions, relocations or storage rewrites made.

## Latest observation - B400 pilot accepted, 2026-09-14

The user has completed the prepared B400 benchmark; do not rerun its launcher.
See results/B400_BENCHMARK_AUDIT_2026-09-14.md. Independent checks verified251
artifact hashes,22 pins and all60 timed passes; all205 background-warning
boundaries remain disclosed. No agent-run benchmark or artifact rewrite.

Whole attempt129,973,339bytes (123.9522MiB), plus229,122-byte adjacent log.
Operational confirmation-to-receipt duration626.18seconds (10.44minutes), not
the timed-stream denominator. Read-only current G: free72,712,364,032bytes
(67.72GiB), not guaranteed future availability. HDC/NCM/RanPAC consumed400labels;
naive/EWC53,replay37. This is not the maximum-update case for every learner.

Conditional144-group benchmark-only scenario at this footprint:17.43GiB and
25.05hours. Using36 groups at compact A25 plus108 at B400 gives17.36GiB.
Neither is a validated all-study bound; adapted-arm acquisition, capture/uptake
snapshots, evaluation/analysis, preserved failures, staging, backups and appendix
remain additional. The high-budget pilot is complete, but broader4.4a/A3 remains
open. Next planning needs a designated backup location/capacity and total output
forecast, not another run of either completed pilot. Runtime cap remains removed.

Preparation and earlier measurements below are dated history, not current launch
instructions or claims that B400 attempt01 is still absent.

## Runtime decision - 2026-09-14

Asked for a maximum total machine time, the user answered "as long as is needed".
Record no user-imposed runtime cap; runtime measurement/forecasting remains useful
for scheduling, but a numerical time ceiling is no longer a blocking decision.
This does not authorize agent-launched experiments, a full grid, unlimited disk,
new hardware/cloud spending, or bypassing integrity/resource guards.

Next bounded feasibility pilot prepared within the existing A/B benchmark policy:
frozen-B,seed0,Experiment-B,allowance400,threshold0.10,all six learners/five paired
blocks. Actual purchases may differ by learner and can be below400. It measures
one higher-update condition, not a proven worst-case bound across arms/seeds/
thresholds. Fresh manual launcher:scripts/benchmark-b400.ps1 -Attempt 1. The
read-only input plan passed; no model was constructed or experiment launched.
Use the current G: artifact location and existing8GiB prelaunch headroom check;
that check is not an agreed whole-study reserve/backup budget or mid-run guarantee.
Preparation check: G: free73,537,986,560bytes (about68.49GiB), and the proposed
B400 benchmark attempt01 does not exist. Availability is a point-in-time reading;
the launcher checks again. No deletion/relocation was performed to obtain it.

## Latest measured update - production attempt04 accepted, 2026-09-14

User-run compact A25 six-head attempt04 passed the independent
[saved-file audit](results/A25_COMPACT_BENCHMARK_AUDIT_2026-09-14.md).
Total127,790,310bytes =121.8703MiB =0.119014GiB, excluding its86,496-byte adjacent
log. Observed directory-size reduction against attempt02:90.9245%. One actual
projection blob serves46 compact archives; receipt dependency sizes reconcile.
No originals were removed. This supersedes the prototype-only acceptance status
below, not the preserved original measurements.

Operational confirmation-to-receipt time was about10.2minutes, including guards,
setup, reference and untimed passes. Conditional144-group benchmark scaling is
17.138GiB and roughly24.5hours at that one pilot's footprint/duration. Storage
and environment/policy differ from attempt02; do not attribute elapsed-time
differences to storage alone. Neither scenario is a validated full-study bound.
High-budget/B work, capture/uptake snapshots, evaluation/analysis, failed attempts,
backup/staging and appendix costs remain additional; A3/resource approval stays open.
The previously reported~71GiB free-space envelope is not a guaranteed future
allocation. Runtime cap is now removed by the decision above; reserve/backup
allowance and a complete storage projection remain unresolved.

## Update - shared-projection test, 2026-09-14

User freed space; live start-of-test availability is now 71.0974 GiB on G:.
The bounded [storage test](results/PROJECTION_STORAGE_TEST_2026-09-14.md) PASSED:
46 saved checkpoints, 1,066 arrays/metadata entries reconstructed exactly from
one shared projection plus deltas; all 252 original files unchanged.
Equivalent pilot layout is 127,790,994 bytes (0.1190 GiB), a 90.9245% reduction.
The same conditional benchmark-only 144-group scenario becomes 17.14 GiB rather
than 188.84 GiB. Other outputs/backup/high-budget costs remain unbounded.
Those sizes measure the isolated prototype, not a production pilot. The user
subsequently approved integration: new budget/benchmark save/load now supports
versioned run-local shared blobs, including prediction exports. At that checkpoint,
production-layout acceptance was pending; it is now independently measured in
attempt04 above. The prototype and real-run receipts remain distinct evidence.
Originals remain intact; no disk space reclaimed. A3 and runtime ceilings remain
open. The original measurements/scenarios below are retained as dated history.

## Measured inputs

Source: accepted frozen-B/seed0/A25 six-learner benchmark attempt02; see
[audit](results/A25_BENCHMARK_AUDIT_2026-09-13.md).

- Six learners, five paired timing blocks; separate reference/warm-up, bootstrap
  latency, memory and state-inventory passes. All outputs preserved.
- Total attempt directory: 1,408,083,356 logical bytes = 1.3114 GiB, excluding
  its adjacent 86,320-byte log. This is artifact storage, not live RAM.
- User confirmation: 17:21:25.802789 +06:00; final receipt/log last write roughly
  17:32:43. About 11.3 minutes elapsed after confirmation. This timestamp-derived
  operational duration includes guards, setup, I/O and untimed passes; it is
  not a measured throughput denominator or a guaranteed scheduling duration.
- G: available bytes at read-only check: 33,005,576,192 = 30.7388 GiB.
  Existing launch guard requires 8 GiB free before starting a pilot; if that
  amount is retained as a reserve, remaining growth room is about 22.74 GiB.
  Neither number is an approved whole-study storage ceiling.
- Existing data, environments, completed attempts and artifacts are preserved.
  Free space can change; remeasure before any future launch.

## What occupies the pilot

| Learner directory | Logical bytes | NPZ bytes |
| --- | ---: | ---: |
| HDC | 667,640,096 | 658,006,658 |
| NCM | 10,884,226 | 1,244,944 |
| RanPAC-style | 676,484,082 | 666,845,652 |
| Naive | 11,385,016 | 1,754,278 |
| EWC | 11,940,280 | 2,311,201 |
| Replay | 28,984,165 | 19,355,044 |

Each head directory contains 41 files. HDC/RanPAC NPZs together account for
about 94% of total pilot bytes. Repeated checkpoint/projection storage is the
first future-layout optimization candidate, not permission to delete old NPZs.
The protocol permits content-addressed immutable projection sharing. These are
the pre-integration measurements; see the dated update above for current status.

## Explicitly conditional scaling scenario

The proposed 972 logical conditions comprise 108 completed historical
reconstructions + 216 Experiment-A cells + 648 Experiment-B cells. Do NOT
schedule the completed 108 again. Benchmark grouping is six learners:

| Proposed budget benchmark coverage | Six-learner groups | Pilot-rate time scenario | Pilot-layout storage scenario |
| --- | ---: | ---: | ---: |
| A: 2 arms x 3 seeds x 6 budgets | 36 | about 6.8 hours | 47.2 GiB |
| B: same x 3 thresholds | 108 | about 20.3 hours | 141.6 GiB |
| A+B | 144 | about 27.1 hours | 188.8 GiB |

Arithmetic: groups x approximately 678 seconds; groups x 1,408,083,356 bytes.
These are workload/layout scenarios, NOT validated forecasts, upper bounds or
proof that each grid cell costs the same. B400 has many more updates than A25;
budget, acquisition, arm, checkpoint count and filesystem compression can alter
cost. Do not multiply five repeats again: this pilot already contains them.
Do not subtract development pilot reuse until a frozen manifest permits it.

Excluded/additional work: budget-training longitudinal/uptake capture not covered
by benchmark retention inventories, saved-state validation/test predictions,
tables/plots and paired longitudinal uncertainty resampling, the appendix,
preserved failures, retries, backup copies, staging/working space and final
release packaging. These still need estimates; no all-in thesis time is claimed.
Shared-process/CUDA sampled peaks in the audit are single-condition observations,
not independent per-head RAM requirements or high-budget working-memory bounds.

## Decision and next work

The historical unchanged-layout scenario exceeded the then-observed free space.
The compact A25 layout is now accepted, but timing/storage are not yet bounded
across high budgets/B and all ancillary outputs. Choose a resource envelope before
authorizing a full grid:

1. Runtime resolved:no user-imposed cap. Keep current G: artifact placement for
   bounded pilots; measure available capacity and settle whole-study reserve/backup
   allocation before final grid approval. Manual launch ownership remains.
2. Then prepare an explicit plan using the verified compact future checkpoint
   layout, additional storage, or a dated scope amendment. Preserve existing artifacts;
   do not silently drop conditions, comparators, seeds or adverse outcomes.
3. Completed: hash-verified shared storage, synthetic round trips and real A25
   production acceptance. No deletion/rewrite of completed attempts.
4. Bound high-budget/B runtime and storage with an approved manual pilot plan;
   a favorable A25 result alone cannot close A3. Include analysis/backup/appendix
   costs before approval, then freeze an immutable run manifest.

Runtime-ceiling input is resolved. Storage forecasting/reserve/backup and A1, A2,
A4/untouched-test release remain independently unresolved. Preparation
of this worksheet is not protocol approval or a scientific result.
