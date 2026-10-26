# B400 longitudinal validation audit

User-run source dated 2026-09-12; audit documentation completed 2026-09-13.
Source: `longitudinal_validation_development/2026-09-12_b_frozen_ncm_s0_b400_t010_attempt01`.
Frozen ViT-B / NCM / Experiment B / seed 0 / allowance 400 / threshold .10.
Status: verified development evidence for one cell, not a comparative result.

## Verification

All 22 output hashes and 48 recorded input/source/code/protocol pins matched at
audit time. Twenty ledgers contain the same 1,132 unique validation IDs and true
labels: 22,640 predictions, not independent images. Classification metrics,
stage recall, ECE and confusion matrices reconstruct exactly from every ledger.
Forgetting and all four uptake summaries reconstruct exactly from those ledgers
and the saved purchase events. Each saved before/after state/RNG fingerprint
matches. The audit read saved artifacts only, with no learner loading or scoring.

The training audit already verified 400 purchases from 450 potential requests,
50 denials, all four novel registrations and 27 training artifact hashes.
Supplied learner supervision is 2,820 labels with bootstrap, conditional on
preselected representations, not the entire project's labeling requirement.

## Results

| Quantity | Value |
| --- | ---: |
| Final validation Macro-F1 | 0.7260553027746831 |
| Final macro recall | 0.8108204852601534 |
| Final accuracy | 0.8162544169611308 |
| Final 15-bin ECE | 0.7359850954580327 |
| Old-class population | 1,074 images / 15 classes |
| Initial old-class macro recall | 0.8468223143805748 |
| Final old-class macro recall | 0.8164332207234669 |
| Mean per-class peak-minus-final recall forgetting | 0.04387119555312556 |

Mean forgetting is 4.3871 percentage points, over the ten approved retention
times only. Initial-to-final old-class recall decline is 3.0389 points; these
differ because forgetting uses each class's observed peak. Neither is a measured
improvement over HDC or another comparator. ECE uses the existing registered-class
softmax; its large value is retained, not calibrated away.

| Novel class | Validation images | Labels purchased | Final recall | Relative-target first observed purchase count | Absolute .80 first observed purchase count |
| --- | ---: | ---: | ---: | ---: | --- |
| 3_a9_2 | 22 | 67 | 0.409091 | 50 | Never reached |
| 5_a7_2 | 20 | 1 | 1.000000 | 1 | 1 |
| 3_a9_3 | 8 | 26 | 0.750000 | 10 | Never reached |
| 6_a11_1 | 8 | 4 | 1.000000 | 1 | 1 |

Relative targets are .80 times final recall; reaching one does not establish
high absolute performance. These are first observed milestone crossings, not
exact crossing times or sustained stabilization. No later below-target drop
was observed at the selected uptake checkpoints. Small novel-class populations
limit precision. No HDC comparison, significance or untouched-test claim follows.

## Storage and provenance

Training directory logical file bytes: 3,148,340; snapshot NPZ files total
1,552,437 bytes; final NPZ: 83,618 bytes. Validation directory: 3,593,399 bytes.
Adjacent logs and shared embeddings are excluded. Compressed file bytes are
not live model memory or process/device peaks, and one NCM cell cannot estimate
the full six-learner grid. Get-PSDrive did not return G: free space in this audit
context; final runtime/storage feasibility remains unverified.

Validation log: `longitudinal_validation_development/preflight_run_686aaaf092b54a318f7eb918c019370e.log`.
Earlier training logs remain split between their original preflight log and
`budget_longitudinal_development/b400_recovery_01.log`. Preserve both completed
attempts; no rerun is needed. Later documentation changes do not replace the
original recorded protocol hashes.

Next decision: [bounded benchmark policy](../BENCHMARK_DEVELOPMENT_PLAN.md).
Comparative grids, paired uncertainty, benchmark implementation, feasibility,
A1-A4 and final test release remain open. Overall status remains PROVISIONAL.
