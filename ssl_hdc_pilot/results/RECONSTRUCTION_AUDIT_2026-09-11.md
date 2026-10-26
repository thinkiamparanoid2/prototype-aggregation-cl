# Historical reconstruction audit — 2026-09-11

Status: **108/108 PASS for training-side observed-output fidelity.**
All experiments were manually launched by the user. The agent performed
read-only saved-artifact verification, not new inference or experiment runs.

## Coverage and verification

- Exact one-to-one run-tag coverage of the original 108 summaries in
  `continual/frozen_vitb16` and `continual/adapted_vitb16_ep3`.
- Six learners × two ViT-B arms × three stream seeds × three thresholds.
- Every report: 976 steps, zero history/summary mismatches, no reported input
  or code changes during execution, `atol=1e-6`, `rtol=0`.
- Independently rehashed all 648 report-listed artifacts: final state NPZ,
  final training predictions, history, summary, manifest and effective spec.
- Rechecked current original source summaries/histories, embeddings, training
  split, canonical specification and implementation files against the manifests.
- All 108 final prediction ledgers have role `trusted_train`, 3,396 rows and
  3,396 distinct IDs each. These are repeated evaluations of the same training
  population, not 366,768 independent samples or held-out predictions.
- The original historical spec hashes remain distinct: pre-EWC-amendment
  `30447621...` for HDC/naive/replay and current `62b545d3...` for EWC/NCM/RanPAC.

## Preserved locations

Relative to `ssl_hdc_pilot/results/reconstruction/`:

| Directory | Cells |
| --- | ---: |
| `2026-09-10_ncm_frozen_s0_t010_attempt01` | 1 |
| `2026-09-11_representative_attempt01/<arm_learner>` | 11 |
| `2026-09-11_remaining96_attempt01/<condition>` | 96 |

Each cell has its own `fidelity_report.json` and provenance manifest. The
96-cell batch also has 96 direct `.log` files and `preflight.txt`, including
the recorded user confirmation. Its full batch directory measured
1,068,464,871 bytes at audit (logs included), not runtime memory.
These directories are protected by non-overwriting tool behavior and recorded
hashes; they are not filesystem write-protected. NPZ states remain Git-ignored
and need separate backup. No source result was replaced.

## Limits and logging

Fidelity excludes explicitly declared wall-clock durations and relocated path
fields, while comparing update counts and scientific outputs. It does not
establish throughput, lower forgetting, matched-budget superiority or held-out
generalization. It is not proof of original embedding/split identity: old
summaries lacked those content hashes. Current hashes pin reconstruction inputs.

The 11-cell transcript ended before confirmation and produced a late
Stop-Transcript error. Its cause remains unresolved; saved artifacts passed
verification independently. The 96-cell batch used direct logs instead.
Its preflight recorded 3% aggregate GPU use and 825 MiB used, with desktop
processes present; no uncontaminated timing or benchmark claim is made.

TODO 1.3, 1.4, C and R2 are complete for historical training-side reconstruction.
Budget Experiments A/B, fixed-population evaluation, timing/storage instrumentation,
validation development, A1-A4 approval and held-out release remain incomplete.

No official run was launched by the agent.
