# Mechanism-ablation plan

Status: prepared only; no official run was launched by the agent. The 120-cell batch launcher
(`scripts/run-mechanism-ablation-all.ps1`) exists and is verified (see "Batch launcher" below);
the user runs it.

All cells use Experiment B, threshold 0.10, allowance 400, `trusted_splits/train.csv` for training and `trusted_splits/val.csv` for evaluation. `trusted_splits/test.csv` is forbidden. Outputs belong only below this directory. Every aggregate report must include per-class F1, recalls for `3_00_0` and `2_00_0`, fixed-population forgetting, and labels consumed. `trusted_val` is a development split; RanPAC-style is an ablation, not published RanPAC; throughput and memory remain measured losses; findings remain PROVISIONAL.

## Locked cells

- Phase 1: HDC, NCM, RanPAC-style × five arms × seeds 0-2 = 45 fresh cells.
- Phase 2: Sign-NCM, Median-NCM × five arms × seeds 0-2 = 30 cells.
- Phase 3: HDC dimensions 1000, 2000, 10000 × five arms × seeds 0-2 = 45 cells.
- Phase 4 is selected only after Phases 1-3; surviving conditions extend seeds 3-9.

The working estimate is 120 cells before Phase 4, about 88 minutes at the observed 44 seconds/cell; this is a planning estimate, not an authorization to run. Verify at least 8 GiB free before every cell. All cells are a later code-provenance set than the archived 36-cell grid, including ViT-B; compare them within this ablation, not across that boundary. ViT-S and ViT-L are ablation-scope only, not approved Experiment-A/B arms.

## Additive extension evidence

The new `sign_ncm` and `median_ncm` exact types are checkpointed without projections. Archived HDC, NCM, and RanPAC snapshots were loaded through the extended adapter and their stored per-array SHA-256 integrity records verified. Canonical default mode remains ViT-B-only; ablation mode is explicit and is recorded in plan/capture metadata.

## User-owned command form

`powershell -NoProfile -File .\scripts\run-mechanism-ablation.ps1 -Arm s_frozen -Learner sign_ncm -Seed 0 -PlanOnly`

No user confirmation has been provided for an execute path.

## evaluate_snapshots.py ablation_mode defect (fixed)

`evaluate_snapshots.py` re-derives a capture's inputs by calling back into
`budget_experiment._prepare` (`capture._prepare` in that module), and previously did not pass
`ablation_mode`, so it always hit the canonical gates in `experiment_a._prepare`
(`unknown learner` for `sign_ncm`/`median_ncm`, arm rejection for `s_frozen`/`l_frozen`/`l_adapted`,
and a `{768}`-only dimension gate). The capture half of a mechanism-ablation cell could pass while
its evaluation could not run at all. The fix threads
`ablation_mode=bool(manifest.get("ablation_mode", False))` into that call, reading the flag back
from the capture's own already-authenticated manifest rather than adding a new CLI surface.
`evaluate_snapshots.py` is not in the capture's pinned `code_hashes` set (only
`experiment_a.py`/`reconstruct_run.py`/`budget_experiment.py` are), so this edit does not
invalidate the archived 36-cell grid.

Verified against the real `b_frozen`/`sign_ncm`/seed-0 ablation capture already on disk at
`.scratch_mechanism_ablation_context_20260915/capture/` (a genuine `status: PASS` receipt with
`ablation_mode: true`): before the fix, re-derivation raised `ValueError: unknown learner`; after
the fix, `evaluate_snapshots --execute` against that same capture produces a `status: PASS`
`integrity_report.json` scoring all 20 snapshots, written to
`.scratch_mechanism_ablation_context_20260915/evaluation/` (outside `results/`, not part of this
plan's own artifact tree).

## Phase-3 --hypervector-dim passthrough (added)

Phase 3 (the HDC dimension sweep at D in {1000, 2000, 10000}) had no CLI path: `hypervector_dim`
previously reached `HDCPrototypeLearner` only from the pinned stream spec, which is locked to
10,000. A narrow, authorized passthrough now exists:

- `budget_experiment.py`: `--hypervector-dim` (int, optional), accepted only together with
  `--ablation-mode` (`parser.error` otherwise).
- `experiment_a.py`: `_prepare(..., hypervector_dim=None)`; locked to `{1000, 2000, 10000}`,
  requires `ablation_mode`.
- `stream_runner.py`: threaded into `_build_learner`/`_run_one`; applies to the `hdc` learner
  only — `ranpac`'s `projection_dim` continues to read `spec.hypervector_dim` unchanged.

D is recorded only as `ablation_hypervector_dim` in the manifest and integrity report, and only
when `ablation_mode` is true — never in `policy` (so `policy_sha256` is unaffected) and never in
the checkpoint context (the 13-key `common` whitelist in `evaluate_snapshots.py`/
`budget_experiment.py` is unchanged). A canonical run with no `--hypervector-dim` produces a
manifest/report with the exact same key set as before this change (verified by direct
before/after comparison of a smoke capture; only path-dependent fields such as `inputs` differ
between runs in different scratch directories, and `policy_sha256` is byte-identical).
`stream_runner.LEARNER_NAMES` and `ABLATION_LEARNER_NAMES` are unchanged.

## Batch launcher

`scripts/run-mechanism-ablation-all.ps1` mirrors `scripts/run-forgetting-grid-all.ps1`: receipt-
based completion (`integrity_report.json` status `PASS` plus every named artifact present on
disk), 3-attempt escalation with nothing deleted/moved/renamed, per-cell 8 GiB free-space guard
that stops the batch, per-cell workload preflight (`benchmark_environment`) that stops the batch
on a block, a single case-sensitive `RUN ABLATION` confirmation, `-PlanOnly` that returns before
the prompt and performs no writes, `--ablation-mode`/`--output-dir` on every generated capture
command (`--output-dir` on evaluation commands too; `evaluate_snapshots.py` intentionally has no
`--ablation-mode` flag per the amendment), learner names read from the installed package
(`ssl_hdc_pilot.scripts.stream_runner.LEARNER_NAMES`/`ABLATION_LEARNER_NAMES`) rather than
hardcoded, and outputs confined to `ssl_hdc_pilot/results/mechanism_ablation/`.

### Run order: Phase 2 -> Phase 1 -> Phase 3

`-Phase` accepts `1`, `2`, `3`, `all` (default `all` = 2 -> 1 -> 3). Phase 2 (Sign-NCM,
Median-NCM: 5 arms x 3 seeds = 30 cells) runs first because it is the decisive phase — it
determines whether Sign-NCM reproduces HDC's collapse resistance without any hyperdimensional
projection, which gates `CODEX_DATASET_REPLICATION_PROMPT.md` and
`CODEX_OPTION_C_CORRUPTION_PROMPT.md`. If the batch is interrupted, the decisive result is
already in hand. Phase 1 (HDC, NCM, RanPAC-style: 45 cells) and Phase 3 (HDC at D in {1000, 2000,
10000}: 45 cells) follow. Total: 30 + 45 + 45 = **120 cells**, verified by direct `-PlanOnly`
execution (121 output lines: one `START` summary line plus exactly 120 cell lines, in the order
phase2 x30, phase1 x45, phase3 x45).

### Batch and resume commands

Plan only (no writes, no prompt):

```
powershell -NoProfile -File .\scripts\run-mechanism-ablation-all.ps1 -PlanOnly
```

Run the full batch (single case-sensitive confirmation, then unattended):

```
powershell -NoProfile -File .\scripts\run-mechanism-ablation-all.ps1
```

Resume after an interruption (same command; completed cells are skipped via
`integrity_report.json` receipts, partial captures are left untouched and escalate to the next
attempt, capped at 3):

```
powershell -NoProfile -File .\scripts\run-mechanism-ablation-all.ps1
```

Single phase or a bounded slice:

```
powershell -NoProfile -File .\scripts\run-mechanism-ablation-all.ps1 -Phase 2
powershell -NoProfile -File .\scripts\run-mechanism-ablation-all.ps1 -StartAt 1 -MaxCells 5
```

### End-to-end verification evidence

- `_prepare()` re-derivation against `.scratch_mechanism_ablation_context_20260915/capture/`
  (real `b_frozen`/`sign_ncm`/seed-0 PASS capture, `ablation_mode: true`): succeeds after the
  Sec.1 fix (`dimension=768`, `bootstrap=1132` rows loaded for validation, `snapshots=20`).
- Full `evaluate_snapshots --execute --preflight-confirmed` against the same capture, written to
  `.scratch_mechanism_ablation_context_20260915/evaluation/` (outside `results/`): produced
  `integrity_report.json` with `status: PASS`, `snapshots_scored: 20`, `state_and_rng_unchanged:
  true`.
- `-PlanOnly` (no `-Phase`): 120 cells, phase2 x30 -> phase1 x45 -> phase3 x45, no directories or
  files written under `results/mechanism_ablation/`.
- `-PlanOnly -Phase 3`: exactly 45 cells, every line carries `dim=`.
- A non-interactive confirmation prompt (no TTY to answer `RUN ABLATION`) fails closed: no
  capture directory is created.

### Sleep / browser preflight note

Every cell runs `ssl_hdc_pilot.scripts.benchmark_environment` (the same workload-contamination
check used by the forgetting grid) immediately before capture and stops the whole batch — it does
not skip the cell — if a blocked process (browser, `python.exe`/`pythonw.exe` stragglers, GPU
compute contamination, etc.) is detected. Before starting an unattended ~120-cell run (documented
estimate ~44 s/cell, ~88 minutes for Phases 1-3, before Phase-4 seed extension), disable sleep and
close browsers/other GPU-using applications on the machine, since the preflight will otherwise
halt the batch on the very first contaminated cell rather than proceeding past it.
