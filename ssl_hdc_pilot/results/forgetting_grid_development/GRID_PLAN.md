# Comparative fixed-population forgetting grid — user-run manifest

Prepared 2026-09-15. No agent-run capture, scoring, benchmark, or test access.
This development grid uses trusted train for capture and trusted validation for
fixed-population scoring; it is not a held-out test or thesis-result release.

Fixed condition: Experiment B, threshold 0.10, allowance 400. Each invocation
captures one state sequence and then scores it on `trusted_splits/val.csv`.
The launcher refuses any existing target. The old completed frozen/NCM/seed-0
capture remains protected by its distinct path; a fresh NCM seed-0 recapture is
included so all 36 grid slots share current code provenance. The confirmation
phrase is `RUN FORGETTING GRID`.

Run from `G:\FOR THESIS\HDC`:

```powershell
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -PlanOnly -Learner hdc -Arm frozen -Seed 0
```

The command above is a read-only plan for cell 1. The execution commands below
are the full manifest. A plan command is
read-only capture plan and never constructs a learner or writes outputs.

The following are the exact 35 execution commands (one command is one cell):

```powershell
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner hdc -Arm frozen -Seed 0
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner hdc -Arm frozen -Seed 1
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner hdc -Arm frozen -Seed 2
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ncm -Arm frozen -Seed 0
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ncm -Arm frozen -Seed 1
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ncm -Arm frozen -Seed 2
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ranpac -Arm frozen -Seed 0
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ranpac -Arm frozen -Seed 1
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ranpac -Arm frozen -Seed 2
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner naive -Arm frozen -Seed 0
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner naive -Arm frozen -Seed 1
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner naive -Arm frozen -Seed 2
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ewc -Arm frozen -Seed 0
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ewc -Arm frozen -Seed 1
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ewc -Arm frozen -Seed 2
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner replay -Arm frozen -Seed 0
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner replay -Arm frozen -Seed 1
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner replay -Arm frozen -Seed 2
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner hdc -Arm adapted -Seed 0
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner hdc -Arm adapted -Seed 1
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner hdc -Arm adapted -Seed 2
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ncm -Arm adapted -Seed 0
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ncm -Arm adapted -Seed 1
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ncm -Arm adapted -Seed 2
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ranpac -Arm adapted -Seed 0
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ranpac -Arm adapted -Seed 1
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ranpac -Arm adapted -Seed 2
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner naive -Arm adapted -Seed 0
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner naive -Arm adapted -Seed 1
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner naive -Arm adapted -Seed 2
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ewc -Arm adapted -Seed 0
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ewc -Arm adapted -Seed 1
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner ewc -Arm adapted -Seed 2
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner replay -Arm adapted -Seed 0
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner replay -Arm adapted -Seed 1
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid.ps1 -Learner replay -Arm adapted -Seed 2
```

**Cell 1 is the first command above: frozen/HDC/seed-0.** Once it passes, measure
its capture and validation directory bytes and confirm whether its capture has a
shared `projection_blobs` dependency. Add the result to the measured-cell-1
addendum in `A3_FEASIBILITY.md`, then re-project the remaining 35 cells. If the
new projection exceeds live G: free space minus 8 GiB, stop; reduce seeds only.

Suggested execution order is frozen first, then adapted, with HDC/NCM/RanPAC-style
before the gradient controls in each seed. The complete manifest is 36 cells.

After every PASS, preserve the capture directory, validation directory, and
unique preflight log. Then run the post-hoc uptake supplement (no model is
loaded or re-scored):

```powershell
$py = '.\.venv-e1\Scripts\python.exe'
$capture = '.\ssl_hdc_pilot\results\budget_longitudinal_development\2026-09-15_grid_b400_frozen_hdc_s0_t010_attempt01'
$evaluation = '.\ssl_hdc_pilot\results\longitudinal_validation_development\2026-09-15_grid_b400_frozen_hdc_s0_t010_attempt01'
rtk proxy $py -m ssl_hdc_pilot.scripts.uptake_repair --source-dir $capture --evaluation-dir $evaluation --train-csv .\ssl_hdc_pilot\trusted_splits\train.csv --output "$evaluation\uptake_repair.json"
```

`RanPAC-style` remains the qualified prototype ablation; do not describe it as
the full published RanPAC method. Keep original validation-snapshot uptake next
to every repaired result.

## Unattended batch

Start once, after disabling sleep/hibernate and closing browsers, games, overlays,
recording apps, and pending Windows Update restarts:

```powershell
rtk proxy powershell -NoProfile -File .\scripts\run-forgetting-grid-all.ps1
```

The single confirmation is `RUN FORGETTING GRID ALL`. Resume after an interruption,
reboot, or power cut with the same command; the machine does not resume automatically
when power returns. The batch writes append-only unique logs and durable
`batch_progress.jsonl` records. An interrupted capture is retained unchanged and the
next run uses attempt02 then attempt03. A fourth incomplete attempt stops for review.
Only the highest completed attempt per cell is selected for analysis; superseded partial
attempts remain preserved and are never silently mixed into results.
