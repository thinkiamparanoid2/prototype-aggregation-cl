# Three-dataset HDC replication — Gate 1 plan

Status: **GATE 1 PASSED FOR ACQUISITION AND EXTRACTION.** Stream specifications are not yet
approved, and adaptation, learner cells, capture, and batch execution remain blocked pending the
post-split feasibility check and Gate 2.

## Sources and licences

| Dataset | Approved source/version | Licence and handling | Acquisition checks |
|---|---|---|---|
| PlantVillage | Mendeley `tywbtsjrjv/1`, Arun Pandian J. and Geetharamani G. | Republication CC0 1.0. Cite it and Hughes & Salathe 2015, arXiv:1511.08060. Use only the without-augmentation archive; never download/open augmentation. | Record archive filename, bytes, SHA-256, download UTC, no-login result. Exclude `Background_without_leaves`; require exactly 54,303 images and 38 classes. Exclude and report any grey/segmented folders. |
| PlantWild v1 | Official PlantWild download page/Hugging Face option, Wei et al. 2024 | CC BY-NC-ND 4.0; non-commercial research only, no image/derivative redistribution. | Require public no-account access; otherwise consult. Record exact source/version, archive receipt and hashes. Expected 18,542 images/89 classes. |
| PlantDoc | `pratikkayal/PlantDoc-Dataset`, Singh et al., CoDS-COMAD 2020 | CC BY 4.0, but source images were scraped from Google Images/Ecosia and third-party copyright is not cleared. Non-redistributive research only; no images in Git, shared archives, or thesis figures. | Preserve the supplied source split. Supplied `test/` images and their exact-duplicate groups are forced into the sealed test; fill the remaining test quota from `train/`. Tag manifest records with `source_split` and prove no supplied-test image enters train, validation, or `splits/all.csv`. |

Every acquisition writes a per-file SHA-256 manifest before splitting. Exact-content duplicates are
grouped before the deterministic split and cannot cross partitions; known near-duplicate checks are
reported. Any access gate, wrong PlantVillage post-filter count/class count, duplicate leakage, or
unexpected class count is a Claude consultation.

## Locked protocol

- Retain classes with at least 30 images; use deterministic content-grouped 60/20/20 splits.
  Write only train/val CSVs plus a sealed-test receipt; never create/read `test.csv`.
- **Amended 2026-09-17 (Claude): spec format requires the literal 0.25 known holdback; novel exclusion via filtered `stream_train.csv`.**
- Known holdback per known class = `floor(0.25 × class_train_count)`, IDs ranked by
  `build_stream_spec._holdback_rank(seed, label, id)` for stream seeds 0-4; H = the (seed-invariant) total.
- `novel_quota = round(0.22 × H)`; `stream_length = H + novel_quota`; budget `round(0.410 × stream_length)`.
- Novel quota is split EQUALLY (`novel_quota // n`, remainder one each in draw order), not proportionally;
  `n = min(max(2, round(0.21K)), floor(Q0/22))`, `Q0 = round(0.22 × Σ_all floor(0.25×count))`, eligible =
  train count ≥ `ceil(Q0/n)`; draw seed 20260917 advances by 1 on any share <22, share over a class's
  train count, or `_prepare_stream` infeasibility on seeds 0-4.
- Excluded novel images (beyond each class's equal share, ranked by `_holdback_rank(20260917, label, id)`)
  are dropped from a per-dataset filtered `stream_train.csv`, never the canonical `train.csv`; excluded
  IDs plus both files' sha256 are recorded in `<Dataset>_stream_feasibility.json`.
- Introductions are evenly spaced; budget is `round(.410*stream_length)`; threshold is 0.10.
  HDC dimension, RanPAC projection dimension, and registered device policy copy the canonical spec.
- Six arms/dataset: frozen and adapted DINOv3 ViT-S/B/L. Adaptation uses only that split's
  train images without labels; probe epochs are 1--5, logistic-regression train/val selection,
  tolerance 0.005. Adapted ViT-S is included although the historical gate selected frozen epoch 0.
- Learners are HDC, NCM and RanPAC-style; seeds 0--4; 270 cells. The budget is a ceiling:
  labels consumed must be <= budget and are recorded per learner. Fewer labels, or different
  learner label counts, are flagged rather than failed and disclosed beside every win/tie/loss call.
  Validation is development-only; no held-out evidence is claimed.

## Post-split feasibility requirement

CONSULT 06 supersedes the earlier all-novel-images feasibility calculation. The canonical registered
stream is 976 events: 800 known holdback plus 176 novel events, an 18.0% novel share. After each real
content-grouped split exists, derive the quota-limited novel count and allocation above, record every
excluded novel-train ID, and call the real `_prepare_stream` for stream seeds 0--4. Advance the draw
seed only by the registered next-seed rule. Every novel class, including the last introduced class,
must have at least 22 post-introduction occurrences on every seed. Stream specifications remain
unapproved and learner cells remain blocked until this check passes and Claude approves the specs.

## Measured resource basis

Archived extraction metadata on 5,767 images gives frozen ViT-S/B/L elapsed times of 72.6783 /
45.1831 / 77.4180 seconds (0.01260 / 0.00783 / 0.01342 seconds/image). Saved NPZ sizes are
8,248,104 / 16,482,925 / 21,981,085 bytes respectively; adapted B/L are 16,463,578 /
22,000,928 bytes. Embedding archives scale approximately linearly with image count and observed
bytes/image; raw data, adaptation checkpoints and extra snapshot scaling are retained separately,
not hidden.

Measured D=1024 synthetic cells project 27.1 CPU hours for the 270-cell study. Original linear output
scaling gives 33.3 GB; correcting the measured fixed HDC/RanPAC non-snapshot files gives 19.5 GB.
D=384 and D=768 are bounded by D=1024 and their embedding archive sizes are projected separately,
not timed.

Existing five-epoch adaptation metadata records per-epoch elapsed times of about 1,924 s (ViT-S),
2,352 s (ViT-B) and 4,088 s (ViT-L) on 51,906 images. The full projected GPU workload is 15.2 hours.
Adapt one dataset × model size at a time before CPU cells. Keep all five epoch checkpoints only until
probe selection and Gate 2 embedding checks pass for that arm, then delete all but the selected one.
The user is adding approximately 66 GB on a second drive for raw archives and selected checkpoints;
G: retains an 8 GiB reserve. Run CPU cells with two workers × six threads, re-time after the first six
real cells, and report if projected wall time exceeds 16.26 hours.

## Gate evidence and execution

Gate 0 is passed and preserved at `gate0_claude_2026-09-17/gate0_comparison.json`: all six
macro-F1/forgetting differences are 0.0. It is not rerun. Gate 1 is passed only for acquisition and
extraction. Stream-spec approval, adaptation, and learner cells remain blocked by the post-split
feasibility check and Gate 2. The batch `-PlanOnly` lists 270 cells in PlantDoc, PlantWild,
PlantVillage order; execution still requires the exact user confirmation. Receipts require PASS plus
every named artifact, three retained attempts maximum, preflight, and the 8 GiB recheck.

## Reporting lock

For every dataset x arm, report validation macro-F1 beside forgetting, per-class F1, accuracy,
ECE, labels and novel-class terminal accuracy. HDC win/loss/tie uses the locked five-seed rule and
paired bootstrap interval versus best control. RanPAC-style is an ablation, not published RanPAC;
five seeds are consistency evidence, not a powered test.
