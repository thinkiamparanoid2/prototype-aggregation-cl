# Supervisor Decision: Rare-Class Policy for the Trusted-Label Split

Status updated: 2026-09-07. Original decision sheet: 2026-07-03.
**Resolved:** supervisor-approved Option A and Option B splits were constructed
July 19. Option A contains 19 classes and 3,396/1,132/1,132 train/val/test
examples; Option B contains 10-repeat, five-fold assignments for all 25 classes.
The Option-B analysis is still unrun. See the
[construction record](../../../ssl_hdc_pilot/RESULTS_LOG.md).
The options below are retained as the original decision rationale, not a
pending request for approval or permission to regenerate the splits.

## Context in Four Sentences

The original train labels (46,713 samples) are generated/pseudo-labels, and the validation set (5,193) was split from that same train set, so it is pseudo-labeled too. The only true-labeled data is the current test set: 5,767 samples across 25 crop_disease_stage classes. We intend to reframe the thesis around honest task performance, which means re-splitting this trusted set into train/validation/test (60/20/20 stratified) and retiring the pseudo-label-trained results to a preliminary-experiments chapter. The blocker: six classes have fewer than 30 samples, so a 60/20/20 split leaves them with only ~2–3 final-test samples each — too few to report credibly.

## The Six Rare Classes (all crop 6)

| Class | Samples | Sibling stage available to merge with |
| --- | ---: | --- |
| 6_a11_2 | 12 | yes (6_a11_1, 40) |
| 6_a12_1 | 13 | yes (6_a12_2, 29) |
| 6_b4_3 | 14 | yes (6_b4_1, 18) |
| 6_b4_1 | 18 | yes (6_b4_3, 14) |
| 6_b5_1 | 21 | no — only stage present |
| 6_a12_2 | 29 | yes (6_a12_1, 13) |

Total: 107 samples, 1.9% of the trusted set. For scale: the four healthy/normal classes hold ~65% of all samples.

## Options

### Option A: Drop classes under 30 samples

Task becomes 19-class with an explicit scope statement ("stages with fewer than 30 trusted examples were excluded").

- Pro: cleanest to defend; every reported class has at least ~8 final-test samples; costs only 1.9% of data.
- Con: thesis no longer covers all crop-6 disease stages; scope reduction must be stated up front.

### Option B: Keep all 25 classes, report via repeated cross-validation

No fixed holdout for rare classes; results come from repeated stratified CV with confidence intervals.

- Pro: full class coverage, honest uncertainty quantification.
- Con: noisier headline numbers; more compute (every model trained k times); no single untouched final test set unless we additionally hold one out, which shrinks rare classes further.

### Option C: Merge sibling stages within the same crop and disease

Merge 6_a11_1+6_a11_2, 6_a12_1+6_a12_2, 6_b4_1+6_b4_3 into per-disease classes.

- Pro: keeps all crop-6 diseases represented.
- Con: partially abandons *stage* classification for crop 6, which is the core task claim; 6_b5_1 (21 samples) has no sibling and stays rare regardless, so merging alone does not solve the problem.

## Recommendation

Option A (drop, 19-class task with scope statement), optionally reporting the six excluded classes in an appendix under Option B-style cross-validation so the information is not lost. Options can be combined; A defines the headline result, B covers the tail as secondary analysis.

## What Happens After the Decision

1. Build stratified trusted splits (fixed seed) from the 5,767 trusted samples under the chosen policy.
2. Extract frozen pretrained-encoder embeddings (image) and IoT features for the trusted splits.
3. Train clean-label classical baselines (linear probe, MLP, fusion) — these become the new headline numbers.
4. Quantum comparison (VQC head / QSVM) only after classical baselines are established; the ~3,460-sample trusted train set is exactly the low-label regime where the quantum comparison is most defensible.

## 2026-07-19 — DECIDED

Supervisor approved the recommended **Option A + B combination** (19-class
headline with fixed 60/20/20 split; all-25-class appendix via 5-fold ×
10-repeat CV). Recorded in the binding handoff §4 the same day. Split
construction cleared.
