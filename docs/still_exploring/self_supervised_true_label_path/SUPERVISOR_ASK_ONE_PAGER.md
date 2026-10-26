# Decision Request: Rare-Class Policy for the Trusted-Label Split

**What we need from you: one approval (or an alternative choice) on how six
very rare classes are handled. This single decision unblocks the entire
supervised phase of the SSL/continual-learning track.**

## Background (3 sentences)

The original training labels of the crop dataset are machine-generated
pseudo-labels; only the 5,767-image test set carries true labels. To keep the
thesis's headline results honest, we have reframed this track around
label-clean evaluation: foundation-model (DINOv3) image representations are
built WITHOUT any labels, and all supervised training/evaluation uses only
the 5,767 trusted images, re-split into train/validation/test (60/20/20,
stratified). The encoder work is already complete; building the trusted
split is the next step, and it requires a policy for the rare classes.

## The problem

The trusted set spans 25 crop-disease-stage classes, heavily imbalanced:
four healthy classes hold ~65% of samples, and **six crop-6 classes have
fewer than 30 samples each** (12, 13, 14, 18, 21, 29 — total 107 samples,
1.9% of the set). Under a 60/20/20 split, each of these classes would have
only ~2–3 final-test images — too few to report per-class results credibly.

## Options

- **A — Drop the six classes (RECOMMENDED as headline).** The task becomes
  19-class with an explicit scope statement. Every reported class keeps ≥8
  test samples; costs 1.9% of the data. Cleanest to defend.
- **B — Keep all 25 classes via repeated cross-validation (RECOMMENDED as
  appendix).** 5-fold × 10-repeat stratified CV with confidence intervals.
  Full coverage, honest uncertainty, but noisier numbers and no single
  untouched test set for the rare classes.
- **C — Merge sibling stages.** Keeps crop-6 diseases represented but
  partially abandons *stage* classification (the core task claim), and one
  rare class has no sibling, so the problem remains. Not recommended.

## Our recommendation: A + B combined

19-class headline (fixed split, one untouched test set) **plus** an appendix
reporting all 25 classes under repeated CV with confidence intervals — so
the rare classes are disclosed with honest uncertainty rather than dropped
silently. We will state explicitly that the 19- and 25-class numbers are
not comparable, and both are reported regardless of outcome
(pre-registered).

## What your sign-off unblocks immediately

The split-construction script is already implemented, tested, and locked
behind this approval. On sign-off we run it once (fixed seed), and the
downstream studies start the same day: per-size frozen-vs-adapted encoder
probes, and the continual-learning comparison (HDC prototype learner vs
CNN fine-tuning/EWC) that forms the track's main contribution.

**If you approve A+B, a one-line reply is enough.** If you prefer B-only,
C, or something else, we will implement that instead — the machinery is
policy-parameterized.
