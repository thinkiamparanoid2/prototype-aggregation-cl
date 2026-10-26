# Non-SSL Paths to an Honest Task/Accuracy Claim

Last updated: 2026-07-03

## Purpose

The main note in this folder (`SELF_SUPERVISED_TRUE_LABEL_ALTERNATIVE.md`) proposes self-supervised pretraining as the fix for the pseudo-label problem. This note answers a different question:

```text
If the thesis headline must be honest task/accuracy performance, what are the paths that involve NO self-supervised learning at all?
```

Decision context (confirmed 2026-07-03):

- Train labels (46,713) are pseudo/generated.
- Val labels (5,193) are also pseudo — val was split from the received train set.
- The only trusted pool is the current test set: 5,767 samples, 25 classes, heavy tail imbalance.
- Headline claim chosen: task/accuracy itself, not model comparison.

Shared prerequisite for paths 1–5 below: **trusted train/val/test splits carved from the 5,767 trusted samples, plus a locked tail-class policy.** See `SUPERVISOR_DECISION_TAIL_CLASS_POLICY.md`. This infrastructure is required by any honest-label path, SSL or not.

## Path 1: Plain Transfer Learning on Trusted Splits (Recommended First)

### Idea

Take the existing E0/E1/E2 architectures unchanged — ImageNet-pretrained MobileNetV3 image branch, IoT MLP branch, classical fusion — and retrain them on the trusted train split with true labels. No new methods. ImageNet pretraining already provides the representation that SSL was supposed to provide.

### Why it fits

- Reuses every piece of code, config, and training recipe already built for E0–E3.
- ImageNet weights fine-tuned on ~3,460 clean samples is a completely standard, defensible setup. Small-data fine-tuning of pretrained CNNs is textbook practice, not a weakness.
- The thesis ladder survives intact: trusted-E0, trusted-E1, trusted-E2, trusted-E3 mirror the current structure, so old pseudo-label results slot in cleanly as a preliminary chapter.

### Concrete steps

1. Build trusted splits (after tail policy decision).
2. Re-run the E0 recipe on trusted_train, tune on trusted_val, report on trusted_test.
3. Same for E1 (IoT) and E2 (fusion). Reduce epochs/LR for the smaller data; add stronger augmentation and early stopping to control overfitting.
4. E3/QSVM comparison afterwards on the resulting embeddings.

### Costs and risks

| Risk | Note |
| --- | --- |
| Overfitting on ~3.5k samples | Freeze early backbone layers, augment heavily, use val-based early stopping |
| Lower absolute numbers than the 46k-sample runs | Expected and honest; the claim changes from "big pseudo-supervised" to "small true-supervised" |
| Class imbalance | Weighted loss or balanced sampling; tail policy handles the extreme cases |

### Thesis claim if it works

```text
Supervised multimodal crop disease-stage classification trained and evaluated exclusively on trusted true labels, with ImageNet transfer learning replacing large-scale pseudo-label supervision.
```

## Path 2: Frozen Foundation-Model Features + Small Heads

### Idea

Do no representation training at all. Extract embeddings once from a strong off-the-shelf vision encoder — DINOv2, CLIP, or the existing ImageNet MobileNetV3 — and train only small classifiers (linear probe, small MLP, SVM) on the trusted splits. Someone else already did the pretraining.

### Why it fits

- Cheapest path by far: one embedding-extraction pass over 5,767 images, then head training takes minutes on CPU.
- Frozen features cannot overfit; only the small head can, and small heads are easy to regularize on ~3.5k samples.
- Produces exactly the compact-embedding setting the QSVM/VQC lane wants: embeddings reduce cleanly to 4/8/12/16 dimensions for quantum kernels.
- Linear-probe results are a respected evaluation protocol in their own right.

### Concrete steps

1. Extract DINOv2 (or MobileNetV3 penultimate-layer) embeddings for all trusted samples; cache as `.npz`.
2. Train linear probe, small MLP, SVM-RBF on trusted_train; tune on trusted_val.
3. Fuse with IoT features (raw or lightly encoded) through a small fusion MLP.
4. PCA-reduce fused embeddings for the quantum comparison lane.

### Costs and risks

| Risk | Note |
| --- | --- |
| Frozen features may miss crop-specific detail | DINOv2 is strong on fine-grained plant tasks in practice; compare against Path 1 fine-tuning to quantify the gap |
| Less "model contribution" to present | Frame the contribution as the honest-label evaluation protocol and the classical/quantum comparison, not the encoder |

### Thesis claim if it works

```text
Label-efficient crop disease-stage recognition using frozen foundation-model representations evaluated under a strict trusted-label protocol.
```

## Path 3: Semi-Supervised Learning (Trusted Labels + Unlabeled Pool)

### Idea

Use the trusted labels as the supervised signal from the start, and exploit the ~47k unlabeled images (their pseudo-labels discarded) through consistency training: FixMatch-style confidence-thresholded self-training, mean-teacher, or simple pseudo-labeling with strict thresholds. This is not SSL pretraining — supervision drives training from epoch one.

### Why it fits

- Directly attacks the small-trusted-set weakness of Paths 1–2: 47k unlabeled images are already on disk.
- Semi-supervised learning is a well-cited field with strong published baselines to reference.
- The label story stays honest: generated labels are never used; the extra images enter only as unlabeled data.

### Concrete steps

1. Train a Path-1 supervised baseline first — it is the mandatory comparator.
2. Add FixMatch-style training: weak/strong augmentation pairs on unlabeled images, pseudo-label only above a high confidence threshold (e.g. 0.95).
3. Dedupe `Data/train/` folders against `val.csv` before using them as the unlabeled pool (val folders physically remain inside `Data/train/`).
4. Report the delta over the supervised baseline at multiple trusted-label fractions (25%, 50%, 100%).

### Costs and risks

| Risk | Note |
| --- | --- |
| More moving parts and tuning than Paths 1–2 | Confidence threshold, augmentation strength, unlabeled batch ratio all matter |
| Confirmation bias: model can amplify its own mistakes | Threshold high; monitor pseudo-label accuracy on trusted_val |
| Compute | Each epoch touches the 47k unlabeled pool; heavier than Path 1 |

### Thesis claim if it works

```text
Semi-supervised crop disease-stage learning: trusted labels plus unlabeled imagery, with generated labels fully excluded from training.
```

## Path 4: Clean the Pseudo-Labels and Stay Supervised

### Idea

Keep training on the 46k set, but first audit it: use confident learning (the cleanlab approach) to estimate which pseudo-labels are likely wrong, then filter or reweight them and retrain. Evaluation stays on trusted labels.

### Why it fits

- Preserves the large training set — no small-data problem at all.
- Turns the label weakness into a measurable, reported quantity ("we estimate X% label noise, concentrated in classes Y/Z") instead of an unspoken caveat.
- Cheap: noise estimation runs on out-of-sample predicted probabilities from models already trained.

### Concrete steps

1. Get out-of-sample predictions on all 46k train samples (cross-validated E0 or E2 predictions).
2. Run confident-learning noise estimation; produce a suspected-noise report per class.
3. Retrain E0/E2 on the filtered/reweighted set; compare clean-test performance before vs after cleaning.
4. Optionally spot-check a sample of flagged images by eye to validate the noise estimates.

### Costs and risks

| Risk | Note |
| --- | --- |
| The claim is still not fully clean | Training labels remain machine-generated, just audited; the supervisor may find this insufficient for the headline |
| Noise estimation is itself model-dependent | Report it as an estimate with sensitivity checks, not ground truth |

### Thesis claim if it works

```text
Noise-audited supervised learning: quantifying and mitigating generated-label noise, with all reported performance measured against trusted true labels.
```

Honesty note: of all paths here, this one least changes the fundamental situation. It strengthens the story but does not produce a fully clean supervised claim.

## Path 5: Hand-Label a Gold Set (Optionally with Active Learning)

### Idea

Grow the trusted pool by human annotation. A few hundred to a few thousand images labeled by the student/supervisor (or a domain expert) either enlarge trusted_train directly or specifically fill the rare crop-6 tail classes. Active learning can pick which images are worth labeling: label the samples the current model is least certain about.

### Why it fits

- Directly fixes the two real bottlenecks: trusted-pool size and the sub-30-sample tail classes.
- Active learning is itself a thesis-worthy angle, and label-cost curves connect naturally to the low-label quantum comparison lane.
- Even a small effort (200–500 targeted labels) could lift every tail class above the 30-sample line and dissolve the tail-policy problem.

### Concrete steps

1. Decide who can credibly label disease stages (this is the real gate — stage labels may need domain expertise).
2. Start with the tail: label unlabeled crop-6 images predicted into the rare classes until each class clears ~30–50 trusted samples.
3. If capacity remains, run uncertainty-sampling active learning to extend trusted_train.
4. Record annotator, date, and confidence for every new label in the trusted manifest.

### Costs and risks

| Risk | Note |
| --- | --- |
| Requires domain competence | Wrong human labels are worse than none; stage boundaries may be genuinely ambiguous |
| Slow and not parallelizable with compute | Human hours are the budget |
| Annotator bias | Keep a second-opinion sample; report agreement if possible |

### Thesis claim if it works

```text
Trusted-label crop disease-stage recognition with targeted expert annotation and label-efficient active learning.
```

## Path 6: Switch to a Dataset with Real Labels

### Idea

Abandon the Dacon label problem entirely: move to a public crop-disease dataset with expert-curated labels — PlantVillage, PlantDoc, Cassava Leaf Disease (Kaggle), or paddy disease datasets — and port the E0→E3 ladder onto it.

### Why it fits

- The label problem does not exist there; fully supervised from day one.
- The agriculture story and the whole model architecture carry over.
- Large, well-cited benchmarks make results easy to contextualize against published numbers.

### Costs and risks

| Risk | Note |
| --- | --- |
| Loses the IoT/multimodal branch | Most public sets are image-only; the multimodal fusion contribution — arguably the thesis's most distinctive part — disappears |
| Restarts the data story late in the timeline | New loaders, new baselines, new numbers with a supervisor review looming |
| Disease vs stage | Most public sets label disease type, not disease stage; the task itself would shift |

### Thesis claim if it works

```text
Classical and hybrid quantum-classical crop disease recognition on expert-labeled benchmark datasets.
```

This is the highest-risk option and should only be taken if the supervisor rejects every trusted-split-based path.

## Comparison Summary

| Path | SSL used? | Reuses existing code | Compute cost | Claim cleanliness | Main risk |
| --- | --- | --- | --- | --- | --- |
| 1 Transfer learning on trusted splits | No | Fully | Low | Clean | Overfitting on ~3.5k samples |
| 2 Frozen foundation features | No | Mostly | Very low | Clean | Features not crop-specialized |
| 3 Semi-supervised | No | Mostly | Medium-high | Clean | Tuning complexity, confirmation bias |
| 4 Clean pseudo-labels | No | Fully | Low | Improved, not clean | Supervisor may find it insufficient |
| 5 Hand-label gold set | No | Fully | Human hours | Cleanest | Needs domain expertise |
| 6 New dataset | No | Architecture only | Medium | Clean | Loses IoT branch, late restart |

## How Quantum Fits Every Path

All paths end in the same place the quantum lane needs: compact embeddings plus trusted labels in a low-label regime (~3.5k trusted train samples, or n = 25–500 subsamples). The QSVM/VQC comparison from `../qsvm_quantum_kernel/QSVM_NEXT_STEPS_PLAN.md` plugs in downstream of whichever path is chosen, with one rule: quantum runs only after the classical baseline of that path is established, and always against a matched classical comparator on identical splits.

## Recommended Ordering

```text
1. Lock tail-class policy (supervisor) -> build trusted splits.
2. Run Path 1 (transfer learning) as the default honest baseline.
3. Run Path 2 (frozen features) in parallel — it is nearly free and gives the QSVM lane its embeddings.
4. If Path 1 numbers disappoint, escalate to Path 3 (semi-supervised).
5. Treat Paths 4 and 5 as supplements (noise audit chapter; targeted tail labeling), not replacements.
6. Path 6 only if the supervisor rejects the trusted-split framing entirely.
```

Paths compose: 1 + 2 + a Path-4 noise-audit section + a Path-5 tail top-up is a coherent single thesis, not four separate ones.
