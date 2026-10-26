# Self-Supervised True-Label Alternative Path

Last updated: 2026-07-03

## Purpose

This note defines an alternate thesis path if we decide that the current pseudo-labeled training setup is too risky for the final story.

The idea is simple:

```text
Use all images without labels for representation learning, then use only trusted true labels for supervised training and evaluation.
```

This is not the current main path. It is a possible restart or fallback path if we want a cleaner label-quality story.

Related exploration notes:

- `docs/still_exploring/quantum_advantage_pivot/QUANTUM_ADVANTAGE_PIVOT_STARTER.md` defines the broader quantum-advantage search space.
- `docs/still_exploring/qsvm_quantum_kernel/QSVM_NEXT_STEPS_PLAN.md` defines the QSVM experiment harness. Under this alternate path, QSVM should use SSL embeddings and trusted true-label splits, not pseudo-label-trained E2 features.

## Very Short Thesis Story So Far

The project currently studies crop disease-stage classification on the Dacon crop dataset using image, IoT/tabular, multimodal, and hybrid quantum-classical models.

Current model ladder:

| Stage | Model family | Role |
| --- | --- | --- |
| E0 | Image-only MobileNetV3 | Visual baseline |
| E1 | IoT/tabular MLP | Sensor baseline |
| E2 | Classical multimodal fusion | Strong classical fusion baseline |
| E3 | Hybrid quantum-classical VQC head | Quantum fusion attempt |

The strongest predictive direction so far has been the classical E2-style fusion and matched classical bottleneck baselines. E3 has been useful for controlled comparison, calibration/robustness analysis, and quantum-layer investigation, but it has not yet produced a clean predictive win over the best matched classical comparator.

A key data concern is now explicit:

```text
The original train labels were generated/pseudo-labels, not true ground-truth labels.
```

That means the current supervised training story depends on pseudo-label quality. If validation/test labels are true labels, then the current pipeline can still evaluate generalization against trusted labels, but the training signal itself is not fully ground-truth supervised.

## Why This Matters

If the thesis claims standard supervised crop disease-stage learning, pseudo-labeled train data weakens the claim. It may still be valid if described honestly, but it changes the framing:

```text
Current framing risk:
Models are trained on generated labels and evaluated against trusted labels.
```

That is not automatically invalid, but it should not be hidden. It becomes a semi-supervised or pseudo-supervised learning setup, not a clean fully supervised setup.

If we want to avoid that issue, the alternate path is to stop using pseudo-labels for supervised learning.

## Alternate Path: Self-Supervised Pretraining + True-Label Fine-Tuning

Use all available images as unlabeled data, then use the trusted true-labeled set for supervised fine-tuning and final evaluation.

High-level pipeline:

```text
All images, no labels
        -> self-supervised image encoder pretraining
Trusted true-labeled subset
        -> supervised classifier/fusion fine-tuning
Held-out true-labeled subset
        -> final evaluation
```

This shifts the project from:

```text
supervised learning on pseudo-labeled train data
```

to:

```text
self-supervised representation learning plus supervised evaluation on true labels
```

## How To Use The True-Labeled Test Set

If the current test labels are true labels, they can be used as the trusted label source. But once we use them for training or validation, they are no longer a final test set.

Correct procedure:

1. Take the current true-labeled test set.
2. Rename it conceptually as the trusted labeled dataset.
3. Create a new stratified split from it.
4. Train/fine-tune only on the new trusted train split.
5. Tune only on the new trusted validation split.
6. Report final results only on the new trusted held-out test split.

Example split:

| New split | Fraction | Use |
| --- | ---: | --- |
| trusted_train | 60% | supervised fine-tuning |
| trusted_val | 20% | model selection, early stopping |
| trusted_test | 20% | final reported result |

If the trusted set is small, use repeated stratified splits or cross-validation for stability, but still keep one final untouched holdout if possible.

Do not train on the whole true-labeled test set and then report performance on it. That would leak the final evaluation set.

## Self-Supervised Image Options

Candidate methods:

| Method | Fit for this project | Notes |
| --- | --- | --- |
| SimCLR | Strong baseline, simple contrastive setup | Needs augmentations and decent batch size |
| BYOL | Avoids negative pairs | More moving parts than SimCLR |
| DINO | Strong representation learning | Heavier, but thesis-friendly if using existing implementation |
| MAE | Good for image reconstruction pretraining | Often best with ViT-style encoders |

Most practical first choice:

```text
Use a pretrained ImageNet MobileNetV3/ResNet/ViT backbone, then self-supervise or fine-tune on crop images.
```

Reason: starting from ImageNet weights is much cheaper and likely stronger than training SSL from random initialization.

## IoT/Tabular Options

The IoT branch can also avoid pseudo-labels.

Possible self-supervised tabular/sensor tasks:

| Task | Description |
| --- | --- |
| Masked feature modeling | Randomly hide sensor features and reconstruct them |
| Denoising autoencoder | Add noise/dropout to IoT features and reconstruct clean inputs |
| Contrastive sensor augmentations | Create two corrupted views of the same sample and align them |

Practical first choice:

```text
Denoising autoencoder or masked feature reconstruction for IoT features.
```

Then use the encoder output as compact features for supervised fusion on trusted labels.

## Multimodal Fusion Under This Path

After image and IoT encoders are pretrained without labels, build a small supervised fusion head using only trusted labels.

Possible model ladder:

| Model | Training signal | Purpose |
| --- | --- | --- |
| SSL-E0 | image SSL + true-label fine-tune | image-only trusted-label baseline |
| SSL-E1 | IoT SSL + true-label fine-tune | sensor-only trusted-label baseline |
| SSL-E2 | image + IoT SSL encoders, classical fusion | strong classical trusted-label baseline |
| SSL-E3 | same encoders, quantum/classical head | quantum comparison under cleaner label story |

This lets the thesis keep the E0/E1/E2/E3 structure, but changes the data claim.

## Where Quantum Fits In This Alternate Path

Quantum should not be forced into raw-image learning. The more realistic quantum tests are after representation learning:

1. Use SSL to produce compact embeddings.
2. Reduce embeddings to small dimensions if needed, such as 4, 8, 12, or 16.
3. Compare quantum heads or quantum kernels against matched classical heads/kernels.

Possible quantum lanes:

| Lane | Why it may fit |
| --- | --- |
| E3 VQC head on SSL embeddings | Keeps continuity with current E3 work |
| QSVM on compact SSL embeddings | Tests quantum kernel advantage in low-data setting |
| Quantum kernel on IoT SSL features | Lower-dimensional and possibly better suited to kernels |

Most defensible quantum question here:

```text
After self-supervised pretraining, can a quantum kernel or compact VQC perform better than matched classical methods in low-label regimes?
```

That is a cleaner and more plausible question than full QML on raw images.

## Experimental Plan

### Phase 0: Label Audit

Goal: lock down what is true-label and what is pseudo-label.

Deliverables:

```text
label_source_audit.md
trusted_label_manifest.csv
```

Questions to answer:

| Question | Needed answer |
| --- | --- |
| Are train labels generated? | Yes, confirmed premise |
| Are validation labels true? | No — val was split from the pseudo-labeled train set (confirmed 2026-07-03) |
| Are test labels true? | User states yes |
| Do image folders exactly match CSV rows? | Verified: val and test CSV rows match folder counts exactly (5,193 / 5,767). Train has 46,713 CSV rows vs 51,906 folders, and 46,713 + 5,193 = 51,906 exactly — the val samples were split out of the original train CSV but their folders remain in `Data/train/`. Dedupe against `val.csv` before using `Data/train/` as an unlabeled SSL pool |
| Are class/stage labels balanced enough for splitting? | No — heavy imbalance, see audit snapshot below |

Audit snapshot (2026-07-03), trusted test set only:

```text
5,767 samples, 25 classes
4 healthy/normal classes hold ~65% of samples (3_00_0: 1177, 4_00_0: 917, 6_00_0: 828, 1_00_0: 810)
6 rare classes under 30 samples: 6_a11_2 (12), 6_a12_1 (13), 6_b4_3 (14), 6_b4_1 (18), 6_b5_1 (21), 6_a12_2 (29)
```

Consequence: the trusted pool is the test set alone. A 60/20/20 stratified split leaves ~2-3 final-test samples for the rarest classes, so a tail-class policy (merge, drop with scope statement, or CV-only reporting) must be locked before building splits.

### Phase 1: Build Trusted Splits

Input:

```text
current true-labeled test set
```

Output:

```text
trusted_splits/train.csv
trusted_splits/val.csv
trusted_splits/test.csv
```

Rules:

- stratify by `crop_disease_stage`
- keep stage distribution visible
- keep seed fixed
- never touch final trusted test during tuning

### Phase 2: Image SSL

Train or adapt an image encoder using all images without labels.

Candidate data:

```text
Data/train/
Data/val/
Data/test/
```

Labels are ignored during SSL.

Output:

```text
ssl_image_encoder/checkpoints/best_encoder.pt
ssl_image_encoder/features/trusted_train_image_embeddings.npz
ssl_image_encoder/features/trusted_val_image_embeddings.npz
ssl_image_encoder/features/trusted_test_image_embeddings.npz
```

### Phase 3: IoT SSL Or Feature Encoding

Use masked reconstruction or denoising on IoT features.

Output:

```text
ssl_iot_encoder/checkpoints/best_encoder.pt
ssl_iot_encoder/features/trusted_train_iot_embeddings.npz
ssl_iot_encoder/features/trusted_val_iot_embeddings.npz
ssl_iot_encoder/features/trusted_test_iot_embeddings.npz
```

### Phase 4: True-Label Supervised Baselines

Train small supervised heads on trusted labels only.

Baselines:

| Baseline | Input |
| --- | --- |
| linear classifier | frozen SSL image embedding |
| small MLP | frozen SSL image embedding |
| IoT MLP | SSL or raw IoT features |
| fusion MLP | image + IoT embeddings |
| SVM/RBF | compact embeddings |

Primary metrics:

```text
Macro F1
Accuracy
Macro recall
Early-stage recall
Per-stage recall
ECE
```

### Phase 5: Quantum Comparison

Only after strong classical baselines are established.

Quantum candidates:

| Candidate | Comparator |
| --- | --- |
| VQC head on 8D SSL fusion embedding | matched classical 8D bottleneck head |
| QSVM on 4/8/12/16D SSL embedding | SVM RBF/poly/linear |
| QSVM low-data curves | classical SVM low-data curves |

Low-data settings:

```text
n = 25, 50, 100, 200, 500 if class counts allow
```

Repeated seeds are required before interpreting any advantage.

## Thesis Story If This Path Works

Possible revised story:

```text
Because the original training labels were pseudo-generated, the thesis reframes the task as label-efficient crop disease-stage recognition. Self-supervised learning is used to learn crop representations from all available images without relying on generated labels. A trusted true-labeled subset is then split into train/validation/test for honest supervised evaluation. Classical and quantum-classical heads are compared under this cleaner label regime, with emphasis on low-label performance, calibration, robustness, and matched-parameter fairness.
```

This story is cleaner than hiding pseudo-label training, but it may require more work and may reduce absolute performance because the trusted labeled set is smaller.

## Risks

| Risk | Why it matters | Mitigation |
| --- | --- | --- |
| Trusted label set is too small | Fine-tuning may be unstable | repeated stratified splits, cross-validation, confidence intervals |
| Class imbalance | Some disease-stage classes may have too few samples | merge rare classes only if thesis scope allows, or report limited-class setting |
| SSL training cost | Full SSL can be expensive | start from pretrained ImageNet encoder, use lightweight SSL/fine-tuning |
| Quantum still may not win | Cleaner labels do not guarantee quantum advantage | treat quantum as controlled comparison or low-data/kernel investigation |
| Claim drift | Project can become too broad | lock the claim before running full experiments |

## Decision Point

Use this alternate path if any of the following become true:

- the pseudo-label issue makes the current supervised thesis story unacceptable
- the lab PC gives enough compute to run SSL experiments cleanly
- the supervisor wants a more defensible label-quality framing
- the quantum advantage search shifts toward low-label or kernel-based learning

Do not switch automatically. The current E0/E1/E2/E3 work is still valuable as a controlled pseudo-label/fusion/quantum comparison. This note defines a cleaner alternate path if we choose to pivot.

## Immediate Next Steps If Activated

1. Verify label provenance for `Data/train.csv`, `Data/val.csv`, and `Data/test.csv`.
2. Count class/stage distribution in the true-labeled test set.
3. Create stratified trusted train/val/test splits from the true-labeled set.
4. Choose image SSL method, starting with the lightest viable option.
5. Train image-only trusted-label baselines before any quantum experiment.
6. Add QSVM/VQC only after the classical trusted-label baselines are strong.
