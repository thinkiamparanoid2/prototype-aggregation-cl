# Feasibility Study — Ground-Up Domain Encoder and Corpus Expansion

Date: 2026-07-20. Feasibility analysis only; **not** a pre-registration and not
a commitment to run. Scheduled *after* the Study-2 encoder-geometry ablation
(`ENCODER_GEOMETRY_PREREG.md`). All hardware and wall-clock numbers are derived
from measurements already in `ssl_hdc_pilot/RESULTS_LOG.md`, with the
extrapolations marked as such.

## 0. The two questions

1. **Ground-up:** can we train our own encoder from scratch on unlabeled crop
   imagery, instead of adapting someone else's foundation model?
2. **Corpus expansion:** can we add external agricultural/plant image datasets
   so that encoder has enough data to learn from?

They are one question in practice: from-scratch SSL is data-bound, and our
current pool is far too small, so (1) is only interesting if (2) works.

## 1. Hardware and current assets (verified)

- Lab PC GPU confirmed 2026-07-20 via `nvidia-smi`:
  **NVIDIA GeForce RTX 4090, 24,564 MiB, driver 591.86**. This also satisfies
  the long-outstanding hardware precondition of the sampler track's gate G6b.
- Free space on `F:`: **260 GB** of 432 GB.
- Unlabeled pool: **51,906 images** (`Data/train`), disjoint from the 5,767
  true-labeled evaluation images in `Data/test`. Firewall verified in Study 1.
- The full DINO multi-crop self-distillation recipe already exists and is
  tested: `ssl_hdc_pilot/scripts/adapt_dinov3.py` (2 global 224px + 4 local
  96px crops, EMA teacher at 0.996, AMP, mandatory gradient checkpointing,
  atomic per-epoch resume).
- No HuggingFace cache exists on this machine; the S/B/L weights are cached on
  the other PC. Any encoder work here begins with downloads.

## 2. Ground-up on the current pool alone: not viable

| Corpus | Images | Outcome |
|---|---|---|
| DINOv3 (LVD-1689M) | ~1.7B | the model we are comparing against |
| DINOv2 | ~142M | strong general features |
| Typical ImageNet SSL | 1.28M | the usual floor for credible from-scratch SSL |
| **Our pool** | **51,906** | **~25× below that floor** |

From-scratch self-supervised training below roughly 1M images reliably
underperforms plain supervised ImageNet transfer. Training a ViT-S from random
initialization on 51,906 images would produce a weak encoder and a predictable
negative result. **Framed as a challenger to DINOv3 it is not worth the GPU
time.**

It has exactly one defensible framing, and it is a good one — see §5.

## 3. Corpus expansion: the numbers

Candidate public plant/agricultural image corpora, ordered by contribution:

| Source | Approx. images | Domain match | Notes |
|---|---|---|---|
| PlantCLEF (2024/2025) | ~1.4M | species ID, field + herbarium | by far the largest single contributor |
| iNaturalist, plant subset | 1M+ | field photos, wide species range | licence varies per observation |
| PlantVillage | ~54k | leaf disease, lab conditions | closest domain; **highest contamination risk** |
| DeepWeeds | ~17k | field weeds | different task, same imaging conditions |
| Cassava Leaf Disease | ~21k | leaf disease, in-field | good domain match |
| Plant Pathology (apple) | ~18k | leaf disease, in-field | good domain match |
| Rice / paddy disease sets | ~10k | leaf disease | good domain match |
| **Our pool** | **51,906** | exact | the anchor |

Realistically assemblable: **~1.5–2.6M images**, i.e. **30–50× the current
pool**, which moves from-scratch SSL from "not viable" to "plausible for a
ViT-S". Storage at ~100–200 GB fits the 260 GB free, but not comfortably
alongside everything else — budget it explicitly.

Note the composition problem: the volume is dominated by PlantCLEF and
iNaturalist, which are **species identification** — whole plants, herbarium
sheets, varied backgrounds — while the target task is **close-up diseased
leaves with severity grading**. The effective in-domain fraction is closer to
~170k images (the disease-specific sets plus ours). More data helps SSL, but
1.4M herbarium photos are not worth 1.4M crop-disease photos here.

## 4. The blocking risk: contamination of the evaluation set

This is the one that can invalidate the whole thing, and it must be settled
before any download.

Our labels are `{crop}_{disease}_{severity}` (e.g. `3_b7_1`, with `00_0` as
healthy) across six crops — the encoding of a **public agricultural-disease
benchmark**. Study 1's strongest methodological asset is that no evaluation
image ever entered the SSL pool. Adding external corpora puts that at risk in
two ways:

1. **Direct re-import.** If our dataset's upstream source is itself among the
   public corpora we download, our 5,767 evaluation images can re-enter the
   pretraining pool through the back door. This would silently destroy the
   firewall.
2. **Near-duplicate re-import.** Even without an exact source match, public
   leaf-disease sets share images and re-crops across releases.

**Mandatory mitigations if this proceeds:**

- Establish and record our dataset's exact upstream provenance **first**. This
  is currently undocumented anywhere in the repo and is a gap regardless of
  this study.
- Exclude any corpus identified as our upstream source, in whole.
- Run exact-hash **and** perceptual-hash (pHash) deduplication of every
  candidate image against all 5,767 trusted images; publish the removal count
  per source. A nonzero count is expected and is fine — an unmeasured count is
  not.
- Re-verify the firewall after assembly with the same check Study 1 used.

Secondary risks: per-source licences vary (fine for thesis research use, but
each needs a line in the writeup) and the assembled manifest becomes a
non-reproducible artifact unless the exact image ID list is committed.

## 5. Why the weak rung is scientifically valuable anyway

Study 1 §5 reports a linear probe on frozen DINOv3 ViT-B reaching **0.887**
macro-F1 on the static 19-class task, and the Study-2 prereg registers an
explicit saturation flag at 0.95. Every encoder discussed so far pushes
features *stronger*, which pushes the comparison further into the region where
all learners look similar and the effect being measured is compressed.

A from-scratch domain encoder is the only proposal on the table that produces a
**deliberately weak rung** — and weak features are where update-mechanism
differences are most visible. It anchors the bottom of the geometry axis better
than ImageNet ResNet-18 does, because it is weak *and* domain-specific rather
than weak *and* generic.

It also gives the 51,906-image pool a headline role: "we have ~52k unlabeled
field images (plus ~1.5M public plant images) and here is what they buy on
their own" is a materially stronger deployment narrative than using them only
as an adaptation warm-up.

## 6. Cost

Measured basis: `arm_d_vits16` ran 5 epochs over 51,906 images in **2.69 h** on
the RTX 5070 — about **0.54 h/epoch**. Extrapolations below assume the 4090 is
roughly 1.5–2.5× that throughput and that per-epoch cost scales with corpus
size; both are estimates, not measurements.

| Scenario | Corpus | Epochs | Estimated 4090 wall-clock |
|---|---|---|---|
| ViT-S, current pool | 52k | 100 | ~1–1.5 days |
| ViT-S, current pool | 52k | 300 | ~3–4.5 days |
| ViT-S, in-domain expansion | ~170k | 100 | ~3.5–5 days |
| ViT-S, full expansion | ~1.5M | 100 | ~4–6 weeks (**out of scope**) |

Plus one-off costs: dataset downloads (~100–200 GB), dedup pass over ~1.5M
images, manifest construction.

**Verdict on scope:** the full 1.5M-image corpus is out of reach for a thesis
timeline on one GPU. The **in-domain expansion (~170k)** is the sweet spot —
3× the current pool, dominated by actual leaf-disease imagery, and ~4 days of
training rather than a month.

## 7. Technical risk: collapse, not accuracy

`RESULTS_LOG.md` already carries a watch item that Arm-D final DINO losses of
0.02–0.05 may indicate healthy convergence **or** partial representation
collapse. Adapting from pretrained weights, that risk is bounded by the
initialization. **From random initialization it is materially higher**, and
DINO-family collapse is often silent — the loss looks excellent while the
features become near-degenerate.

Required if this runs:

- The linear probe gate is the collapse detector and the sole arbiter, with the
  `WEAK_ENCODER` threshold from the Study-2 prereg applying.
- Register in advance that a collapsed encoder is **reported as collapsed**,
  not quietly retuned into a working one.
- Log embedding-variance and rank diagnostics per checkpoint, not just loss.

## 8. Recommendation

1. **Run Study 2 first** (four frozen encoders, `ENCODER_GEOMETRY_PREREG.md`).
   It is hours of GPU, answers the geometry question directly, and its
   saturation flags tell us whether a weak rung is even needed.
2. **Settle dataset provenance now**, independently of everything else. It is a
   one-afternoon documentation task, it is a gap in the thesis record today,
   and every contamination judgement in §4 depends on it.
3. **If and only if** Study 2 shows saturation compressing the learner gaps,
   proceed to a ground-up ViT-S on the **in-domain expanded corpus (~170k)**,
   pre-registered separately as Study 3, framed as the weak/domain rung — never
   as a challenger to DINOv3.
4. **Do not** attempt the full 1.5M-image corpus, and do not attempt any
   from-scratch model larger than ViT-S, on this timeline.
