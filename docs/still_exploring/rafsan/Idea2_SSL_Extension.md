# Idea 2 Extension — SSL Backbone Ladder for HDC Continual Learning

*Design note, 2026-07-15. Extends
`Idea2_HDC_Continual_Learning_Crop_Disease.md`. Numbers verified against the
repo; DINOv3 facts web-sourced (see end).*

## 0. The question and the answer

**Q:** Can Idea 2 be a follow-up from a self-supervised (SSL) backbone,
using the large pseudo-labeled pool that currently plays no part?

**A: Yes — and it is the strongest version of the idea.** SSL needs images,
not labels, so the pseudo-pool stops being dead weight (excluded by the
clean-label rule) and becomes a pretraining corpus. The pseudo-labels
themselves are never touched — only the pixels.

## 1. The data-pool math (verified in-repo)

| Pool | Location | Count | Label status | Role here |
|---|---|---|---|---|
| Pseudo-pool | `Data/train/` + `Data/val/` | 51,906 + 5,193 = **57,099** | pseudo (untrusted) | **SSL pretraining corpus (pixels only)** |
| Evaluation | trusted 3-seed splits from `Data/test/` | **1,794** (of 5,767) | true | all supervised train/stream/eval |

**Leakage firewall (checked, not assumed):** all 1,794 split ids resolve to
`Data/test/`; overlap with `Data/train` = 0, with `Data/val` = 0. Pretraining
only on the pseudo-pool therefore *provably* never sees an evaluation image —
the standard "your SSL features already saw the test set" objection is dead
on arrival.

**Second bonus — clean attribution.** With a label-free backbone, *all*
supervised signal flows through the HDC prototype mechanism (the component
Idea 2 studies). This defuses the "the CNN embedding does the real work"
critique far better than an ImageNet-supervised backbone.

## 2. The design: a three-rung backbone ladder

The backbone is the experimental axis, not a knob to maximize. Idea 2's
contribution is the *update mechanism*, so what matters is how the
continual-learning advantage **interacts** with backbone strength. Three
rungs, weak → strong:

| Rung | Backbone | Pretraining | Pool used? | Why it's here |
|---|---|---|---|---|
| 1 | ResNet-18 | ImageNet, frozen | no | weak baseline; **already built** in `build_codes.py`; the rung where continual gaps are most visible |
| 2 | DINOv2 ViT-S | **domain-continued on the 57k pool**, frozen after | **yes** | the pool-justifying rung; answers "does domain-continuing a foundation model help?" |
| 3 | DINOv3 (ConvNeXt-S or ViT-S/16) | LVD-1689M, frozen off-the-shelf | no | strong ceiling reference; best available features |

Notes:
- **Rung 2 is the only rung that gives the 57k images a job.** DINOv2 ViT-S
  is the ceiling that fits *continued* SSL pretraining on a 12 GB RTX 5070
  (continue from released weights + strong augmentation; from-scratch SSL on
  57k would be underpowered).
- **Rung 3 is frozen-only by necessity.** DINOv3 was trained on ~1.7B images
  with a recipe (Gram anchoring, massive scale) that cannot be meaningfully
  continued at our scale — so it is used off-the-shelf. It is the "how good
  does it get" ceiling, not a pool consumer. License is custom-but-
  commercial-friendly (fine for a thesis; note it in the writeup).

For every rung: re-extract embeddings via the existing `build_codes.py`
extraction path with a swapped checkpoint, then run the **identical**
continual-stream protocol (same splits, stream orderings, confidence
thresholds) across three learners: HDC-continual, naive CNN fine-tune,
EWC/replay CNN.

**The finding this produces:** does HDC's continual-update advantage survive
as features strengthen (ResNet → DINOv2-domain → DINOv3)? Survives even
under DINOv3 → strongest possible result. Washes out under DINOv3 but holds
on weaker rungs → a genuine characterization of *when* the mechanism
matters. Either way, publishable.

## 3. Probe-first gate (do this before building anything)

Push each backbone's frozen embeddings through the logistic-regression probe
already in `build_codes.py`. If DINOv3 features already saturate the 5
classes near ceiling, the continual-learning differences will be invisible
no matter how good the stream harness is — a signal to **harden the
evaluation first** (more classes, or finer-grained disease grades) before
spending weeks on the protocol. One afternoon; potentially saves the study.

## 4. Hard rules to pre-register

- Pseudo-labels are NEVER consumed as labels anywhere — SSL/frozen features
  use pixels only; no semi-supervised arm. (One sentence closes the
  contamination question.)
- All backbones frozen for downstream work; no supervised fine-tuning of the
  backbone. Rung 2's SSL continuation is label-free.
- Evaluation images never seen by any SSL objective — enforced by the
  verified pool disjointness; assert it in code at extraction time.
- Backbone strength is a reported axis; no cherry-picking the rung that
  flatters HDC.

## 5. Timeline / team impact

- **+2–3 weeks** on top of the 6–8-week Idea 2 core (from the earlier
  assessment). The Rung-2 SSL continuation parallelizes with Idea 2's
  Phases 2–4 because it only gates its own rung; Rungs 1 and 3 need no
  pretraining at all.
- Team split unchanged from Idea 2; the SSL continuation is a bounded task
  for whoever owns the encoder (reuses `build_codes.py`'s extraction plumbing
  and the `.venv-gpu` environment).

## 6. Risks (pre-register as reportable, not as failures)

| Risk | Handling |
|---|---|
| Strong features (DINOv3) wash out continual-learning gaps | Keep ResNet-18 rung for visibility; probe-first gate detects saturation early |
| SSL-on-57k (Rung 2) underperforms ImageNet/DINOv3 features | Reportable finding; the probe gate quantifies it before the full protocol |
| DINOv3 frozen-only ⇒ pool underused on that rung | By design; Rung 2 is where the pool earns its place — state this explicitly |
| GPU contention with the thesis run queue | SSL continuation is bounded (ViT-S, days) and schedulable around official runs |
| Augmentation/hyperparameter choices look arbitrary | Document the SSL recipe fully; it is label-free so it cannot leak evaluation signal |

## 7. Sources (DINOv3 facts)

- Encord — "DINOv3 Explained: Scaling Self-Supervised Vision Transformers"
- Lightly — "DINOv3 Technical Deep Dive"
- HuggingFace — `facebook/dinov3-convnext-small-pretrain-lvd1689m`

(Model family: ViT-S 21M / ViT-S+ / ViT-B 86M / ViT-L 300M / ViT-H+ / ViT-7B,
plus ConvNeXt T/S/B/L; frozen-feature SOTA; custom commercial-friendly
license; on HF Hub.)
