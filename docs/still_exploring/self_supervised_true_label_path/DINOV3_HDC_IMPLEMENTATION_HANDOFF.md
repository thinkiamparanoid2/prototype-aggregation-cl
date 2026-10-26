# DINOv3 → Continual-HDC Path — Pre-Registration & Codex Implementation Handoff

> **Execution status — 2026-09-07:** the dated specification below is retained
> as protocol history. The approved A+B splits, S/B/L ladder, probe gates,
> stream spec, and 108-run six-learner comparison are now present. The earlier
> tail-policy blocker was resolved July 19; NCM/RanPAC were added under the
> August 2 protocol and their results committed September 7. Read the
> [current handoff](../../../ssl_hdc_pilot/HANDOFF.md) and
> [PROVISIONAL report](../../../ssl_hdc_pilot/results/SSL_HDC_REPORT.md).
> Statements below about unbuilt phases are not current launch instructions.
> No preregistered numerical parameter or expectation is changed by this note.

Date: 2026-07-16. Status: **implementation handoff — start the unblocked
phases now.** This document is the binding spec for the implementing agent
(Codex). It supersedes the method-selection discussion in
`SELF_SUPERVISED_TRUE_LABEL_ALTERNATIVE.md` (which remains valid for the data
audit and phase skeleton). Where they disagree, this file wins.

Related: `SUPERVISOR_DECISION_TAIL_CLASS_POLICY.md` (the one open blocker,
scoped below), `NON_SSL_TRUE_LABEL_PATHS.md` (counter-paths, not this plan).

---

## 0. What this path is, in one paragraph

The original training labels are pseudo-labels; only the test set is true.
This path abandons pseudo-label supervision entirely. It uses a strong
pretrained vision foundation model (**DINOv3**) as the image representation,
never trains on untrusted labels, and puts the thesis contribution in a
**continual / incremental learning head (Hyperdimensional Computing, HDC)**
evaluated on a realistic streaming-deployment protocol with human-in-the-loop
labeling. Clean label story below, novel contribution on top.

## 1. The claim / where the novelty lives (read before coding)

- The **encoder is not the contribution** — it is plumbing. DINOv3's
  self-supervised training was done by Meta on 1.7B images; we do not claim
  it. Using DINOv3 frozen means our pipeline does no representation-learning
  of its own; that is acceptable because the novelty is downstream.
- **The contribution is the continual-learning head + the protocol + the
  question:** *do foundation-model (and domain-adapted) representations
  enable label-efficient, forgetting-resistant continual crop-disease
  learning, and does an HDC prototype learner beat CNN continual-learning
  baselines on update cost and human-label budget?*
- The one **encoder-side research question** we do allow, run as a bounded
  ablation, not a solo bet: *does a billion-scale foundation model need
  agricultural domain adaptation, or are its features already crop-ready?*
  Answered by comparing frozen vs domain-adapted DINOv3 (§5). Either
  direction is a finding — "adaptation didn't help" means "foundation
  features already suffice," which is informative, not a null.

## 2. Verified data facts (confirmed on disk 2026-07-16)

| Pool | Count | Labels | Role |
|---|---|---|---|
| `Data/train/` | 51,906 folders | pseudo (discarded) | encoder training pool |
| `Data/val/` | 5,193 folders | pseudo (discarded) | duplicate carve-out of train; unlabeled use only |
| `Data/test/` | 5,767 folders | **TRUE** | the trusted supervised pool |

`Data/train.csv` = 46,713 rows; 46,713 + 5,193 = 51,906. The val images are
physically inside `Data/train/`; `Data/val/` is a duplicate subset. Unique
image count ≈ 51,906 (train, incl. val) + 5,767 (test) ≈ **57,673**.

## 3. Data-pool rules (leakage guards — non-negotiable)

1. **Encoder training pool = the 51,906 `Data/train/` images only, labels
   discarded.** Deduplicate against `Data/val/` so no image is counted twice.
2. **The 5,767 trusted `Data/test/` images NEVER enter encoder training** —
   not even as unlabeled pixels. This kills the "the encoder already saw the
   test set" objection. Frozen DINOv3 obviously satisfies this; the
   domain-adaptation arm (§5.2) must explicitly exclude the trusted pool.
3. Trusted labels are used ONLY in the supervised/continual phases (§6.4+),
   never to train or select the encoder.

## 4. Tail-class policy — the ONE open blocker, and exactly what it gates

Trusted set: 5,767 samples, 25 classes, heavy imbalance; six crop-6 classes
have <30 samples. Chosen policy, per
`SUPERVISOR_DECISION_TAIL_CLASS_POLICY.md`: **Option A + B combined.**
**2026-07-19: SUPERVISOR APPROVED A+B** (communicated to the user; recorded
here the same day). The split gate is open; the official Option-A and
Option-B split commands in `ssl_hdc_pilot/RUN_COMMANDS.md` are cleared for
user launch with `--supervisor-approved`.

- **A (headline):** drop the six <30-sample classes → a **19-class** task
  with an explicit scope statement; one fixed 60/20/20 stratified split.
- **B (appendix):** all 25 classes under repeated stratified k-fold CV
  (5-fold × 10 repeats), reporting per-class recall with confidence
  intervals so the tail is disclosed with honest uncertainty, not hidden.
- Mandatory disclosures: (i) 19-class and 25-class macro-F1 are NOT
  comparable (different baselines); (ii) both reported regardless of outcome
  (pre-registered); (iii) tail classes get no untouched holdout — the wide
  CIs are the disclosure.

**What this blocks and what it does NOT:** it blocks Phase 1 (building the
official trusted splits) and everything supervised downstream (Phase 4/5).
It does **NOT** block encoder work (Phase 2/3) — those ignore labels. Build
the split script now, parameterized by policy; do not run the official split
until the policy is confirmed.

## 5. Encoder arms (three, feeding one identical downstream head)

All arms output frozen per-image embeddings; the encoder is frozen before any
downstream head sees it (no end-to-end fine-tuning of the encoder — it muddies
the comparison and the clean-label story).

### 5.1 Arm F — Frozen DINOv3 (the safe baseline, always works)
- **Resolved 2026-07-16:** repo id `facebook/dinov3-vits16-pretrain-lvd1689m`
  (DINOv3 ViT-S/16, 21M params, 384-d embeddings, patch 16, pretrained on
  LVD-1689M web images). License accepted by the user on HuggingFace; the
  user's local HF token authorizes downloads — do not re-request access.
  LVD-1689M (natural images) is deliberate: the crop images are ground-level
  leaf close-ups, NOT satellite imagery, so the SAT-493M variants are the
  wrong domain (and have no small model anyway).
- Zero training. Extract embeddings for all trusted images (and, for the
  domain-adapt comparison, the same images through Arm D).

### 5.1a Model-size decision (locked 2026-07-16)
- **Both arms use ViT-S/16 on the local RTX 5070.** Same-size rule: the
  frozen-vs-adapted comparison is only clean if the ONLY difference is the
  adaptation — never mix encoder sizes across arms.
- ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`) was considered and
  deferred: on 12 GB, DINO multi-crop adaptation of ViT-B lands at small
  batch sizes where teacher centering destabilizes — raising Arm-D collapse
  risk for a hardware reason. ViT-7B was rejected outright (adaptation needs
  ~150+ GB; no adapted arm ⇒ no experiment).
- **Conditional upgrade path (pre-registered, optional):** IF the lab PC's
  GPU is confirmed by nvidia-smi (never yet verified) AND schedule allows,
  the full two-arm study MAY be rerun at ViT-B/16 on that machine as a
  separately tagged pass. Never mix sizes within a pass; the ViT-S pass
  remains the headline unless the ViT-B pass fully replaces it before any
  results are interpreted.

  **2026-07-19 amendment — encoder-size LADDER replaces the single-size
  design (recorded BEFORE any Arm-D run; supersedes the block above):**
  measured feasibility sweeps on the local RTX 5070 (RESULTS_LOG entries of
  2026-07-19: ViT-S 0.758 steps/s / 2,241 MiB device peak; ViT-B 0.531 /
  4,006 MiB; ViT-L 0.309 / 9,780 MiB — all at batch 32, workers 0, no OOM)
  showed all three sizes are feasible locally, removing the 4090 condition.
  The design becomes a three-rung encoder-size ladder, mirroring the
  thesis's qsampler bit-width ladder:
  - Arm D trains at ALL THREE sizes — ViT-S/16, ViT-B/16, ViT-L/16 —
    sequentially (projected ~4 h + ~6 h + ~10 h ≈ 20 h, inside a 24 h
    window), identical recipe (LR 1e-5, EMA 0.996, 2+4 crops, AMP, gradient
    checkpointing, batch 32, workers 0, 5 epochs, per-epoch checkpoints,
    EMA-teacher export), separate output directories per size.
  - The same-size rule holds PER RUNG: frozen-X vs adapted-X only.
  - **HEADLINE = ViT-B/16, locked now, before any Arm-D result exists.**
    The downstream continual-HDC study (§6 Phase 4) runs on the ViT-B pair.
    ViT-S and ViT-L rungs are reported as the encoder-scale study (frozen
    and adapted probe results per rung) regardless of outcome; no post-hoc
    reselection of the headline under any result. If the ViT-B rung's
    adapted arm fails its probe gate, the headline downstream study uses
    frozen ViT-B (Arm F remains the guaranteed fallback) — not a different
    rung.
  - The 1,024 MiB headroom term is retained as a general smoke-to-run
    variability buffer; its original video-streaming justification is
    retired (the user has ended GPU video use during runs). ViT-L's
    9,780 MiB whole-device peak is within limits on an otherwise-idle
    card; the operational note stands: no other GPU applications during
    the ViT-L rung.
  - Probe gate: applied per rung with the identical fixed parameters
    (§5.2 amendment 2026-07-18); Arm-F trusted embeddings are extracted
    per size (384-d / 768-d / 1024-d) before gating.

### 5.2 Arm D — Domain-adapted DINOv3 (the novelty bet, bounded)
- Continue DINO self-distillation from the Arm-F weights on the **51,906
  encoder-pool images only** (§3). ViT-S/16 to stay tractable on one RTX 5070.
- HONEST RISK the agent must respect: naive continued DINO training commonly
  makes features *worse* (representation collapse / forgetting). Mitigations
  the agent must implement: short adaptation schedule, low LR, EMA teacher,
  and a **frozen-vs-adapted linear-probe checkpoint gate** — if adapted
  features do not match-or-beat frozen on a small held-out probe, stop and
  report; do not tune to chase it. Arm D is a sub-study; Arm F is the
  fallback that guarantees the thesis still stands.

  **2026-07-18 amendment — probe-gate policy resolved (recorded before any
  Arm-D run; answers the four questions in ssl_hdc_pilot/HANDOFF.md):**
  training and gating are DECOUPLED, because adaptation itself is label-free
  and only the gate needs labels.
  1. *Training (unblocked now):* Arm-D trains on the sanctioned unlabeled
     manifest only, saving a checkpoint per epoch (plus the frozen baseline
     as checkpoint 0). No labels, no trusted pixels, no mid-run gating.
  2. *Gating (after the official trusted splits exist):* for each saved
     checkpoint, extract embeddings for trusted_train and trusted_val; fit a
     multinomial logistic-regression probe on trusted_train (19-class
     Option-A labels, class-weighted, seed 20260716); score macro-F1 on
     trusted_val. The gate PASSES for the best adapted checkpoint iff its
     trusted_val macro-F1 >= the frozen baseline's minus 0.005 (noise
     allowance); checkpoint selection = highest trusted_val macro-F1.
  3. *Leakage accounting (stated, not hidden):* this is encoder-checkpoint
     selection using trusted TRAIN/VAL labels — standard model selection.
     trusted_test is never touched by the gate; the encoder never trains on
     trusted pixels or labels; pseudo-labels are used nowhere. This is
     within the §3 rules: rule 3 permits trusted labels in
     supervised/selection roles; rules 1–2 (encoder-training exclusions)
     remain fully intact.
  4. *Fixed parameters:* probe = sklearn LogisticRegression(max_iter=2000,
     class_weight="balanced", random_state=20260716) on raw 384-d
     embeddings; cadence = every epoch checkpoint; metric = macro-F1;
     tolerance = 0.005. None of these are tunable at run time.

### 5.3 Arm S — from-scratch SimCLR (OPTIONAL, only if time allows)
- ImageNet-init ResNet-50, SimCLR contrastive pretrain on the 51,906 pool.
- Purpose: shows why foundation models matter (expected to trail both DINOv3
  arms). Skip if the schedule is tight; not load-bearing.

## 6. Phase plan

- **Phase 0 — audit:** done (§2, and the prior doc). No action.
- **Phase 1 — trusted splits:** build `trusted_splits/{train,val,test}.csv`
  from the 5,767 trusted set under the §4 policy, stratified by
  `crop_disease_stage`, fixed seed, tail policy parameterized.
  ⛔ official split run gated on supervisor; script buildable now.
- **Phase 2 — encoder arms:** Arm F extraction (now), Arm D adaptation
  (now), Arm S (optional). Heavy runs are user-launched (§7).
- **Phase 3 — freeze & extract embeddings:** run trusted images through each
  frozen encoder → `embeddings/{arm}/{split}.npz`. Reuses the
  `full_qcnn_pilot/scripts/build_codes.py` extraction pattern (frozen model,
  GPU-batched, quality probes). Arm-F extraction needs no split (per-image),
  so it can run before Phase 1; splitting is applied downstream.
- **Phase 4 — downstream heads on trusted labels:**
  - Static baselines: linear probe, small MLP, SVM on each arm's embeddings.
  - **The contribution — continual/streaming protocol** (Rafsan Idea 2):
    item memory + HDC class prototypes; confidence-margin trigger;
    new-class registration; incremental update `prototype += encode(x)`.
    Stream simulator: 3 known classes + held-out classes introduced mid-stream;
    low-confidence → simulated human label event; log accuracy-over-time,
    forgetting curve, new-class uptake, and human-label budget.
  - **CNN continual baselines (the fair fight, non-negotiable):** naive
    fine-tune AND at least one real method (EWC or replay buffer). Do not
    ship an HDC-vs-strawman comparison.
  - Study grid: {Arm F, Arm D, (Arm S)} × {HDC-continual, CNN-naive,
    CNN-EWC/replay}. The embedding arm is a studied variable, not a hidden
    confounder.
- **Phase 5 — evaluation & writeup:** forgetting curves, new-class uptake
  speed, per-update compute cost, human-effort cost; macro-F1 / macro-recall
  / per-stage recall / ECE. Quantum head is OUT OF SCOPE for this branch
  (scope control — do not add it here).

## 6a. Streaming-protocol pre-registration (2026-07-19 amendment — locked
BEFORE the official splits or any continual run exist)

The Phase-4 stream spec (`ssl_hdc_pilot/stream_spec.json`) is GENERATED
deterministically from the approved Option-A split by a script applying the
following locked rules — no hand-chosen class lists:

1. **Novel classes (the "emerging diseases"):** the FOUR disease classes
   with the smallest trusted-set counts among the 19 retained classes
   (ties broken alphabetically). All other 15 classes (including all four
   healthy classes) are initially known. Rationale: rarest ≈ plausibly
   later-emerging; rule is deterministic from data.
2. **Initial prototypes / initial training:** trusted_train samples of the
   15 known classes only.
3. **The stream:** all remaining trusted_train samples — novel-class
   samples plus a held-back 25% of each known class (deterministic per
   seed) so known-class drift remains measurable. Novel classes enter at
   staggered points: 20%/40%/60%/80% of stream length, ordered by
   descending class count. Within constraints, order is shuffled by the
   stream seed.
4. **Seeds:** three stream orderings (seeds 0/1/2) over the SAME split.
5. **Confidence trigger:** top1−top2 similarity margin; thresholds swept
   over the fixed set {0.05, 0.10, 0.20} — all three reported, none
   selected post-hoc as "the" result.
6. **Label budget accounting:** every confirmed-label event (low-confidence
   trigger AND novel-class registration) counts against the budget;
   identical accounting for HDC and all CNN baselines.
7. **Evaluation:** forgetting measured on trusted_test known-class
   accuracy/macro-recall over stream time; uptake on trusted_test
   novel-class recall after each novel introduction; final metrics per §9.
   trusted_test is never streamed and never trains anything.
8. Headline learner comparison on the ViT-B frozen embeddings (per the
   ladder amendment); the adapted-vs-frozen embedding comparison joins
   only after the per-rung probe gates have run.

## 7. RUN OWNERSHIP (non-negotiable, standing protocol)

The implementing agent **NEVER launches official runs.** Smokes (1-epoch /
tiny-subset) and unit tests only. Every official command — DINOv3 domain
adaptation, full 57k/5.7k embedding extraction, the official trusted split,
the streaming-protocol runs, CNN-baseline training — is **appended to a
`RUN_COMMANDS.md` in the SSL pilot folder in labeled, user-launch blocks.**
Log an explicit "no official run was launched by the agent" line in every
handoff entry.

## 8. Environment

- Use **`.venv-e1`** (torch 2.11.0+cu128, torchvision, CUDA on the RTX 5070 —
  confirmed working; same env as `build_codes.py`).
- Add: `transformers` (or `timm`) for DINOv3 loading, and an HDC library
  (`torch-hd`/`torchhd`). Record exact versions installed. Do NOT touch
  `.venv-gpu` (that is the quantum CuPy env).
- Create the pilot under a new folder, e.g. `ssl_hdc_pilot/`, mirroring the
  existing pilots (`scripts/`, `results/`, `HANDOFF.md`, `RESULTS_LOG.md`,
  `CHANGELOG.md`, `RUN_COMMANDS.md`). Only MD/PY/CSV/JSON/NPZ inside it.

## 9. Metrics & pre-registered decision rules

- Primary (contribution): forgetting curve (accuracy on original classes over
  the stream), new-class uptake (examples to stabilize a new class),
  per-update compute cost, human-label budget. Report macro-F1 / macro-recall
  / per-stage recall / ECE — **never accuracy alone** (65% healthy imbalance).
- Encoder sub-study: frozen vs domain-adapted DINOv3 linear-probe macro-F1 on
  trusted_val; "adaptation helps" = adapted ≥ frozen beyond noise, else the
  finding is "foundation features already crop-ready."
- All comparisons on the SAME trusted splits; encoder never sees trusted data
  in training; test split touched once for the headline (CV for the tail).

## 10. What to implement FIRST (unblocked today)

1. Scaffold `ssl_hdc_pilot/` and install deps (§8); record versions.
2. Unlabeled-pool builder: enumerate the 51,906 `Data/train/` images, dedup
   vs `Data/val/`, exclude any trusted-test ids; write a manifest CSV.
3. Arm-F frozen DINOv3 loader + a per-image embedding extractor (smoke on
   ~50 images; confirm embedding dim from the model card). Append the
   full-extraction command to RUN_COMMANDS.md (user-launched).
4. Phase-1 split script, tail-policy parameterized, dry-runnable but NOT
   executed for the official split (await supervisor).
5. Arm-D domain-adaptation training script + the frozen-vs-adapted probe gate
   (§5.2); smoke on a tiny subset; append the official training command.
Defer Phase 4/5 head code until embeddings exist and the split policy lands.

## 11. Hard don'ts

- No training on pseudo-labels, ever. No trusted-test images in encoder
  training. No end-to-end encoder fine-tuning for the headline.
- No launching official runs (agent = smokes + scripts + appended commands).
- No quantum head in this branch (scope).
- No editing pre-registered expectations after results exist; a contradicting
  result IS the finding.
- No accuracy-only reporting; no HDC-vs-naive-only comparison (EWC/replay
  required). Do not tune Arm D to beat frozen — gate and report.
- Repo ids are resolved in §5.1/§5.1a — use them verbatim; the license is
  already accepted by the user. Do not switch model size or pretraining
  corpus (LVD vs SAT) without a dated amendment here.


## 2026-09-10 — supervisor-priority clarification and proposed endpoint reconciliation

The user relayed that the supervisor wants clear indicators of reduced catastrophic
forgetting through HDC and higher throughput, and authorized the documentation
update. The thesis now tests those two outcomes explicitly against NCM and
RanPAC-style as primary prototype competitors, alongside naive, EWC, and replay.
This is a statement of research priorities, not evidence that either win exists.

The [revised core draft](../../TODO/HDC_CORE_EVALUATION_PROTOCOL_DRAFT_2026-09-09.md)
replaces the earlier same-day AULC-only primary preference with fixed-population
forgetting and cached-embedding continual-head throughput as primary thesis
outcomes. Macro-F1/AULC, initial/final old-class performance, novel-class uptake,
label expenditure, and memory remain required evidence of the resulting tradeoff.
It requires comparator-specific contrasts and joint wins against the same
comparator at the same arm/budget; a gradient-only win cannot establish superiority
over NCM/RanPAC-style. Throughput excludes the encoder and is not edge deployment.

Current evidence is provisional: rolling forgetting favors HDC over the tested
gradient learners under unequal supervision; its advantage over the prototype
controls depends on encoder arm. Recorded update durations favor NCM/RanPAC-style
over HDC and are not validated throughput measurements. See the current report.

All earlier text and registrations above remain historical provenance. This dated
scope note does not retroactively register new metrics or change stream parameters,
splits, hyperparameters, or result artifacts. Detailed measurement rules, timing
repetitions, practical criteria, the A+B appendix, and test release still require
the draft's A1–A4 freeze process. No implementation, official run, or test access
is authorized by this note. No official run was launched by the agent.
