# Study 2 Pre-Registration — Encoder-Geometry Ablation for Continual HDC

> **Status reconciliation — 2026-09-07: unrun; protocol reconciliation pending.**
> The original July 20 registration below is retained verbatim as history.
> Its references to a final four-learner Study 1 and immunity to forgetting
> are superseded by the [updated six-learner report](../../../ssl_hdc_pilot/results/SSL_HDC_REPORT.md).
> Study 1 is PROVISIONAL, its final scores use the training pool after
> streaming, budgets differ, and NCM/RanPAC challenge HDC-specific attribution.
> The geometry explanation remains a hypothesis. Before execution, record a
> dated protocol amendment resolving learner coverage, annotation accounting,
> evaluation populations, and resulting comparison counts. This status note
> does not change the registered encoder set, thresholds, or expectations,
> and does not select a new experimental design.

Date: 2026-07-20 (written BEFORE any new encoder is downloaded, extracted, or
probed). Parent study: `DINOV3_HDC_IMPLEMENTATION_HANDOFF.md` and
`ssl_hdc_pilot/results/SSL_HDC_REPORT.md` (Study 1, final 2026-07-20).

**Study 1 is not reopened.** Its 72-cell grid, its verdicts, and its report
stand exactly as committed. This document registers a *separate* study with its
own question, its own encoder set, and its own locked rules — the same
Stage-1 → Stage-2 structure used on the quantum-sampler track. Nothing here
retro-fits arms into Study 1's headline comparison, and no Study-1 number is
recomputed, restated, or superseded by this study.

## 0. Question under test

Study 1 found that on frozen DINOv3 ViT-B features, an HDC prototype learner
reaches 0.772 final macro-F1 against 0.370 / 0.369 / 0.234 for naive
fine-tuning / EWC / replay, and is essentially immune to the catastrophic
forgetting that collapses all three. Study 1 §2 explains the surprising
frozen > adapted result as domain adaptation "distorting the prototype geometry
HDC relies on for incremental bundling."

That is a claim about **representation geometry**. Study 1 cannot test it,
because every rung it used (ViT-S/B/L, frozen and adapted) comes from a single
self-distillation family. Scaling within one family varies capacity, not
geometry.

**Q: Does HDC's continual-learning advantage over gradient baselines depend on
the geometry of the frozen representation, or does it hold across
representation-learning objectives and backbone architectures?**

No new claim of quantum or algorithmic novelty is made. This is an
external-validity check on a finding that already exists.

## 1. Encoder set (locked; exactly four)

| # | Encoder | Objective family | Architecture | Role |
|---|---|---|---|---|
| 1 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | self-distillation | ViT | anchor; already extracted in Study 1 |
| 2 | DINOv3 ConvNeXt-L | self-distillation | ConvNeXt | architecture control (family held constant) |
| 3 | `google/siglip2-so400m-patch14-384` | language-contrastive + multi-objective | ViT | different objective |
| 4 | `apple/aimv2-3B-patch14-224` | multimodal autoregressive | ViT | different objective |

Rung 1 is the bridge to Study 1: it is the same embedding file, unchanged, so
Study 2's numbers can be read against Study 1's without re-running anything.

**Deliberately excluded, with reasons recorded now:**

- **C-RADIOv4-H/SO400M** — its teachers are SigLIP2-g-384, **DINOv3-7B**, and
  SAM3. Because one teacher is the model family already on the ladder, it
  cannot serve as an independent geometry. It is a legitimate but *different*
  question ("does a blended geometry behave like its DINO teacher?") and is
  deferred to a possible Study 2b, where the confound must be stated in the
  text.
- **InternViT-6B** — VLM-coupled training makes the objective attribution
  muddy; adds scale, not a new geometry.
- **Perception Encoder intermediate-layer study** — requires a layer index,
  which is a free parameter over ~50 candidates. Deferred rather than run with
  an unlocked knob (see §3).
- **DINOv3 ViT-7B adaptation** — infeasible on 24 GB (~134 GB for the
  registered full-encoder DINO recipe; ~53.6 GB even with bf16 + 8-bit
  optimizer states) and pre-registered non-adaptable in
  `ssl_hdc_pilot/scripts/encoder_models.py`. Frozen 7B extraction remains
  available as an optional ceiling reference and, if run, is reported
  separately from the four-rung comparison.

## 2. Everything held fixed from Study 1

Changing the encoder is the only manipulation. All of the following are reused
without modification, and any deviation invalidates the study:

- The supervisor-approved Option-A trusted splits (3,396 / 1,132 / 1,132; 19
  classes). `trusted_test` streams nowhere and trains nothing.
- The four learners (HDC, naive fine-tune, EWC with decay 0.9, replay) at their
  Study-1 hyperparameters. No retuning per encoder under any outcome.
- Three stream seeds, three confidence thresholds, identical class-introduction
  schedule and stream ordering.
- HDC hypervector dimension and projection seed.
- The probe-gate procedure (linear probe on `trusted_val`, macro-F1).

## 3. Locked extraction choices (the anti-forking-path section)

These are fixed now, before any number is seen, precisely because each is a
knob that could otherwise be tuned toward a preferred result:

1. **Pooling and layer.** Final-layer CLS token, matching Study 1's
   `last_hidden_state[:, 0, :]`. **No layer search is permitted in this
   study.** For ConvNeXt, which has no CLS token, the registered equivalent is
   global average pooling of the final feature map; this substitution is
   recorded as a known architecture-forced asymmetry and stated in the report.
2. **Preprocessing.** Each encoder's own official image processor, at its
   native training resolution. Forcing a common resolution would handicap
   encoders trained at other scales.
3. **Precision.** bfloat16 inference, fp32 storage, as in the 7B rung.
4. **No L2 normalization at extraction.** See §4 — for HDC it is provably a
   no-op, and applying it to new rungs only would break comparability with
   Study 1's committed embeddings.
5. **Batch size** is a throughput-only parameter and may be chosen freely; it
   must not change any recorded value.

## 4. Embedding-scale invariance (resolved analytically, not by re-running)

`ssl_hdc_pilot/scripts/continual_hdc.py:84-85` encodes as
`sign(x @ P)` (`torchhd.functional.normalize` binarizes to bipolar ±1).
Per-sample L2 normalization divides `x` by the positive scalar `‖x‖`, and
`sign((x/‖x‖) @ P) = sign(x @ P)`. **The HDC arm is therefore exactly
invariant to embedding scale**, and re-running it under normalized inputs would
reproduce Study 1's results bit-for-bit.

Registered actions:

- Add a unit test asserting bit-identical hypervectors, prototypes, and
  predictions under per-sample L2 rescaling. This *replaces* an HDC re-run.
- The gradient baselines are **not** invariant (`continual_baselines.py:79`
  feeds embeddings into learned layers). Whether to normalize their inputs is a
  baseline-strength decision, registered here as: **run the gradient baselines
  both ways in Study 2 and report the stronger result as the comparator.** This
  can only make the HDC comparison harder, never easier, and is adopted for
  exactly that reason.
- Study 1's numbers are not restated under this rule.

## 5. Gates

**G-P (per-encoder probe gate, before any continual grid).** Linear probe on
`trusted_val`. Two outcomes are recorded, neither of which halts the study:

- Probe macro-F1 **≥ 0.95** ⇒ flag `SATURATION_RISK` for that encoder. Study 1
  §5 already reports 0.887 for frozen ViT-B, so this is a live possibility.
  Saturated rungs are still run, but the report must state that learner
  differences at that rung are compressed by ceiling effects and must not be
  read as evidence that the update mechanism stopped mattering.
- Probe macro-F1 **≤ 0.40** ⇒ flag `WEAK_ENCODER`; extraction is re-verified
  (NaN/zero-row/shape checks) before the rung is accepted.

**G-D (provenance gate).** For every rung: no NaN, no zero rows, row count
5,767, dimension matches the model's declared hidden size, and the trusted-ID
join is exact. Same checks Study 1 applied.

## 6. Primary and secondary outcomes

**Primary.** For each encoder, the gap between HDC and the **best** gradient
baseline on that encoder, in (a) final macro-F1 and (b) end-of-stream
known-class accuracy. The study's answer is the *pattern of that gap across the
four encoders*, not any single cell.

**Secondary.** Per-update compute; label requests (with Study 1 §4's
accuracy-vs-labels caveat carried forward); novel-class uptake; static probe
score per rung.

**Statistical handling.** Three stream seeds and three thresholds per cell, as
in Study 1; medians across seeds reported with per-seed values in the run
summaries. With four encoders the comparison count is stated explicitly in the
report and no per-cell significance claim is made from a single seed triple.

## 7. Pre-registered expectations (locked before results)

Written now so that either outcome is reportable and neither can be reframed
afterwards:

1. **Modal expectation:** HDC's advantage persists in *direction* on all four
   encoders, and *narrows* as static probe quality rises. Forgetting resistance
   is a structural property of prototype bundling and should not depend on the
   objective; the size of the win should depend on headroom.
2. **The architecture control (rung 2) is expected to matter least.** If
   ConvNeXt-L and ViT-L differ materially while the cross-objective rungs do
   not, the geometry explanation in Study 1 §2 is wrong and must be revised.
3. **A genuinely possible outcome:** the advantage disappears on the
   language-contrastive rung. That would be a *finding* — it would localize
   HDC's benefit to self-distillation geometry and sharpen, not weaken, the
   thesis claim. It is registered here as reportable, not as a failure.
4. **Contamination asymmetry.** SigLIP 2 (WebLI, ~10B image-text pairs) and
   AIMv2 train on large web corpora. The source dataset here uses
   `{crop}_{disease}_{severity}` labels over six crops and appears to derive
   from a public agricultural-disease benchmark; if so, its images may be
   present in those pretraining corpora, whereas the Study-1 leakage firewall
   (eval images never in *our* SSL pool) does not bound that. This does not
   invalidate the comparison — the learners are compared on fixed features —
   but it is stated in the report as a limit on cross-rung interpretation, and
   the dataset's exact provenance must be confirmed and recorded before the
   report is written.

## 8. Hard don'ts (inherited)

No retuning of learner hyperparameters per encoder; no layer search; no test
peeking (selection on `trusted_val` only); no editing this document after any
Study-2 number exists except by a dated amendment appended below; no agent
launches an official run — every official command is appended to
`ssl_hdc_pilot/RUN_COMMANDS.md` for the user; if a result contradicts §7, that
IS the finding and it is reported as such.

## 9. Deliverable

`ssl_hdc_pilot/results/ENCODER_GEOMETRY_REPORT.md`, with the four-rung primary
table, the probe-gate row per encoder including any `SATURATION_RISK` flag, the
§7 expectation-vs-outcome table, and the §7.4 contamination limit. Study 1's
report is linked, not modified.

## 2026-09-07 documentation reconciliation

No Study-2 result artifacts were found in the audited checkout. The original
four-learner clause above is not a description of current Study 1, which now
includes HDC/NCM/RanPAC/naive/EWC/replay. Its normalization, learner-selection,
and outcome rules need to be reconciled together before a new grid is run;
this documentation task has not silently amended those scientific choices.
