# QLoRA-DINO Side Pilot — Bounded Spec (NOT thesis work)

Date: 2026-07-19. Status: **parked side project — implementation may begin
only when no thesis-critical work wants the GPU or the implementing agent.**
Promoted from the idea parked in `FUTURE_EXTENSIONS.md` §1 into a bounded,
kill-gated pilot spec. This document is the whole scope; anything beyond it
is a new decision.

## Quarantine rules (before anything else)

1. Lives in its own pilot folder (`qlora_dino_pilot/` when created); never
   imports from, writes into, or modifies `ssl_hdc_pilot/` or any
   pre-registered pipeline. May READ the sanctioned unlabeled manifest CSV
   (it is the same leakage-safe pool).
2. Appears in the thesis ONLY if later promoted by a dated note in the
   binding SSL handoff. Default fate: standalone curiosity / possible
   workshop-paper seed.
3. Same run-ownership protocol as everything else: agent implements and
   smokes; official (long) runs are user-launched from a RUN_COMMANDS.md.
4. Thesis-critical work always preempts it — GPU, agent cycles, and human
   attention. No exceptions, including interesting partial results.

## The question (one sentence)

Can LoRA adapters trained with a DINO-style self-distillation objective
domain-adapt a 4-bit-quantized frozen DINOv3 ViT-7B on ~52k crop images,
on a single 12 GB RTX 5070, without representation collapse?

Novelty claim if it works: "QLoRA-DINO" — self-distillation-based
parameter-efficient domain adaptation of a quantized vision foundation
model. We found no prior instance (checked July 2026, informally; a real
literature pass is REQUIRED before any external claim).

## Design (minimum viable, nothing more)

- Base: `facebook/dinov3-vit7b16-pretrain-lvd1689m` (must be downloaded —
  ~13 GB; NOT yet in the local cache, only an empty stub), loaded 4-bit
  (bitsandbytes NF4, fp16 compute dtype).
- Student = quantized base + fp16 LoRA adapters (attention q/k/v/o +
  MLP projections; r=8, alpha=16 to start — record, do not tune-sweep).
- Teacher = the SAME shared quantized base + an EMA copy of the LoRA
  deltas only (momentum 0.996, as the main track). This EMA-over-LoRA
  teacher is the novel engineering piece — the quantized base is shared
  memory between student and teacher; only adapter weights are duplicated.
- Objective: DINO loss between teacher(global crops) and student(global +
  local crops), 2 global + 4 local, AMP, gradient checkpointing,
  batch 2–4 (whatever fits), LR 1e-4 on LoRA params only (LoRA tolerates
  higher LR than full fine-tuning; still record-not-tune).
- Data: the sanctioned 51,906-image manifest, labels never read.
- Success readout (later, after the thesis trusted splits exist): linear
  probe frozen-7B-int4 vs QLoRA-adapted-7B-int4 on trusted_train/val,
  same fixed probe as the thesis gate. Until splits exist, the ONLY
  readouts are loss curves and feature-variance diagnostics (collapse
  detection) — no labels, no improvised probes.

## Kill gates (written before code; each is terminal, one debug pass max)

- K1 — fit: student+teacher+optimizer does not run at batch >= 2 with
  checkpointing on 12 GB → dead.
- K2 — speed: > 5 s/step at the fitting batch → dead (nothing trainable
  in side-project time).
- K3 — stability: NaN/Inf loss or feature-variance collapse within a
  1,000-step smoke, surviving ONE debugging attempt → dead.
- K4 — schedule: any moment the pilot blocks thesis work → paused
  immediately, no negotiation.
- A killed pilot gets a dated post-mortem paragraph in this file and rests
  in peace. Kill-gate outcomes are results, not failures to hide.

## Windows/tooling notes for the implementing agent

- bitsandbytes: official Windows wheels exist since 0.43.x — verify import
  + CUDA kernel availability in `.venv-e1` BEFORE building anything else;
  if unavailable on this stack, that is K1-adjacent and reported, not
  worked around with WSL heroics.
- PEFT/transformers LoRA targeting for DINOv3 ViT layers: confirm module
  names by inspection, do not guess.
- The 7B download (~13 GB) goes into the existing G: HF cache; check G:
  free space first.
- Reuse patterns (crop pipeline, EMA machinery, guards) by COPYING from
  ssl_hdc_pilot scripts into the new pilot folder — never by importing
  across the quarantine line.

## Explicit non-goals

No thesis claims. No comparison against the S/B/L ladder results. No
quantization-quality study. No 8-bit variant unless 4-bit passes K1–K3
first. No hyperparameter sweeps. No hardware upgrades justified by this.

---

## 2026-07-19 — PROJECT MOVED

This side pilot graduated to its own fully independent repository:
`G:\QLORA-VISION` (binding spec: `docs/PILOT_SPEC.md` there, v2 — satellite
oracle design, corrected novelty claim per the 2026-07-19 literature
review, projection-head fix). This file is retained as the historical v1
record only; do not implement from it. The quarantine is now physical:
separate repo, separate data, separate venv; only the HF weight cache is
shared. Thesis work still preempts the side project everywhere.
