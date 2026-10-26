# SSL-HDC Track — Future Extensions (parked ideas, not commitments)

Ideas recorded so they are not lost. Nothing here is pre-registered, scheduled,
or part of the current ladder. Promoting any item requires its own dated
pre-registration section in the binding handoff BEFORE implementation.

## 1. QLoRA-DINO: adapter-based domain adaptation of quantized DINOv3-7B

Recorded 2026-07-19, from the "can we quantize the 7B?" discussion.

- **Idea:** the 6.7B-param DINOv3 ViT-7B/16 cannot be full-parameter
  domain-adapted on consumer hardware (fp16 training states are far beyond
  any single card), but an int4/int8-quantized frozen base (~3.4–6.7 GB)
  with fp16 LoRA adapters trained on top could, in principle, be adapted
  with a DINO-style self-distillation objective — the EMA teacher tracking
  the EMA of the LoRA deltas over the shared quantized base.
- **Why parked, not pursued:** (a) it is a novel, unvalidated training
  method — "QLoRA-DINO" does not exist in the literature we know of, so it
  is a research project, not a config change; (b) activations of a
  4096-dim/40-layer ViT at multi-crop batch sizes remain several GB of
  fp16 regardless of weight quantization, so feasibility on 12 GB is
  unproven even with LoRA; (c) throughput on a consumer card is likely
  days-per-schedule (quantized matmuls save memory, not time); (d) the
  current downstream label budget (~3.4k trusted train samples) cannot
  exploit encoder-quality differences at that scale; (e) quantization
  noise confounds any frozen-7B "encoder-scale reference" measurement.
- **What would make it worth reviving:** a much larger trusted label pool,
  a >=24 GB local card (or the lab 4090 finally confirmed via nvidia-smi),
  or a thesis chapter explicitly about parameter-efficient adaptation of
  vision foundation models — where QLoRA-DINO itself would be the claimed
  contribution and would need its own baselines (frozen 7B fp16 on a big
  card, LoRA-only without quantization, full-parameter ViT-L).
- **Nearest cheap cousin (also unscheduled):** frozen quantized-7B
  embeddings as an appendix encoder-scale reference — rejected for now
  because int4/int8 rounding corrupts exactly the feature-quality signal
  such an appendix would exist to measure.
