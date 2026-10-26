# Idea 2 — Assessment, Readiness, and Risk Notes

*Review notes, 2026-07-15. Companion to
`Idea2_HDC_Continual_Learning_Crop_Disease.md` and `Idea2_SSL_Extension.md`.
All repo numbers below were verified against the actual codebase and data,
not estimated.*

## Verdict

Idea 2 is the stronger of the two proposals in this folder: high merit as an
**applied / protocol contribution**, low execution risk, and roughly a third
of its engineering already exists in this repo. Its critical path is the
EWC/replay CNN baseline; its critical framing rule is "protocol study, not
new algorithm."

## Merit — what holds up

- The §1.3 argument is the core strength: HDC noise-robustness is a
  *theorem* (a reviewer reads that demo as a sanity check), while
  continual/streaming learning is a *protocol you design and measure* — the
  contribution lives in the evaluation, which is how real continual-learning
  papers are judged.
- The target problem (novel disease variants appearing mid-season, labels
  trickling in via agronomist confirmation) is real and documented, not
  invented to flatter HDC.
- The headline metrics — forgetting curves, per-update compute cost,
  human-label budget — survive a defense without winning any accuracy race.

## Honesty corrections the thesis must absorb

1. **The algorithmic content is NOT novel.** OnlineHD (cited in the idea
   doc) and a body of HDC lifelong/continual-learning papers already exist.
   The defensible claim: *first rigorous deployment-protocol study of HDC
   continual learning in agriculture, with human-effort accounting and fair
   CNN continual-learning baselines.* Framed as a new algorithm, it gets
   shredded.
2. **HDC's incremental update is not literally forgetting-free.** Bundled
   prototypes saturate and interfere as examples accumulate, especially
   after re-binarization. Measure it; don't assert immunity.
3. **The confidence-margin trigger detects *uncertainty*, not *novelty*.**
   "New disease" vs "known disease, low confidence" is open-set recognition
   — hard. The human-in-the-loop closes it (the human supplies the "it's
   new" fact), which means the label-request budget IS the experiment.
   Thresholds must be swept and reported, never single-point tuned.
4. **The embedding-attribution attack** ("the CNN backbone does the real
   work") is answered by (a) metrics that isolate the *update mechanism*
   (update cost, forgetting), and (b) the SSL extension's label-free
   backbone, through which no supervised signal flows.

## Readiness — what already exists in this repo (verified)

| Idea 2 phase (12-week plan) | Status via this repo |
|---|---|
| P1: env, dataset, splits, baselines | **~70% done.** GPU envs exist; 3-seed trusted splits exist (`full_qcnn_pilot/splits/`); 5 classes meets the class-incremental minimum (3 known + 2 held-out). Gaps: Torchhd install, one ResNet-transfer baseline. |
| P2: HDC encoder + prototypes | **~25% done — the risky quarter is solved.** "Use pretrained embeddings, not raw pixels" is already implemented and validated: `full_qcnn_pilot/scripts/build_codes.py` extracts frozen ResNet-18 512-d features, GPU-batched, with a logistic-regression quality probe. The HDC layer on top is new but standard Torchhd work. |
| P3: stream simulator | New code; metrics module, seed protocol, and prereg/logging discipline transfer wholesale. |
| P4: CNN continual baselines | **New.** Naive fine-tune trivial; EWC or replay is the one substantial implementation job in the whole idea. |
| P5: evaluation + writeup | Reporting/figure/log infrastructure and prereg methodology transfer directly. |

**Net: ~35% of the engineering exists, and it is the foundational third**
(clean labels, trusted splits, validated embedding pipeline, evaluation
discipline). Realistic timeline: **6–8 weeks**, dominated by P4.
The SSL backbone ladder adds +2–3 weeks (see `Idea2_SSL_Extension.md`).

## Dataset reality (verified numbers)

- Trusted pool: **1,794 true-labeled images** (from the 5,767 in
  `Data/test/`), 5 classes. Train split: healthy 706; four diseases 85–100
  each (~145–165 per disease across all partitions).
- **The stream is short.** Novel classes have ~150 examples total. Feasible
  — HDC's few-shot uptake is flattered by this — but forgetting curves will
  be noisy: use all 3 seed-splits AND multiple pre-registered stream
  orderings.
- **Heavy imbalance** (healthy ≈ 65%). Stream composition and the confidence
  threshold interact with it; report macro-F1, never accuracy.
- **Label provenance is already solved for this idea.** The clean-label path
  (true-labeled images only, trusted splits) is exactly what a
  human-in-the-loop simulation needs: every simulated "agronomist confirms"
  event is backed by a real label. The 57k pseudo-labeled images are usable
  only as label-free pixels (SSL) — see `Idea2_SSL_Extension.md`, including
  the verified zero-overlap leakage firewall.

## Fastest concrete start (de-risks P2 in a weekend)

Fork `build_codes.py`'s feature extraction into a shared module,
`pip install torch-hd`, and prototype the 3-class static HDC baseline
against the existing logistic-regression probe. If pursuing the SSL
extension: run the probe-first gate (DINOv3 frozen embeddings through the
same probe) before building any stream harness.

## Appendix — Idea 1 (QHDC) in one paragraph

Higher variance, lower floor. The "first applied QHDC" claim is real
(arXiv:2511.12664 is months old), but: (a) hardware/simulation caps QHDC at
D ≤ 128 vs classical HDC's ~10,000 — near-orthogonality, and with it HDC's
capacity, degrades badly at that width, so the headline may become "our
quantum method loses to its own classical version"; (b) the paper's ~500×
speedup is circuit-depth vs iterative VQC on hardware and will NOT replicate
on simulators — do not bank on it; (c) this team's own QCNN result
(0.448 vs quadratic-null 0.726) is a live warning for quantum-as-the-model
approaches. If pursued, reframe around **noise characteristics of
quantum-native vs classical bundling** (ties into the corruption-robustness
theme) rather than accuracy or speed, and sequence it AFTER Idea 2 — §5 of
the Idea 2 doc already notes bundling has a direct QHDC analogue, so Idea 2
can grow into Idea 1; the reverse is not true.
