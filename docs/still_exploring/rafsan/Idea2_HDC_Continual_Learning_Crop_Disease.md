# Thesis Direction 2: Continual On-Device HDC Learning for Crop Disease Detection

## 0. One-Paragraph Summary
Instead of demonstrating HDC's noise-robustness (a known, textbook property — not a real contribution), we build a system around HDC's genuinely under-exploited advantage: **prototypes can be updated instantly, with a single new example, using plain vector addition — no retraining, no backprop, no catastrophic forgetting of other classes.** We simulate a realistic field-deployment scenario where new or ambiguous disease cases show up over time, and the model has to incorporate them on the fly, on-device, while a farmer or agronomist in the loop occasionally confirms a low-confidence label. We benchmark this against how badly a standard CNN degrades or needs full retraining in the same streaming/continual scenario. The contribution is a genuinely open, acknowledged problem in deployed agricultural AI (models failing on novel disease variants they weren't trained on) rather than a replay of a known algebraic property.

---

## 1. What This Actually Is

### 1.1 The property we're exploiting
Classical HDC training works like this:
- Each class (e.g. "Healthy," "Blight," "Rust") has a **prototype hypervector** — a bundled (majority-vote-summed) combination of all training examples' encoded hypervectors for that class.
- Adding a new labeled example to a class is just: `prototype_new = prototype_old + encode(new_example)`, then re-binarize if using binary HDC.
- This is **not** how neural networks work. A CNN has no equivalent operation — adding one new training example generally means re-running (or at least fine-tuning) the whole network, and fine-tuning on new classes/examples risks **catastrophic forgetting**: the model gets worse at the old classes while learning the new one.
- HDC's bundling operation is commutative and incremental by construction — the prototype for "Healthy" doesn't need to know anything changed about "Blight" when a new Blight example comes in, and vice versa.

### 1.2 The actual problem this solves (why it's not just a toy exercise)
In real field deployment:
- New disease strains, or diseases not present in the original training set, appear over a growing season.
- Labeled data trickles in slowly (an agronomist confirms a diagnosis days or weeks after a photo was taken), rather than arriving as a full labeled dataset up front.
- Retraining a CNN from scratch every time new labels arrive is computationally expensive and operationally impractical for edge devices (the same "edge-native" framing from the original HDC proposal) — and *incremental* fine-tuning of a CNN on just the new data is well known to cause forgetting of the original classes.
- This is a documented, real, currently-unsolved friction point in deployed agricultural AI systems — not a problem we invented to justify HDC.

### 1.3 What "novelty" actually means here
The original noise-robustness demo (corrupt 30% of bits, show the model still works) tests a property that is guaranteed by the HDC construction itself — high-dimensional distributed representations are robust to bit corruption by design, and this has been shown many times in the HDC literature. A reviewer who knows HDC will read that experiment as a sanity check, not a finding.

Continual/streaming-class learning is different: it's an evaluation protocol (not a known theorem), it requires you to design the experiment (how classes stream in, how confidence thresholds trigger human-in-the-loop labeling, how forgetting is measured), and the comparison against a CNN under the identical protocol is where the actual empirical contribution lives. This is closer to how real continual-learning papers are structured and evaluated.

---

## 2. How It Should Be Implemented

### Phase 1 — Environment & Baseline Setup (Weeks 1–2)
- Set up `Torchhd` (PyTorch-based HDC library).
- Choose the dataset (see Section 6) and define a **class-incremental split**: e.g. start training with 3 disease classes, hold out 1–2 more classes to be "discovered" later during the streaming phase.
- Train a static baseline CNN on the initial 3 classes only, and a static HDC model on the same initial 3 classes, to confirm both reach reasonable starting accuracy.

### Phase 2 — Build the HDC Continual Pipeline (Weeks 3–5)
1. **Item memory + encoder**: as in a standard HDC pipeline — random base hypervectors for feature positions, encoder mapping input (ideally a pretrained embedding, not raw pixels, for reasonable accuracy) into hyperdimensional space.
2. **Initial prototype training**: bundle the initial 3 classes' training examples into 3 class prototypes.
3. **Confidence scoring**: implement a similarity-margin confidence measure — when a new sample's top-1 and top-2 class similarity scores are close, flag it as "low confidence."
4. **New-class registration**: implement the mechanism for creating a brand-new prototype the first time a genuinely new class label is confirmed (this is the "instant new class" test from the original HDC proposal, but embedded in a realistic streaming pipeline instead of a standalone demo).
5. **Incremental update**: implement `prototype += encode(new_labeled_example)` for both existing-class corrections and new-class additions.

### Phase 3 — Human-in-the-Loop Simulation (Weeks 5–7)
1. Simulate a stream of incoming samples across a time-ordered sequence (mix of the 3 known classes plus the 1–2 held-out "new" classes appearing partway through the stream).
2. When confidence is low (per Phase 2's threshold), simulate a "farmer/agronomist confirms label" event — this simulates the human-in-the-loop correction with a small delay, mirroring real deployment latency.
3. Update the relevant prototype immediately upon confirmation; no retraining pass.
4. Log accuracy over time as the stream progresses, plus a breakdown of how many labels were requested (i.e., how much human effort the system needed).

### Phase 4 — Build the CNN Comparison Pipeline (Weeks 5–8, parallel to Phase 3)
This is the core comparison, so it needs to be done carefully and fairly:
1. Implement the **naive CNN baseline**: fine-tune on newly confirmed labels as they arrive, using standard incremental fine-tuning (small learning rate, few epochs) — this is the "what most people actually do today" baseline.
2. Implement at least one **standard continual-learning CNN baseline** for a fair fight — e.g. Elastic Weight Consolidation (EWC) or a simple replay-buffer approach — so the thesis isn't just "beating an undefended strawman."
3. Run the identical streaming protocol from Phase 3 through both CNN variants.
4. Measure: accuracy on original classes over time (forgetting curve), accuracy on new classes as they're introduced, and total compute/time cost of each update step.

### Phase 5 — Evaluation & Writeup (Weeks 9–12)
Core comparison table/plots across HDC-continual, naive CNN fine-tuning, and continual-learning CNN baseline (EWC/replay):
1. **Forgetting curve**: accuracy on originally-known classes, tracked over the length of the stream, for all three methods.
2. **New-class uptake speed**: how many examples of a new class are needed before accuracy on that class stabilizes, for all three methods.
3. **Update cost**: wall-clock time and compute cost per incremental update (this should be dramatically in HDC's favor and is an easy, clean number to report).
4. **Human-effort cost**: how many low-confidence queries the system generated per method, as a proxy for the burden placed on the farmer/agronomist doing the labeling.
5. **(Optional, if time allows) Combine with Idea 1**: since bundling is the operation being exploited here and it also has a direct QHDC analogue, this pipeline could later be extended into a "does the incremental-update advantage survive under QHDC too?" follow-up — worth a paragraph in future work even if not implemented.

---

## 3. Team Split Suggestion (4 people)
- **Person A**: HDC encoder + prototype/confidence/incremental-update pipeline (Phase 1–2), owns `Torchhd` code.
- **Person B**: human-in-the-loop stream simulator + logging harness (Phase 3) — this is the experiment-design-heavy role; needs to work closely with whoever is writing the evaluation section.
- **Person C**: CNN baselines, including implementing EWC or a replay buffer correctly (Phase 4) — this is the most implementation-heavy non-HDC role and benefits from prior deep learning experience.
- **Person D**: evaluation metrics, plots, statistical comparison, and paper/thesis writing coordination (Phase 5, ongoing).

---

## 4. Risks & Mitigations
| Risk | Mitigation |
|---|---|
| Raw-pixel HDC encoding underperforms on real images, making the whole comparison look bad regardless of the continual-learning story | Use a pretrained CNN/ViT embedding as the HDC encoder's input rather than raw pixels — this is standard practice and doesn't affect the novelty claim |
| CNN baseline is a strawman, weakening the comparison's credibility | Implement at least one real continual-learning method (EWC or replay buffer) as the CNN's best shot, not just naive fine-tuning |
| Streaming protocol design is arbitrary/unconvincing | Base the class-incremental split and stream ordering on a defensible real-world story (e.g., "these two diseases are documented as emerging later in the season" or similar), and document the protocol clearly so it's reproducible |
| Confidence-threshold tuning for the human-in-the-loop trigger feels ad hoc | Report results across a few threshold settings, not just one cherry-picked value, and discuss the accuracy/human-effort tradeoff explicitly |
| Timeline slip | Phase 1–2 (core HDC pipeline) is low-risk and can be done in the first 4–5 weeks independent of how the CNN baseline comparison goes |

---

## 5. Dataset Options (to decide as a team)
- A plant disease dataset with enough distinct classes to support a meaningful class-incremental split (need at least 4–5 classes total: a few "known from the start," a few "introduced later" in the stream).
- If the same dataset used for the VQC/QHDC thesis work is reused here, that gives a nice angle for the thesis as a whole: one shared dataset, multiple paradigms (VQC, QHDC, HDC-continual) compared under a consistent evaluation setup.

---

## 6. Key References
- Kanerva, P. "Hyperdimensional computing: an introduction to computing in distributed representation with high-dimensional random vectors." *Cognitive Computation* 1(2), 2009.
- Hernández-Cano, A. et al. "OnlineHD: robust, efficient, and single-pass online learning using hyperdimensional system." (2021) — directly relevant prior work on HDC's online-update properties, worth reading closely before implementation.
- Kirkpatrick, J. et al. "Overcoming catastrophic forgetting in neural networks" (EWC) — for the CNN continual-learning baseline.
- `Torchhd` — classical HDC library (PyTorch-based).

---

## 7. Bottom Line
This direction targets a real, acknowledged deployment problem (models degrading against novel disease variants encountered after training) using the one HDC property that's actually under-exploited in the literature (cheap incremental prototype updates), rather than re-demonstrating a known robustness property. It's lower quantum/math risk than Idea 1, doesn't require any new quantum tooling, and produces clean, easy-to-defend numbers (update cost, forgetting curves) that make for a strong thesis defense narrative even if the underlying accuracy numbers aren't record-breaking.
