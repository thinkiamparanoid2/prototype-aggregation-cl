# Thesis Direction 1: Quantum-Enhanced Hyperdimensional Computing (QHDC) for Crop Disease Classification

## 0. One-Paragraph Summary
We build a crop disease classifier using **Quantum Hyperdimensional Computing (QHDC)** — a paradigm published for the first time in November 2025 that maps the algebra of classical Hyperdimensional Computing (binding, bundling, permutation, similarity) directly onto native quantum operations (superposition, phase oracles, Quantum Fourier Transform, state fidelity), instead of just running classical HDC math near a quantum computer. No one has applied QHDC to a real biological/agricultural classification problem yet — the original paper validates it only on symbolic reasoning and generic toy classification tasks. That gap is our contribution. This direction reuses the VQC (Variational Quantum Circuit) pipeline, environment, and quantum-simulation debugging experience the team already has, rather than starting a new toolchain from zero.

---

## 1. What This Actually Is

### 1.1 The two ingredients, separately

**Hyperdimensional Computing (HDC) — the "brain-inspired fingerprint" idea**
- Every input (a pixel, a sensor reading, a feature) is represented as a hypervector: a random, high-dimensional (typically ~10,000-D) binary or bipolar vector.
- Three operations do all the work:
  - **Binding** (element-wise multiplication / XOR): combines two hypervectors into a new, unrelated one. Used to attach a value to a feature (e.g. `position ⊗ pixel_intensity`).
  - **Bundling** (element-wise majority-vote addition): merges many hypervectors into one "average" hypervector. Used to build a class prototype from many training examples.
  - **Permutation** (cyclic shift): encodes order/position.
- Classification = encode a new sample into a hypervector, compare it (cosine similarity or Hamming distance) against each class's prototype vector, pick the closest.
- Key properties: one-shot/few-shot learning (no backprop), robust to bit-level noise/corruption, cheap to update (add a new example to a prototype — no retraining).

**Quantum Hyperdimensional Computing (QHDC) — what's actually new**
- Published by Cumbo et al. (Cleveland Clinic / Case Western), arXiv:2511.12664, "Quantum Hyperdimensional Computing: a foundational paradigm for quantum neuromorphic architectures," npj Unconventional Computing (2026).
- Instead of encoding hypervectors as classical bit arrays, QHDC encodes them as **quantum states**. Each of the three HDC operations gets a direct quantum-native implementation:
  | Classical HDC operation | QHDC quantum implementation |
  |---|---|
  | Hypervector | Quantum state (superposition over basis states) |
  | Bundling (majority-vote sum) | Linear Combination of Unitaries (LCU) + Oblivious Amplitude Amplification (OAA) — a quantum-native averaging process |
  | Binding | Quantum phase oracles |
  | Permutation | Quantum Fourier Transform (QFT) |
  | Similarity comparison | Quantum state fidelity, measured via a Hadamard Test |
- Validated on a 156-qubit IBM Heron r3 processor for symbolic analogical reasoning and supervised classification — but only on generic/toy datasets, not on a real applied problem.
- Reported result: up to ~500x speedup over other quantum classifiers in cross-validation timing (simulator/hardware comparison), because QHDC's operations map to shallow, low-depth circuits — unlike variational quantum classifiers (VQC/QNNs) which need iterative depth-heavy classical-quantum optimization loops.

### 1.2 What we are contributing
**First applied QHDC classifier on a real biological/agricultural dataset**, with a rigorous three-way comparison: classical HDC vs. QHDC vs. a standard CNN baseline, on crop disease image data. This is a legitimate "first" claim (not manufactured novelty from picking a random unexplored domain) because:
- The QHDC paper itself is 8 months old at most and explicitly frames future work as "applying to more challenging cognitive and biomedical problems."
- No applied QHDC paper currently exists in agriculture, plant pathology, or (as far as our search found) any applied imaging domain at all.
- It is mathematically distinct from what the team already built (VQC = variational, iterative, gradient-based; QHDC = algebraic, one-shot, non-iterative) — so this isn't "redo the VQC thesis with new branding," it's a genuinely different quantum ML paradigm, but one that reuses the same circuit-simulation and quantum-debugging skills.

---

## 2. Why This Matters / The Novelty Argument (for the thesis defense and paper intro)

1. **Timing**: QHDC is new enough that "first applied to domain X" is a real, defensible claim, not a stretch.
2. **Distinct from VQC**: Standard VQC/QNN approaches require iterative gradient-based optimization on quantum hardware, which is slow, noisy, and prone to barren plateaus. QHDC is non-iterative and algebraic — one encoding pass, no training loop. This is a genuinely different value proposition than "we ran a VQC on crops," and is worth stating explicitly to the committee.
3. **Practical relevance**: because QHDC circuits are shallow (low depth, few CNOTs — e.g. the paper reports depth-1 diagonal gates for prototype prep, depth-4/2-CNOT permutation), it is one of the few quantum ML approaches that could plausibly run on real near-term hardware rather than only in simulation — this is a genuine "near-term quantum utility" story, which is currently a hot framing in quantum ML venues.
4. **Honest limitation to state upfront**: current QHDC hardware demonstrations are capped around D ≤ 128 dimensions due to resource constraints. Classical HDC uses D ~ 10,000. This means our real image classification will run primarily on **classical simulation of the QHDC circuits**, with a small-scale real-hardware (or realistic noisy-simulator) demonstration as a proof-of-concept subsection — not "full QHDC replaces our CNN on real quantum hardware." State this clearly in the paper; don't oversell it.

---

## 3. How It Should Be Implemented

### Phase 1 — Environment & Baselines (Weeks 1–2)
- Reuse the existing VQC pipeline's environment (Python, GPU/CUDA setup already resolved).
- Add: `Torchhd` (classical HDC library) and a quantum simulation framework — `Qiskit` (since the QHDC paper used IBM hardware/Qiskit) or `PennyLane` if the team prefers a more PyTorch-native quantum simulation interface.
- Pick the dataset (see Section 6).
- Train two baselines:
  - A standard CNN (e.g. ResNet-18/MobileNet transfer learning) — the "what everyone already does" baseline.
  - A classical HDC model (Torchhd) — the "what HDC alone gets you" baseline, no quantum involved yet.

### Phase 2 — Classical HDC Encoder (Weeks 2–4)
- Build the item memory (random base hypervectors for spatial/feature positions).
- Build the encoder: map image features (either raw pixel patches or, more realistically, embeddings from a pretrained feature extractor — see note below) into hypervector space via binding + permutation.
- Build class prototypes via bundling.
- Validate: check base vectors are near-orthogonal, confirm classification works at a reasonable accuracy versus the CNN baseline.
- **Practical note**: raw-pixel HDC encoding tends to underperform on complex natural images. A stronger, still-legitimate design is to first pass images through a small pretrained CNN/ViT to get a compact embedding, then encode *that* embedding into hyperdimensional space. This is common practice in recent HDC literature and doesn't compromise the novelty claim, since the novelty is in the QHDC layer, not the classical encoder.

### Phase 3 — QHDC Implementation (Weeks 4–8) — the core technical contribution
This is where most of the team's effort and debugging time should go.
1. **State encoding**: implement the mapping from a (dimension-reduced, D ≤ 128) hypervector to a quantum state (amplitude or basis encoding — follow the mapping described in the QHDC paper).
2. **Bundling as LCU + OAA**: implement the quantum-native averaging circuit that bundles multiple class-sample states into one prototype state.
3. **Binding as phase oracles**: implement feature-value binding via phase oracle circuits.
4. **Permutation as QFT**: implement positional/sequence encoding via the Quantum Fourier Transform.
5. **Similarity via Hadamard Test**: implement fidelity-based similarity measurement between a query state and each class prototype state.
6. Validate each operation individually against its classical HDC counterpart on small synthetic examples before assembling the full pipeline (this is the standard "test each primitive before the full circuit" debugging discipline — critical given how hard quantum bugs are to diagnose).

### Phase 4 — Evaluation & Novelty Experiments (Weeks 8–12)
Run all three models (CNN, classical HDC, QHDC) on:
1. **Standard accuracy/F1 benchmark** on held-out test data.
2. **Noise robustness test**: corrupt input at 10%/30%/50% — compare degradation curves across all three models (this restates the known HDC property but now also shows whether QHDC's quantum-native noise characteristics differ from classical HDC — that comparison itself is new).
3. **Circuit depth / resource cost comparison**: benchmark QHDC circuit depth and gate count against a standard VQC/QNN classifier (this directly demonstrates the "shallow circuit, near-term feasible" claim from the original paper, now grounded in a real applied task).
4. **Small-scale real-hardware demonstration** (optional, high-value if time allows): run a reduced-dimension version of the pipeline on real IBM quantum hardware (or a realistic noisy simulator) for a proof-of-concept subsection, mirroring how the original QHDC paper validated on 156 qubits.
5. **Timing comparison**: classical HDC vs. QHDC (simulated) vs. VQC training time — this is where the "500x speedup" claim from the original paper either replicates or doesn't in your applied setting, which is itself a publishable finding either way.

---

## 4. Team Split Suggestion (4 people)
- **Person A**: classical HDC baseline + encoder (Phase 1–2), owns `Torchhd` pipeline.
- **Person B**: quantum circuit implementation of the four QHDC primitives (Phase 3), owns `Qiskit`/`PennyLane` code — likely the most technically demanding role, best suited to whoever debugged the VQC pipeline before.
- **Person C**: CNN baseline + dataset engineering + noise-corruption experiment harness (Phase 1, 4).
- **Person D**: evaluation infrastructure, benchmarking/timing harness, results visualization, and paper/thesis writing coordination (Phase 4, ongoing).
- All four should co-review Phase 3, since a bug there is "incredibly hard to tell if it's code or math" — the same risk Gemini flagged for the sheaf network idea applies here too, just to a smaller, more contained piece of the pipeline.

---

## 5. Risks & Mitigations
| Risk | Mitigation |
|---|---|
| QHDC circuits don't scale to real image dimensionality | Use pretrained embeddings + dimensionality reduction (PCA/autoencoder) to get D ≤ 128 before quantum encoding |
| Quantum simulation is slow at scale | Cap experiments to simulator-feasible sizes; treat any real-hardware run as a small proof-of-concept, not the main result |
| Debugging quantum circuits is hard to distinguish "code bug" vs "math bug" | Unit-test each of the four primitives (bundling, binding, permutation, similarity) against classical HDC ground truth before integration |
| Overselling "runs on a quantum computer" | Be explicit in the paper about what's simulated vs. hardware-validated — reviewers respond very badly to overclaiming in quantum ML |
| Timeline slip | Classical HDC + CNN baselines (Phases 1–2) are low-risk and can be finished in the first 3–4 weeks regardless of how Phase 3 goes, so there's always a fallback deliverable |

---

## 6. Dataset Options (to decide as a team)
- A standard public plant disease image dataset (e.g. PlantVillage or similar) for the primary benchmark — well-established, lets reviewers sanity-check results against known baselines.
- If available, the same crop-image data already being used for the VQC thesis, for continuity and direct comparison against the existing quantum pipeline's results.

---

## 7. Key References
- Cumbo, F. et al. "Quantum Hyperdimensional Computing: a foundational paradigm for quantum neuromorphic architectures." arXiv:2511.12664 (Nov 2025); published in *npj Unconventional Computing* (2026).
- Kanerva, P. "Hyperdimensional computing: an introduction to computing in distributed representation with high-dimensional random vectors." *Cognitive Computation* 1(2), 2009.
- `Torchhd` — classical HDC library (PyTorch-based).
- Qiskit / PennyLane — quantum circuit simulation frameworks.

---

## 8. Bottom Line
This direction is genuinely novel (first applied QHDC on a real domain), reuses infrastructure and skills the team already has (quantum simulation, circuit debugging from the VQC project), and has a built-in fallback (classical HDC + CNN baselines are low-risk and deliverable early). The main execution risk is concentrated in Phase 3 (the quantum primitive implementations), which is exactly where the team should front-load review and pairing time.
