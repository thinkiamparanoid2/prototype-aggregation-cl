# HDC literature claim-scope check — 2026-09-15

Bounded primary-source check of OnlineHD, C-FSCIL, LifeHD and ImageHD, not an exhaustive review or a replication. No experiments were run and no historical results were modified. Published benefits are conditional on the algorithm, comparator, hardware and measured operation; they do not guarantee the same ranking for our benchmark.

## OnlineHD

OnlineHD is an adaptive classifier, not plain prototype summation: prediction errors trigger similarity-weighted additions to the correct class and subtractions from the predicted class. The paper explicitly identifies saturation and loss of uncommon patterns under naive accumulation (Sections II–III).

Its efficiency comparison in Section V-C uses optimized binary-vector FPGA implementations versus DNN and baseline HDC. Training gains partly reflect fewer required iterations. It reports iterative/single-pass training speedups of 5.9x/17.1x versus DNN; inference speedup is separately 3.5x. The abstract's 3.5x wording should therefore not be treated as an unambiguous training-only figure. Section V-D separately evaluates precision/platform tradeoffs on Raspberry Pi 3B+ and Kintex-7; CPU efficiency favors shorter, higher-precision vectors. These are not measured NCM/RanPAC frozen-feature stream-throughput comparisons. [OnlineHD, Sections III and V-C–D](https://www.alehc.com/papers/date2021-onlinehd.pdf)

Interpretation: a float32 10,000-dimensional software bundling implementation is not a reproduction of the paper's adaptive update or optimized FPGA configuration. Its performance does not directly confirm or refute those speedup claims.

## C-FSCIL

C-FSCIL combines a meta-learned frozen feature extractor, learned fully connected embedding, and class-prototype memory. Mode 1 averages prototypes without incremental gradients; Modes 2–3 additionally bipolarize or nudge prototypes and retrain the embedding layer. Its evaluated embedding dimension is 512, not 10,000 (Sections 4–5).

Natural-image comparisons include AL-MML, IDLVQ-C, Semantic KD, VAE, F2M and CEC; Omniglot additionally uses ProtoNet. The Omniglot Mode-3 result goes from 87.21% to 85.70% accuracy while growing from 1,200 to 1,623 classes. This is a small measured accuracy drop, not a theorem of zero forgetting. The experiment's expanding-class accuracy is also not our fixed-population forgetting metric. Its compute argument concerns constrained incremental updates, not measured superiority over our NCM/RanPAC controls. [C-FSCIL, Sections 4–5 and Tables 1–3](https://arxiv.org/pdf/2203.16588)

Interpretation: its learned representations and interference-reduction procedures matter; simply invoking high dimensionality does not reproduce them.

## LifeHD and ImageHD: distinct systems and workloads

[LifeHD, Sections7.3 and7.5](https://arxiv.org/html/2403.04759v1) compares against
unsupervised continual neural representation-learning baselines and measures
training latency/energy per batch on Raspberry Pi and Jetson platforms. Its
two-tier working/long-term cluster memory is not our one-prototype-per-class
supervised learner. This does not establish speed superiority over NCM using
already frozen, cached representations.

[ImageHD](https://arxiv.org/abs/2604.21280) specifically implements an AMD Zynq
ZCU104 FPGA streaming architecture with word-packed binary hypervectors,
bitwise operations, bounded cluster management and a quantized CNN front end.
Its hardware results are not guarantees for a float32 PyTorch implementation on
an RTX GPU. HDC is a family of representations and algorithms, not a single
hardware implementation.

## Our implementation and the proper inference

[Our HDC learner](../../ssl_hdc_pilot/scripts/continual_hdc.py) uses a dense
float32 random projection to10,000 dimensions, float32 bipolar vectors,
unweighted accumulation/re-binarization, and cosine similarity. It is not packed
binary inference, OnlineHD adaptive learning, LifeHD cluster management or
C-FSCIL's learned/nudged embeddings. The [locked stream spec](../../ssl_hdc_pilot/stream_spec.json)
uses linear neural heads and one update step; all methods receive cached frozen
features. Avoid describing these controls as repeatedly retrained full CNNs.

NCM also avoids backpropagation and works directly in the input embedding space.
The [benchmark](../../ssl_hdc_pilot/scripts/benchmark_core.py) times all predictions
and acquisitions plus purchased updates: cheap individual updates need not yield
fast full-stream throughput. These are structural differences, not a profiled
causal decomposition of HDC's slowdown; RanPAC-style also projects to high dimension.
Current results test this implementation and protocol, not all possible HDC systems.

## Local wording that needs qualification

The archived exploratory [Idea 2 proposal](../still_exploring/rafsan/Idea2_HDC_Continual_Learning_Crop_Disease.md) opens with an overbroad claim of instantaneous updates and no catastrophic forgetting of other classes. The companion [assessment](../still_exploring/rafsan/Idea2_ASSESSMENT.md) already explicitly rejects literal forgetting-free updates. These are not equivalent to a promise in the curated literature list.

Not overwriting another class's stored prototype does not prove that its predictions remain unchanged: adding a competing class can change the winning similarity. This is a logical distinction, not a new empirical result.
