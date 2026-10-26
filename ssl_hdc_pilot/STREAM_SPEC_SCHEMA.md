# Continual Stream Spec v1

The official simulator reads exactly `ssl_hdc_pilot/stream_spec.json`. The
user generates that file from the supervisor-approved Option-A directory with
`scripts/build_stream_spec.py`; the locked Section 6a values are constants,
not command-line options. Its required structure is
`ssl_hdc_pilot/stream_spec.schema.json`, and the runner validates the same
contract directly.

An official invocation hard-errors when the canonical spec is missing, when
the file has `smoke=true`, when a different spec path is supplied, or when the
trusted split lacks supervisor-approved canonical provenance. Synthetic tests
use temporary specs with `smoke=true` and must invoke the explicit `--smoke`
path. A smoke spec can never enter the official path.

## Stream construction

- `initially_known_classes` and `class_introductions` must exactly cover the
  labels present in the supplied trusted split. The NPZ is label-free and may
  contain extra image IDs, but every split ID must have one embedding.
- Generated official specs store an auditable 25% known-class holdback manifest
  for every stream seed. IDs are selected by a stable SHA-256 rank over
  `{seed, class, image_id}`; indivisible per-class counts use floor. Every other
  known-class training row forms the initial training set. Legacy synthetic
  smoke specs may use `bootstrap.confirmed_per_class`.
- `class_introductions[].at_step` is zero-based after bootstrap. At that step
  the class becomes eligible for sampling; before it, the class is unavailable.
  Multiple classes may be activated at the same step.
- `seeds.stream_order` is the complete preregistered ordering-seed sweep.
  Within each class, sorted image IDs are shuffled by each seed, and the same
  RNG selects among active classes with remaining examples. The runner rejects
  an infeasible schedule instead of silently moving an activation.
- `confidence_thresholds` is the complete preregistered sweep. Every learner
  uses the same low-confidence rule: request a label when the top-1 minus
  top-2 cosine-similarity score margin is strictly below the threshold and
  budget remains. The simulator alone sees hidden labels; a learner receives
  one only through bootstrap or a confirmed request event.
- The first confirmed occurrence of each novel class is a registration event
  even if its margin is not below the threshold. That event and every
  low-confidence confirmed-label event each consume one budget unit; a single
  event satisfying both conditions is counted once.

## Learners and updates

All six learners run for every threshold on the identical ordered stream and
frozen embedding NPZ. The embedding dimension is inferred at runtime; 384,
768, and 1024 dimensions follow the same code path. No learner accepts images
or an encoder.

- HDC uses the spec's projection seed and the fixed 10,000-dimensional bipolar
  space. A confirmed vector is added to a real-valued class accumulator; the
  exposed prototype is re-binarized after every addition. Exact-zero ties keep
  the previous prototype bit. It has no optimizer, backpropagation, or epochs.
- NCM keeps an online running mean of each class's per-sample L2-normalized
  embeddings and classifies by cosine similarity.
- RanPAC uses HDC's projection seed and dimension with a frozen standard
  Gaussian projection and ReLU, then keeps online class means and classifies by
  cosine similarity. It adds no decorrelation or ridge step.
- Naive, online EWC, and reservoir replay share one linear/MLP head definition,
  optimizer settings, confidence calculation, registration behavior, and
  label accounting. Naive trains only on the current confirmed event. EWC
  consolidates an online diagonal Fisher after each event and penalizes drift
  from consolidated parameters. Replay trains on the current event plus a
  seeded Algorithm-R reservoir sample; buffer size comes only from the spec.
- A class is registered only when its first confirmed novel label reaches the
  learner. A high-confidence mistake does not reveal a hidden label.

## Outputs and metric definitions

Each `{embedding, stream seed, learner, threshold}` run gets a tagged history
CSV and summary JSON. History records prediction, similarity margin, top
probability, request/denial flags, registration, measured update wall-time,
correctness, and per-class accuracy over each class's latest `rolling_window`
encounters.

The summary reports two predictive blocks using shared logic from
`common/thesis_eval.py`: `prequential_metrics` scores predictions made as
the stream evolved, while `final_metrics` performs a no-update pass over the
full supplied trusted split after the last stream update. Both report accuracy
together with macro-F1, macro-recall, per-class recall, per-stage recall when
stage suffixes are available, and ECE. It also includes:

- forgetting: at every step, macro-average the latest rolling per-class
  accuracies for initially-known classes; final forgetting is peak minus final;
- new-class uptake: class-specific examples through the first point where the
  spec's number of consecutive sliding windows reach its minimum accuracy;
- update compute: count, total, mean, median, and p95 wall-time across bootstrap
  and confirmed stream updates;
- label budget: bootstrap charge, stream requests, total charge, remaining
  budget, and exhaustion state.

Real wall-time fields naturally vary. Given a fixed spec, artifacts, device,
and deterministic test clock, ordering, predictions, requests, updates,
metrics, and serialized outputs are byte-identical.


## 2026-08-02 prototype-baseline and replay-audit amendment

- `ncm` keeps an online running mean of each class's per-sample
  L2-normalized embeddings and uses the same top-1 minus top-2 cosine-similarity margin as HDC.
- `ranpac` uses the HDC projection seed and dimension, a frozen standard
  Gaussian projection, ReLU, online class means, and cosine-margin inference.
  This controlled RanPAC-style ablation has no decorrelation or ridge step.
- `--learners` accepts a guarded comma-separated subset; the complete default
  is `hdc,ncm,ranpac,naive,ewc,replay`. `--ewc-only` remains for its
  historical correction block and cannot be combined with `--learners`.

No official run was launched by the agent.
