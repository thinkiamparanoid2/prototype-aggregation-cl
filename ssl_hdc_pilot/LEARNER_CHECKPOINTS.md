# Continual learner checkpoints — development implementation

Implemented 2026-09-10 under the user's authorization of next-steps 1–2.
This is state-preservation infrastructure, not an approved experiment or a frozen
evaluation protocol. The [core draft](../docs/TODO/HDC_CORE_EVALUATION_PROTOCOL_DRAFT_2026-09-09.md)
still has open A1–A4 gates. No official run was launched by the agent.

## Public interface

`ssl_hdc_pilot.scripts.learner_checkpoint` exposes:

- `save_learner_checkpoint(path, learner, context=..., shared_projection_root=None)`: write a detached NPZ
  snapshot; the filename must end in `.npz`. Existing files are never overwritten.
- `load_learner_checkpoint(path, device="cpu", restore_rng=False, shared_projection_root=None)`: return a
  `LearnerCheckpoint` with `.learner`, `.context`, and `.metadata`. The loaded
  learner supports the existing `predict`, `update`, and registration interfaces.
- Global Python, NumPy, and Torch RNG snapshots are saved. Loading leaves caller
  RNGs unchanged by default; `restore_rng=True` explicitly restores saved global
  states. The learner's private replay RNG is always restored. CUDA RNG snapshots
  cover already initialized devices; explicit CUDA restoration requires the same
  visible device count. Deferred, uninitialized CUDA RNG state is not exported.

Exact prediction and continued-update equality is tested on synthetic CPU data.
Accepting a device argument does not establish cross-device or cross-version
bitwise reproducibility. CPU reload does not reseed or queue seeding of CUDA.

## State coverage

| Learner | Preserved state |
| --- | --- |
| HDC | Fixed projection, real accumulators, current bipolar prototypes including tie history, ordered registry and registered-but-empty classes |
| NCM | Ordered registry, means, counts, input/prototype dimension |
| RanPAC-style | NCM state plus the actual fixed projection |
| Naive | Linear/MLP configuration, weights, ordered output mapping, learner seed, per-module training modes, gradient flags and stored gradients |
| EWC | Neural state plus means, penalty Fisher, and running Fisher |
| Replay | Neural state plus buffer capacity, ordered embedding/label items, total seen count, and private sampling/replacement RNG |

The registered per-update SGD optimizer is recreated without momentum by the
existing learner; there is no persistent optimizer state to restore. All actual
projection values are saved, not regenerated from a presumed seed. Default API
saves and existing checkpoints are self-contained. Since 2026-09-14, new budget
longitudinal captures and benchmark runs use the shared storage described below.

Schema version 1 is tagged `learner_resumable`. It is **not a resumable stream
controller**: the caller must supply subsequent events and enforce the budget,
registration policy, ordering, and split guards. Benchmark prediction-only
exports have a separate artifact kind and are rejected by this resumable loader.
No checkpoint load scores any dataset automatically.

## Shared projection storage (2026-09-14)

The logical learner schema remains version 1. An independent physical envelope
`_storage_json` identifies `shared_projection_npz`, version 1. HDC/RanPAC archives
omit the repeated projection and reference `projection_blobs/<digest>.npz` inside
the explicitly supplied run root. The digest covers dtype, shape and C-order
array bytes. The blob contains the actual float32 projection, not a seed.
Every reuse and load verifies its inventory and content digest; missing, corrupt
or unsupported blobs/envelopes fail before learner construction/RNG restoration.
Identical projections share one blob across heads and snapshot/prediction exports.
Other learners remain self-contained. No global cache or cross-run dependencies.

For compact checkpoints, pass `shared_projection_root=run_directory` on load.
Back up or copy the **entire run directory**, including `projection_blobs`, and
pass its new location after relocation. Individual compact NPZ files are not
standalone backups. References cannot contain arbitrary paths or escape the
explicit root. Historical default-save callers remain compatible and unchanged.

Artifact receipts include each blob; longitudinal validation requires the exact
indexed-checkpoint dependency closure. Benchmark per-file NPZ bytes exclude shared
blobs, while new dependency-closure byte fields include them. Whole-run storage
counts each blob once. Live model/replay tensor and peak-memory definitions do
not change: disk sharing is not a RAM, throughput, or scientific benefit claim.
Production attempt04 accepted2026-09-14:121.87MiB total,90.92% smaller than the
attempt02 directory. See results/A25_COMPACT_BENCHMARK_AUDIT_2026-09-14.md for
verified archive/dependency coverage and the distinction from a controlled
compression-only or live-memory comparison.

## Runner integration

The existing runner accepts `save_final_state=True` in `run_study`, or the opt-in
CLI flag `--save-final-state`. This adds `<run_tag>_final_state.npz` alongside
the usual history and summary, and exposes `StudyArtifact.final_state_npz`.
With the option disabled that field is `None`; no state file is written.

Saving occurs after the complete stream and before the historical training-pool
final evaluation. The callback records the run tag, threshold, seed, spec hash,
ordered bootstrap/stream IDs, completed-stream position, charged labels, confirmed
novel classes, input file hashes, and actual source-file hashes. The latter identify
working code even when it is uncommitted; they are not an approved freeze manifest.
The standalone save API records only the context supplied by its caller.

Existing history/summary formats and default CLI output remain unchanged. Under
a deterministic test clock, enabled and disabled runner paths produce byte-equal
CSV/JSON outputs for all six learners. State saving is outside measured updates;
it is not throughput instrumentation and can still affect a real run's environment.

This flag is not permission to rerun completed runbook blocks. Historical
reconstruction tooling is now implemented in [RECONSTRUCTION.md](RECONSTRUCTION.md).
All 108 official historical-cell fidelity checks passed on 2026-09-11; see the
[audit](results/RECONSTRUCTION_AUDIT_2026-09-11.md). Intermediate snapshots,
equal-budget development, fixed-validation evaluation and benchmark development
are now implemented; full grids, protocol freeze and untouched-test release
remain pending. Existing receipts retain their original code/file hashes.

## Artifact safeguards and limitations

NPZ loading uses `allow_pickle=False`. JSON metadata describes the format,
configuration and ordered registry; array inventory, shape, dtype and SHA-256
checks detect array corruption. Restoration also checks expected dimensions,
finite values, bipolar prototypes, replay occupancy, and neural state coverage.
These checks are not authentication: use only locally trusted artifacts.

Files are serialized to a unique sibling temporary file, flushed, then published
with an exclusive atomic hard link; the temporary name is removed. Final names
are never overwritten. This requires filesystem hard-link support (unsupported
filesystems fail closed). No links connect completed artifacts to each other.
A process crash can leave a temporary file or an unreferenced valid blob; neither
is silently deleted/reused as a successful run. There is no whole-run transaction
or automatic resume. Source results/specification and historical hashes stay
unchanged. New code hashes require fresh provenance; legacy scientific receipt
guards are not waived just because the low-level legacy loader remains compatible.
Binary checkpoints remain Git-ignored; a code commit is not a checkpoint backup.

## Verification boundary

Tests exercise HDC ties and empty classes; NCM/RanPAC means/counts/projection;
linear and MLP neural round trips; new classes before and after saving; EWC
continuation; replay overflow and subsequent sampling; mixed model modes,
gradients, global RNG restoration, overwrite refusal and corrupted arrays.
Runner integration uses generated vectors and an artificial nine-example split.

This proves the tested implementation behaviors, not fidelity of the 108
historical scientific cells, held-out generalization, or a retention/speed claim.
The standard local CPU suite is invoked with `scripts/hdc.ps1 -Action test`.
