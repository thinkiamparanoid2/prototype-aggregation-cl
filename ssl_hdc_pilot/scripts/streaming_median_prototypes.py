"""Bounded-memory streaming-median NCM prototype control.

Motivation
----------
``MedianNCMPrototypeLearner`` (``ssl_hdc_pilot/scripts/aggregation_prototypes.py``)
computes an *exact* coordinate-wise median class prototype by storing every
confirmed embedding for a class and re-running ``torch.median`` over the full
stacked list on every single update. Per-class memory and per-update compute
therefore both grow linearly (median compute also needs a sort, so update
cost is worse than linear) with the number of confirmed examples, so no
memory or update-speed claim can be made for it. ``StreamingMedianNCMPrototypeLearner``
below is a control that reproduces the same public interface and the same
raw-embedding, cosine-classification design, but keeps **fixed-size**
per-class state that never grows with the number of confirmed examples.

Algorithm: P^2 (Piecewise-Parabolic quantile estimator)
---------------------------------------------------------
Reference: R. Jain and I. Chlamtac, "The P^2 Algorithm for Dynamic
Calculation of Quantiles and Histograms Without Storing Observations",
Communications of the ACM 28(10), 1985.

P^2 tracks a target quantile (here p = 0.5, the median) online using five
"markers" per scalar stream: the running minimum, the running maximum, the
median estimate, and two flanking markers positioned (in rank) halfway
between the median and each extreme. Each marker keeps (a) its current
height (an estimate of the value at its target rank) and (b) its current
integer position (its rank among all observations seen so far for that
stream). After every new observation, the marker containing it is located,
positions to its right are incremented, each middle marker's *desired*
position (a deterministic, data-independent function of the observation
count) is advanced, and any middle marker whose actual position has drifted
by more than one rank from its desired position is moved one rank via a
parabolic interpolation through its two neighbours (falling back to linear
interpolation if the parabolic estimate would not lie strictly between the
neighbours, which guarantees the five marker heights stay non-decreasing).
The median estimate is always the height of the middle (third) marker.

This implementation keeps one independent P^2 estimator *per output
dimension per class* (the five markers for dimension j only ever see
component j of the confirmed embeddings for that class), because each
embedding dimension has a different marginal distribution. The five
*desired* marker positions and their per-step increments are pure functions
of the observation count (not of the data), so they are shared scalars
across all dimensions of a class; only the marker *heights* and *actual*
integer *positions* are per-dimension, because which marker a given
observation falls under is data-dependent per dimension.

Per-class memory as a function of dimension D
-----------------------------------------------
Once initialised (>= 5 confirmed examples for that class; see "Warm-up"
below), a class's state is exactly:

- ``heights``:   5 x D float64 tensor  -> 40 * D bytes
- ``positions``: 5 x D int64 tensor    -> 40 * D bytes
- ``desired``:   5 float64 scalars     -> 40 bytes  (O(1), not O(D))
- ``increments``:5 float64 scalars     -> 40 bytes  (O(1), constant for p=0.5)

Total: ``80 * D + 80`` bytes, independent of the number of confirmed
examples N. (Float64/int64 are used internally for the marker arithmetic to
control accumulation error over long streams; ``get_prototype`` casts the
returned median vector to float32 to match the embedding dtype used
elsewhere in this codebase.) Contrast with the exact learner, whose stored
state for a class with N confirmed examples is ``N * D`` float32 values,
i.e. ``4 * N * D`` bytes -- unbounded in N.

Update cost: O(D) per confirmed example (a fixed number of elementwise
comparisons/adjustments across the 5 x D marker arrays), versus the exact
learner's cost of re-stacking and re-sorting all N stored vectors per
dimension on every update. Predict cost is O(D x C) for both learners alike
(C = number of registered classes), since classification is cosine
similarity against C stored prototype vectors in both cases.

Warm-up rule
------------
P^2 needs five observations to seed its five markers. For a class with
fewer than 5 confirmed examples, this implementation buffers the raw
embeddings (at most 4 vectors, i.e. at most ``4 * D`` float32 values -- still
bounded, and smaller than the steady-state footprint) and reports the exact
median of the buffered examples so far via ``torch.median`` (identical
function to the exact learner's aggregation rule), so the prototype is
numerically continuous across the 5th update. On the 5th confirmed example
the buffer is sorted, consumed to seed the five markers, and discarded; from
then on the state is the fixed-size marker arrays described above.

Hyperparameters
----------------
There are exactly two design constants, and both are fixed by the
definition of the estimator, not chosen by looking at any accuracy result
and never tuned on validation data:

- ``p = 0.5``: the target quantile. Fixed because we want the *median*.
- 5 markers: the minimum number the classical P^2 recurrence requires
  (min, max, and 3 interior markers so the middle one can be interpolated
  from two flanking neighbours on each side). This is not a tunable
  knob -- fewer markers is not a valid P^2 configuration.

No step size, decay schedule, learning rate, bin count, or assumed data
range appears anywhere in this module. In particular, unlike a fixed-bin
histogram estimator, P^2 never assumes a bounded value range: markers
outside the current [min, max] simply become the new min/max.

Known failure modes
--------------------
- **Cold start**: for the first 4 confirmed examples of a class, the
  reported prototype is the exact median of a very small buffer, which is a
  poor estimate of the population median and can change sharply from one
  update to the next; this is unavoidable for any estimator without direct
  storage and is identical in this respect to the exact learner restricted
  to the same few examples.
- **Small-sample bias after warm-up, and a persistent bias under skew**: for
  roughly symmetric per-dimension distributions (e.g. Gaussian), the P^2
  estimate converges quickly and closely to the exact median (empirically,
  max per-dimension absolute error ~0.05 for N(1, 2) after ~3000 samples;
  see the convergence test in
  ``ssl_hdc_pilot/tests/test_streaming_median_prototypes.py``). For skewed
  distributions the classical P^2 recurrence has a known, *non-vanishing*
  small bias: empirically, for Exponential(1) data (true median ln 2 ~=
  0.693) the mean absolute error plateaus around 0.09-0.12 from N=500 out to
  N=8000 rather than continuing to shrink -- P^2 has no formal proof of
  asymptotic unbiasedness for arbitrary distributions, only that it tracks
  the target rank locally. This is still far smaller than the unbounded
  error a running mean would show under the same skew plus any outliers
  (see the bounded-influence test), but it means streaming-vs-exact
  agreement on real, possibly skewed, embedding dimensions should be
  measured empirically per class rather than assumed -- this is exactly
  what ``streaming_median_fidelity.py`` does.
- **Non-stationary class distributions**: P^2, like the exact running
  median, weighs all historical examples equally in rank terms and has no
  forgetting factor. If a class's embedding distribution drifts partway
  through the stream (e.g. a domain shift), both the exact and the
  streaming median track the *cumulative* distribution, not the most recent
  one; this control does not fix that, it only changes memory/compute.
- **No explicit range assumption, but marker heights track observed
  extremes**: an extreme-magnitude example directly overwrites the running
  min or max marker height. This does *not* move the reported median
  (the middle marker), which only ever moves by an interpolated step
  bounded by its current neighbours' heights -- see the bounded-influence
  test -- but it does mean the min/max markers themselves are not robust
  (they are not used as the prototype).
- **Ties / degenerate dimensions**: if a dimension is constant (all
  observations identical), marker positions can develop equal neighbouring
  heights; division in the parabolic/linear step is guarded with a small
  denominator clamp so this cannot raise a ZeroDivisionError, but the
  marker heights simply stay at the constant value, which is correct.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ssl_hdc_pilot.scripts.continual_hdc import Prediction, UpdateResult


_TARGET_QUANTILE = 0.5  # median; fixed by definition, not a tunable knob.
_NUM_MARKERS = 5  # minimum P^2 requires; fixed by algorithm design.
_WARMUP_COUNT = 5
_EPS = 1e-12  # denominator guard for degenerate/tied dimensions only.


class _PerClassP2Median:
    """Fixed-size P^2 streaming median state for one class, one dimension per column."""

    def __init__(self, *, dim: int, device: torch.device) -> None:
        self.dim = dim
        self.device = device
        self.count = 0
        self._buffer: list[torch.Tensor] | None = []
        self.heights: torch.Tensor | None = None  # (5, dim) float64
        self.positions: torch.Tensor | None = None  # (5, dim) int64
        p = _TARGET_QUANTILE
        self.desired = torch.tensor(
            [1.0, 1.0 + 2 * p, 1.0 + 4 * p, 3.0 + 2 * p, float(_NUM_MARKERS)],
            dtype=torch.float64, device=device,
        )
        self.increments = torch.tensor(
            [0.0, p / 2, p, (1.0 + p) / 2, 1.0], dtype=torch.float64, device=device,
        )

    def update(self, x: torch.Tensor) -> None:
        x64 = x.to(torch.float64)
        self.count += 1
        if self.heights is None:
            assert self._buffer is not None
            self._buffer.append(x64)
            if len(self._buffer) < _WARMUP_COUNT:
                return
            stacked = torch.stack(self._buffer, dim=0)  # (5, dim)
            self.heights = torch.sort(stacked, dim=0).values.clone()
            self.positions = (
                torch.arange(1, _NUM_MARKERS + 1, dtype=torch.int64, device=self.device)
                .unsqueeze(1)
                .repeat(1, self.dim)
            )
            self._buffer = None
            return

        heights, positions = self.heights, self.positions
        assert heights is not None and positions is not None

        below = x64 < heights[0]
        above = x64 >= heights[4]
        heights[0] = torch.where(below, x64, heights[0])
        heights[4] = torch.where(above, x64, heights[4])

        ge = torch.stack([x64 >= heights[i] for i in range(_NUM_MARKERS)], dim=0)  # (5, dim)
        cell = torch.clamp(ge.sum(dim=0) - 1, 0, _NUM_MARKERS - 2)  # (dim,) in {0,1,2,3}

        marker_idx = torch.arange(_NUM_MARKERS, device=self.device).unsqueeze(1)  # (5,1)
        incr_mask = marker_idx > cell.unsqueeze(0)  # (5, dim)
        self.positions = positions + incr_mask.to(torch.int64)
        self.desired = self.desired + self.increments

        for i in (1, 2, 3):
            self._maybe_adjust_marker(i)

    def _maybe_adjust_marker(self, i: int) -> None:
        heights, positions, desired = self.heights, self.positions, self.desired
        assert heights is not None and positions is not None

        n_im1 = positions[i - 1].to(torch.float64)
        n_i = positions[i].to(torch.float64)
        n_ip1 = positions[i + 1].to(torch.float64)
        d = desired[i] - n_i

        move_up = (d >= 1) & ((n_ip1 - n_i) > 1)
        move_down = (d <= -1) & ((n_im1 - n_i) < -1)
        if bool(move_up.any()):
            self._shift_marker(i, +1, move_up)
        if bool(move_down.any()):
            self._shift_marker(i, -1, move_down)

    def _shift_marker(self, i: int, direction: int, mask: torch.Tensor) -> None:
        heights, positions = self.heights, self.positions
        assert heights is not None and positions is not None
        d = float(direction)
        q_im1, q_i, q_ip1 = heights[i - 1], heights[i], heights[i + 1]
        n_im1 = positions[i - 1].to(torch.float64)
        n_i = positions[i].to(torch.float64)
        n_ip1 = positions[i + 1].to(torch.float64)

        denom_outer = (n_ip1 - n_im1).clamp(min=_EPS) if direction > 0 else (n_ip1 - n_im1).clamp(max=-_EPS)
        denom_up = (n_ip1 - n_i).clamp(min=_EPS)
        denom_down = (n_i - n_im1).clamp(min=_EPS)
        term1 = (n_i - n_im1 + d) * (q_ip1 - q_i) / denom_up
        term2 = (n_ip1 - n_i - d) * (q_i - q_im1) / denom_down
        parabolic = q_i + (d / denom_outer) * (term1 + term2)
        use_parabolic = (parabolic > q_im1) & (parabolic < q_ip1)

        neighbor_h = heights[i + direction]
        neighbor_n = positions[i + direction].to(torch.float64)
        linear_denom = (neighbor_n - n_i)
        linear_denom = linear_denom.clamp(min=_EPS) if direction > 0 else linear_denom.clamp(max=-_EPS)
        linear = q_i + d * (neighbor_h - q_i) / linear_denom

        new_q = torch.where(use_parabolic, parabolic, linear)
        heights[i] = torch.where(mask, new_q, q_i)
        positions[i] = torch.where(mask, positions[i] + direction, positions[i])

    def median(self) -> torch.Tensor | None:
        if self.heights is not None:
            return self.heights[2].to(torch.float32)
        if self._buffer:
            return torch.median(torch.stack(self._buffer, dim=0), dim=0).values.to(torch.float32)
        return None

    def nbytes(self) -> int:
        if self.heights is not None:
            assert self.positions is not None
            tensors = (self.heights, self.positions, self.desired, self.increments)
        elif self._buffer:
            tensors = tuple(self._buffer)
        else:
            return 0
        return sum(t.element_size() * t.numel() for t in tensors)


class StreamingMedianNCMPrototypeLearner:
    """Bounded-memory streaming-median control for ``MedianNCMPrototypeLearner``.

    Same public interface as ``MedianNCMPrototypeLearner``
    (``register_class``, ``update``, ``predict``, ``known_classes``,
    ``get_prototype``): native embedding dimension, no L2 normalisation, no
    projection, cosine-similarity classification against per-class
    prototypes. The only difference is how the per-class prototype is
    aggregated -- see the module docstring for the algorithm.
    """

    def __init__(self, *, input_dim: int, device: torch.device | str = "cpu") -> None:
        self.input_dim = int(input_dim)
        self.device = torch.device(device)
        self._states: dict[str, _PerClassP2Median] = {}

    @property
    def known_classes(self) -> tuple[str, ...]:
        return tuple(self._states)

    def _vector(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.ndim != 1 or x.numel() != self.input_dim:
            raise ValueError("invalid embedding dimension")
        return x

    def register_class(self, label: str) -> bool:
        if label in self._states:
            return False
        self._states[label] = _PerClassP2Median(dim=self.input_dim, device=self.device)
        return True

    def update(self, embedding: torch.Tensor, label: str) -> UpdateResult:
        registered = self.register_class(label)
        self._states[label].update(self._vector(embedding))
        return UpdateResult(class_registered=registered)

    def get_prototype(self, label: str) -> torch.Tensor:
        state = self._states[label]
        median = state.median()
        if median is None:
            raise RuntimeError("class has no confirmed examples")
        return median.clone()

    def get_vectors(self, label: str) -> list[torch.Tensor]:
        """Interface parity helper: returns the buffered raw vectors, if any.

        Unlike ``MedianNCMPrototypeLearner.get_vectors`` (which always has the
        full example list), this is only non-empty during warm-up (< 5
        confirmed examples); afterwards the raw examples are not retained,
        which is the entire point of this control.
        """
        state = self._states[label]
        return [v.to(torch.float32).clone() for v in state._buffer] if state._buffer else []

    def state_nbytes(self, label: str) -> int:
        """Total bytes of this class's persistent aggregation state."""
        return self._states[label].nbytes()

    def predict(self, embedding: torch.Tensor) -> Prediction:
        active = [
            (label, state.median())
            for label, state in self._states.items()
            if state.median() is not None
        ]
        if not active:
            return Prediction(None, 0.0, 0.0)
        query = self._vector(embedding)
        labels, prototypes = zip(*active)
        similarities = F.cosine_similarity(query.unsqueeze(0), torch.stack(prototypes), dim=1)
        top_index = int(similarities.argmax())
        margin = (
            float(torch.topk(similarities, 2).values.diff()[0].abs())
            if len(similarities) > 1
            else 0.0
        )
        return Prediction(
            label=labels[top_index],
            confidence=margin,
            top_probability=float(torch.softmax(similarities, 0)[top_index]),
        )
