from __future__ import annotations

import builtins
import pathlib

import pytest
import torch

from ssl_hdc_pilot.scripts.aggregation_prototypes import MedianNCMPrototypeLearner
from ssl_hdc_pilot.scripts.continual_hdc import Prediction, UpdateResult
from ssl_hdc_pilot.scripts.streaming_median_prototypes import (
    StreamingMedianNCMPrototypeLearner,
)


# ---------------------------------------------------------------------------
# Bounded memory: state size must not grow with the number of updates.
# ---------------------------------------------------------------------------


def test_state_bytes_identical_after_10_and_10000_updates() -> None:
    torch.manual_seed(0)
    dim = 12
    learner = StreamingMedianNCMPrototypeLearner(input_dim=dim)

    for _ in range(10):
        learner.update(torch.randn(dim), "a")
    bytes_after_10 = learner.state_nbytes("a")

    for _ in range(9_990):
        learner.update(torch.randn(dim), "a")
    bytes_after_10000 = learner.state_nbytes("a")

    assert bytes_after_10 == bytes_after_10000
    # Sanity: the fixed footprint really is O(D), not accidentally zero.
    assert bytes_after_10 > 0


def test_state_bytes_do_not_depend_on_update_count_across_many_checkpoints() -> None:
    torch.manual_seed(1)
    dim = 8
    learner = StreamingMedianNCMPrototypeLearner(input_dim=dim)
    checkpoints = []
    for batch in range(5):
        for _ in range(200):
            learner.update(torch.randn(dim), "a")
        checkpoints.append(learner.state_nbytes("a"))
    assert len(set(checkpoints)) == 1


def test_exact_median_learner_storage_grows_but_streaming_learner_does_not() -> None:
    """Direct contrast with the exact learner this module is a control for."""
    torch.manual_seed(2)
    dim = 10
    exact = MedianNCMPrototypeLearner(input_dim=dim)
    streaming = StreamingMedianNCMPrototypeLearner(input_dim=dim)

    def exact_bytes() -> int:
        vectors = exact.get_vectors("a")
        return sum(v.element_size() * v.numel() for v in vectors)

    for _ in range(20):
        v = torch.randn(dim)
        exact.update(v, "a")
        streaming.update(v, "a")
    exact_at_20 = exact_bytes()
    streaming_at_20 = streaming.state_nbytes("a")

    for _ in range(2000):
        v = torch.randn(dim)
        exact.update(v, "a")
        streaming.update(v, "a")
    exact_at_2020 = exact_bytes()
    streaming_at_2020 = streaming.state_nbytes("a")

    assert exact_at_2020 > exact_at_20  # unbounded growth, as documented
    assert streaming_at_2020 == streaming_at_20  # bounded, as designed


# ---------------------------------------------------------------------------
# Convergence toward the exact median on synthetic data.
# ---------------------------------------------------------------------------


def test_converges_to_exact_median_on_symmetric_synthetic_data() -> None:
    """Gaussian per-dimension data: P^2 tracks the exact median closely."""
    torch.manual_seed(3)
    dim = 16
    n = 3000
    data = torch.randn(n, dim) * 2.0 + 1.0

    learner = StreamingMedianNCMPrototypeLearner(input_dim=dim)
    for i in range(n):
        learner.update(data[i], "a")

    exact = torch.median(data, dim=0).values
    streaming = learner.get_prototype("a")

    # Stated tolerance: for well-behaved (roughly symmetric) per-dimension
    # distributions, P^2's median estimate stays within 0.1 absolute of the
    # exact median at this sample size and spread (empirically ~0.05 max
    # abs error; see module docstring "Known failure modes").
    torch.testing.assert_close(streaming, exact, atol=0.1, rtol=0.0)


def test_convergence_improves_with_more_samples_on_symmetric_data() -> None:
    torch.manual_seed(4)
    dim = 6

    def max_error(n: int) -> float:
        torch.manual_seed(4)
        data = torch.randn(n, dim)
        learner = StreamingMedianNCMPrototypeLearner(input_dim=dim)
        for i in range(n):
            learner.update(data[i], "a")
        exact = torch.median(data, dim=0).values
        return float((learner.get_prototype("a") - exact).abs().max())

    small_n_error = max_error(60)
    large_n_error = max_error(4000)
    assert large_n_error < small_n_error


def test_skewed_data_streaming_median_stays_within_a_documented_bound() -> None:
    """Exponential data has a known, non-vanishing P^2 bias (see docstring).

    This does not claim convergence to zero error; it pins down the bound
    actually observed so a future change to the estimator cannot silently
    regress well past the documented behaviour.
    """
    torch.manual_seed(5)
    dim = 8
    n = 2000
    data = torch.distributions.Exponential(1.0).sample((n, dim))

    learner = StreamingMedianNCMPrototypeLearner(input_dim=dim)
    for i in range(n):
        learner.update(data[i], "a")

    exact = torch.median(data, dim=0).values
    streaming = learner.get_prototype("a")
    max_abs_error = float((streaming - exact).abs().max())

    assert max_abs_error < 0.25


# ---------------------------------------------------------------------------
# Bounded-influence property: one extreme outlier cannot move a dimension far
# once a majority of examples agree, unlike a running mean.
# ---------------------------------------------------------------------------


def test_bounded_influence_of_a_single_extreme_outlier() -> None:
    torch.manual_seed(6)
    dim = 4
    n_agree = 51
    cluster = torch.randn(n_agree, dim) * 0.01  # tightly agreeing examples near 0

    learner = StreamingMedianNCMPrototypeLearner(input_dim=dim)
    running_mean = torch.zeros(dim)
    for i in range(n_agree):
        learner.update(cluster[i], "a")
        running_mean = running_mean + (cluster[i] - running_mean) / (i + 1)

    before_median = learner.get_prototype("a").clone()
    before_mean = running_mean.clone()

    outlier = torch.full((dim,), 1.0e6)
    learner.update(outlier, "a")
    running_mean = running_mean + (outlier - running_mean) / (n_agree + 1)

    after_median = learner.get_prototype("a")
    median_shift = float((after_median - before_median).abs().max())
    mean_shift = float((running_mean - before_mean).abs().max())

    # The running mean is dragged toward the outlier by construction
    # (shift ~= outlier / N). The streaming median's *reported* value is the
    # middle marker, which only ever moves by an interpolated step bounded
    # by its current neighbouring marker heights -- never toward the raw
    # outlier magnitude -- so its shift is orders of magnitude smaller.
    assert mean_shift > 1000.0
    assert median_shift < 1.0
    assert median_shift < mean_shift / 1000.0


def test_bounded_influence_holds_for_several_outliers_in_a_row() -> None:
    torch.manual_seed(7)
    dim = 3
    cluster = torch.randn(101, dim) * 0.05

    learner = StreamingMedianNCMPrototypeLearner(input_dim=dim)
    for i in range(101):
        learner.update(cluster[i], "a")
    before = learner.get_prototype("a").clone()

    for magnitude in (1.0e3, -1.0e4, 1.0e6, -1.0e8):
        learner.update(torch.full((dim,), magnitude), "a")

    after = learner.get_prototype("a")
    assert float((after - before).abs().max()) < 5.0


# ---------------------------------------------------------------------------
# Interface parity with MedianNCMPrototypeLearner.
# ---------------------------------------------------------------------------


def test_register_class_and_update_semantics_match_median_ncm() -> None:
    torch.manual_seed(8)
    dim = 5
    exact = MedianNCMPrototypeLearner(input_dim=dim)
    streaming = StreamingMedianNCMPrototypeLearner(input_dim=dim)

    v1, v2, v3 = torch.randn(dim), torch.randn(dim), torch.randn(dim)

    exact_first = exact.update(v1, "known_a")
    streaming_first = streaming.update(v1, "known_a")
    exact_repeat = exact.update(v2, "known_a")
    streaming_repeat = streaming.update(v2, "known_a")
    exact_novel = exact.update(v3, "novel_b")
    streaming_novel = streaming.update(v3, "novel_b")

    for result in (exact_first, streaming_first, exact_repeat, streaming_repeat, exact_novel, streaming_novel):
        assert isinstance(result, UpdateResult)

    assert exact_first.class_registered == streaming_first.class_registered is True
    assert exact_repeat.class_registered == streaming_repeat.class_registered is False
    assert exact_novel.class_registered == streaming_novel.class_registered is True
    assert exact.known_classes == streaming.known_classes == ("known_a", "novel_b")


def test_get_prototype_raises_for_unknown_and_unconfirmed_classes_like_median_ncm() -> None:
    dim = 4
    exact = MedianNCMPrototypeLearner(input_dim=dim)
    streaming = StreamingMedianNCMPrototypeLearner(input_dim=dim)

    with pytest.raises(KeyError):
        exact.get_prototype("nope")
    with pytest.raises(KeyError):
        streaming.get_prototype("nope")

    exact.register_class("registered_only")
    streaming.register_class("registered_only")
    with pytest.raises(RuntimeError):
        exact.get_prototype("registered_only")
    with pytest.raises(RuntimeError):
        streaming.get_prototype("registered_only")


def test_predict_returns_the_same_prediction_shape_as_median_ncm() -> None:
    torch.manual_seed(9)
    dim = 6
    exact = MedianNCMPrototypeLearner(input_dim=dim)
    streaming = StreamingMedianNCMPrototypeLearner(input_dim=dim)

    class_a = torch.randn(30, dim) + torch.tensor([5.0] * dim)
    class_b = torch.randn(30, dim) - torch.tensor([5.0] * dim)
    for i in range(30):
        exact.update(class_a[i], "a")
        streaming.update(class_a[i], "a")
        exact.update(class_b[i], "b")
        streaming.update(class_b[i], "b")

    query = torch.tensor([5.0] * dim)
    exact_prediction = exact.predict(query)
    streaming_prediction = streaming.predict(query)

    assert isinstance(exact_prediction, Prediction)
    assert isinstance(streaming_prediction, Prediction)
    assert exact_prediction.label == streaming_prediction.label == "a"
    assert 0.0 <= exact_prediction.top_probability <= 1.0
    assert 0.0 <= streaming_prediction.top_probability <= 1.0


def test_predict_with_no_confirmed_classes_returns_none_label_like_median_ncm() -> None:
    dim = 3
    exact = MedianNCMPrototypeLearner(input_dim=dim)
    streaming = StreamingMedianNCMPrototypeLearner(input_dim=dim)
    query = torch.randn(dim)

    exact_prediction = exact.predict(query)
    streaming_prediction = streaming.predict(query)

    assert exact_prediction.label is None
    assert streaming_prediction.label is None
    assert exact_prediction.confidence == streaming_prediction.confidence == 0.0
    assert exact_prediction.top_probability == streaming_prediction.top_probability == 0.0


def test_invalid_embedding_dimension_is_rejected_like_median_ncm() -> None:
    dim = 5
    exact = MedianNCMPrototypeLearner(input_dim=dim)
    streaming = StreamingMedianNCMPrototypeLearner(input_dim=dim)
    wrong_shape = torch.randn(dim + 1)

    with pytest.raises(ValueError):
        exact.update(wrong_shape, "a")
    with pytest.raises(ValueError):
        streaming.update(wrong_shape, "a")


# ---------------------------------------------------------------------------
# Guard: this module and its tests must never open the trusted test split.
# ---------------------------------------------------------------------------


def test_guard_fails_if_trusted_test_split_is_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fidelity/interface work here is pure in-memory tensor computation; it
    must never touch ssl_hdc_pilot/trusted_splits/test.csv. This patches
    both ``open`` and ``Path.open`` to raise if that exact file is targeted,
    then exercises the full public surface of the streaming learner to prove
    it never happens.
    """
    forbidden = ("trusted_splits", "test.csv")

    def _is_forbidden(path: object) -> bool:
        parts = pathlib.Path(str(path)).parts
        return any(parts[i:i + 2] == forbidden for i in range(len(parts) - 1))

    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):  # noqa: ANN001 - mirrors builtins.open signature
        if _is_forbidden(file):
            raise AssertionError(f"forbidden open of trusted test split: {file}")
        return real_open(file, *args, **kwargs)

    real_path_open = pathlib.Path.open

    def guarded_path_open(self, *args, **kwargs):  # noqa: ANN001
        if _is_forbidden(self):
            raise AssertionError(f"forbidden Path.open of trusted test split: {self}")
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(pathlib.Path, "open", guarded_path_open)

    torch.manual_seed(10)
    dim = 4
    learner = StreamingMedianNCMPrototypeLearner(input_dim=dim)
    for i in range(12):
        learner.update(torch.randn(dim), "a" if i % 2 == 0 else "b")
    learner.predict(torch.randn(dim))
    learner.get_prototype("a")
    learner.state_nbytes("a")
