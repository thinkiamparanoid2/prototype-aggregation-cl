"""Synthetic acceptance for the mechanism-ablation aggregation controls.

CODEX_MECHANISM_ABLATION_PROMPT.md deliverable #3, filled per
CODEX_ABLATION_BATCH_RUNNER_AMENDMENT.md Sec.4. Sign-NCM/Median-NCM are
projection-free native-dimension controls used to test whether HDC's collapse
resistance survives without any hyperdimensional projection. Test 5 is the
regression test for the Sec.1 evaluate_snapshots.py ablation_mode passthrough
defect; a synthetic smoke-scale fixture stands in for a real arm.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ssl_hdc_pilot.scripts import budget_experiment as capture
from ssl_hdc_pilot.scripts import evaluate_snapshots as evaluation
from ssl_hdc_pilot.scripts.aggregation_prototypes import (
    MedianNCMPrototypeLearner,
    SignNCMPrototypeLearner,
)
from ssl_hdc_pilot.scripts.continual_prototypes import NCMPrototypeLearner
from ssl_hdc_pilot.scripts.learner_checkpoint import (
    load_learner_checkpoint,
    save_learner_checkpoint,
)
from ssl_hdc_pilot.tests.test_evaluate_validation import study
from ssl_hdc_pilot.tests.test_stream_runner import _StepClock


# ---------------------------------------------------------------------------
# 1. Bounded influence of sign accumulation.
# ---------------------------------------------------------------------------

def test_sign_accumulation_bounds_a_single_outliers_influence_but_ncm_moves():
    """One extreme-magnitude outlier cannot flip a majority-agreed Sign-NCM bit,
    but the same outlier does move (and here, flip) an NCM running-mean prototype.

    Five identical "majority" samples give a weak-but-consistent positive signal
    in dimension 0 (raw value 0.1) alongside a larger, stable signal in the other
    dimensions. A sixth "outlier" sample has an extreme negative value in
    dimension 0 (-1000) and near-zero elsewhere.

    Sign-NCM only ever accumulates +-1 per sample regardless of magnitude, so the
    net vote after 5 agreeing votes and 1 opposing vote (5 - 1 = 4) stays
    positive: the prototype bit is unchanged. NCM's per-sample L2 normalization
    lets the outlier's extreme magnitude dominate its own normalized direction
    (~-1 in dimension 0), and its running-mean update pulls the tiny existing
    mean past zero: bounded per dimension in isolation, but still able to flip
    the running mean the discretized Sign-NCM accumulator will not.
    """
    majority = torch.tensor([0.1, 1.0, 1.0, 1.0])
    outlier = torch.tensor([-1000.0, 0.001, 0.001, 0.001])

    sign_learner = SignNCMPrototypeLearner(input_dim=4)
    ncm_learner = NCMPrototypeLearner(input_dim=4)
    for _ in range(5):
        sign_learner.update(majority, "a")
        ncm_learner.update(majority, "a")

    sign_before = sign_learner.get_prototype("a")[0].item()
    ncm_before = ncm_learner.get_prototype("a")[0].item()
    assert sign_before > 0
    assert ncm_before > 0

    sign_learner.update(outlier, "a")
    ncm_learner.update(outlier, "a")

    sign_after = sign_learner.get_prototype("a")[0].item()
    ncm_after = ncm_learner.get_prototype("a")[0].item()

    # Sign-NCM: majority vote (5 vs 1) is unchanged; the bit cannot flip.
    assert sign_after == sign_before == 1.0
    # NCM: the same outlier measurably moves, and here flips, the running mean.
    assert ncm_after != pytest.approx(ncm_before)
    assert ncm_after < 0 < ncm_before


# ---------------------------------------------------------------------------
# 2. Projection-free checkpoint round-trip, both learners.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [SignNCMPrototypeLearner, MedianNCMPrototypeLearner])
def test_projection_free_checkpoint_round_trip(cls, tmp_path):
    learner = cls(input_dim=5)
    samples = {
        "known_a": [torch.tensor([1.0, 0.0, -1.0, 2.0, 0.5]),
                    torch.tensor([0.9, -0.1, -0.8, 1.8, 0.4])],
        "known_b": [torch.tensor([-1.0, 1.0, 0.0, -2.0, -0.5])],
    }
    for label, vectors in samples.items():
        for vector in vectors:
            learner.update(vector, label)

    path = tmp_path / "checkpoint.npz"
    save_learner_checkpoint(path, learner, context={"scope": "unit_test"})
    # No shared_projection_root is required: these learners never emit a
    # "projection" array, so there is no projection blob to resolve.
    restored = load_learner_checkpoint(path)

    assert set(restored.learner.known_classes) == set(learner.known_classes)
    assert restored.context == {"scope": "unit_test"}
    probe = torch.tensor([1.0, -0.2, -0.9, 1.9, 0.45])
    original_prediction = learner.predict(probe)
    restored_prediction = restored.learner.predict(probe)
    assert restored_prediction.label == original_prediction.label
    assert restored_prediction.top_probability == pytest.approx(original_prediction.top_probability)
    for label in learner.known_classes:
        assert torch.equal(restored.learner.get_prototype(label), learner.get_prototype(label))


# ---------------------------------------------------------------------------
# 3. Arm loading and dimension handling at 384, 768 and 1024.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [SignNCMPrototypeLearner, MedianNCMPrototypeLearner])
@pytest.mark.parametrize("dim", [384, 768, 1024])
def test_arm_dimension_handling_at_registered_widths(cls, dim):
    """The three registered ablation-arm embedding widths (ViT-S/B/L) must all
    be usable end to end, and cross-dimension embeddings must be rejected
    rather than silently truncated or padded."""
    learner = cls(input_dim=dim)
    vector = torch.linspace(-1.0, 1.0, steps=dim)
    result = learner.update(vector, "class_a")
    assert result.class_registered is True
    prediction = learner.predict(vector)
    assert prediction.label == "class_a"
    assert learner.get_prototype("class_a").numel() == dim

    with pytest.raises(ValueError):
        learner.update(torch.zeros(dim - 1), "class_b")


# ---------------------------------------------------------------------------
# 4. Guard test: the trusted test partition must never be opened.
# ---------------------------------------------------------------------------

def test_guard_forbids_opening_trusted_test_partition(monkeypatch):
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        if self.name == "test.csv":
            raise AssertionError("trusted test must never be opened")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    learner = SignNCMPrototypeLearner(input_dim=4)
    learner.update(torch.tensor([1.0, 0.0, 0.0, 0.0]), "a")
    assert learner.predict(torch.tensor([1.0, 0.0, 0.0, 0.0])).label == "a"

    median_learner = MedianNCMPrototypeLearner(input_dim=4)
    median_learner.update(torch.tensor([1.0, 0.0, 0.0, 0.0]), "a")
    assert median_learner.predict(torch.tensor([1.0, 0.0, 0.0, 0.0])).label == "a"


# ---------------------------------------------------------------------------
# 5. New: ablation capture round-tripped through evaluate_snapshots to PASS.
#    Regression test for Sec.1: before the ablation_mode passthrough fix, the
#    evaluator's re-derivation call hit `unknown learner` for sign_ncm/median_ncm.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("learner_name", ["sign_ncm", "median_ncm"])
def test_ablation_capture_round_trips_through_evaluate_snapshots_to_pass(study, tmp_path, learner_name):
    embeddings, train, spec, val = study
    source = tmp_path / "source"
    report = capture.run_budget_cell(experiment="B", threshold=2., embeddings_path=embeddings,
        split_csv=train, stream_spec_path=spec, learner_name=learner_name, stream_seed=17,
        budget=3, output_dir=source, smoke_mode=True, clock=_StepClock(), ablation_mode=True)
    assert report["status"] == "PASS"
    assert report["ablation_mode"] is True

    kwargs = dict(source_dir=source, embeddings_path=embeddings, validation_csv=val, smoke_mode=True)
    evaluation_output = tmp_path / "evaluation"
    result = evaluation.run_snapshot_validation(**kwargs, output_dir=evaluation_output)
    assert result["status"] == "PASS"
    receipt = json.loads((evaluation_output / "integrity_report.json").read_text())
    assert receipt["status"] == "PASS"
