"""Synthetic-only fixed-population and saved-state validation checks."""
import csv
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ssl_hdc_pilot.scripts import evaluate_validation as evaluation
from ssl_hdc_pilot.scripts.continual_prototypes import NCMPrototypeLearner
from ssl_hdc_pilot.scripts.experiment_a import run_experiment_a_cell
from ssl_hdc_pilot.scripts.learner_checkpoint import load_learner_checkpoint
from ssl_hdc_pilot.scripts.stream_runner import LEARNER_NAMES
from ssl_hdc_pilot.tests.test_stream_runner import _StepClock, _write_smoke_study


CLASSES = ["1_00_0", "2_b1_1", "3_b2_2"]


@pytest.fixture
def study(tmp_path):
    embeddings, train, spec = _write_smoke_study(tmp_path, 4)
    replacements = dict(zip(["known_a", "known_b", "novel_c"], CLASSES))
    for path in (train, spec):
        text = path.read_text()
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text)
    with np.load(embeddings) as archive:
        ids = archive["image_ids"].copy()
        vectors = archive["embeddings"].copy()
    val_ids = ["val_0", "val_1", "val_2"]
    np.savez_compressed(embeddings, image_ids=np.concatenate([ids, val_ids]),
                        embeddings=np.concatenate([vectors, np.eye(3, 4, dtype=np.float32)]))
    validation = tmp_path / "val.csv"
    with validation.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label"])
        writer.writerows(zip(val_ids, CLASSES))
    return embeddings, train, spec, validation


def make_source(study, tmp_path, learner="ncm", budget=0):
    embeddings, train, spec, validation = study
    source = tmp_path / "source"
    run_experiment_a_cell(embeddings_path=embeddings, split_csv=train, stream_spec_path=spec,
                         learner_name=learner, stream_seed=17, budget=budget,
                         output_dir=source, smoke_mode=True, clock=_StepClock())
    return dict(source_dir=source, embeddings_path=embeddings,
                validation_csv=validation, smoke_mode=True)


@pytest.mark.parametrize("learner", LEARNER_NAMES)
@pytest.mark.parametrize("budget", [0, 7])
def test_saved_state_validation_is_fixed_pure_and_nonoverwriting(study, tmp_path, learner, budget):
    kwargs = make_source(study, tmp_path, learner, budget)
    source_before = {p.name: p.read_bytes() for p in kwargs["source_dir"].iterdir()}
    plan = evaluation.plan_validation(**kwargs)
    assert plan["samples"] == 3 and plan["classes"] == CLASSES
    output = tmp_path / "evaluation"
    report = evaluation.run_validation(**kwargs, output_dir=output)
    assert report["status"] == "PASS" and report["state_and_rng_unchanged"]
    assert report["data_role"] == "synthetic_smoke"
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["classes"] == CLASSES
    assert len(metrics["metrics"]["confusion_matrix"]) == 3
    assert sum(map(sum, metrics["metrics"]["confusion_matrix"])) == 3
    assert metrics["learner_rng_fingerprint_before"] == metrics["learner_rng_fingerprint_after"]
    if budget == 0:
        assert metrics["unregistered_classes"] == [CLASSES[2]]
        assert metrics["metrics"]["per_class_recall"][CLASSES[2]] == 0
    assert {p.name: p.read_bytes() for p in kwargs["source_dir"].iterdir()} == source_before
    with pytest.raises(FileExistsError):
        evaluation.run_validation(**kwargs, output_dir=output)


def test_hand_calculated_metrics_keep_unregistered_class_and_stage_semantics():
    learner = NCMPrototypeLearner(input_dim=2)
    learner.update(torch.tensor([1., 0.]), "1_00_0")
    learner.update(torch.tensor([0., 1.]), "2_b1_1")
    classes = ["1_00_0", "2_b1_1", "3_b2_1"]
    rows = [("a", classes[0]), ("b", classes[1]), ("c", classes[2])]
    result = evaluation.evaluate_fixed_population(learner, rows, {
        "a": torch.tensor([1., 0.]), "b": torch.tensor([0., 1.]), "c": torch.tensor([0., 1.])}, classes=classes)
    metrics = result["metrics"]
    assert metrics["macro_f1"] == pytest.approx(5 / 9)
    assert metrics["macro_recall"] == pytest.approx(2 / 3)
    assert metrics["per_stage_recall"] == {"0": 1., "1": 1., "2": None, "3": None}
    assert metrics["ece"] == pytest.approx(float(torch.sigmoid(torch.tensor(1.))) - 2 / 3)
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 0]]


def test_neural_training_mode_and_rng_preserved(study, tmp_path):
    kwargs = make_source(study, tmp_path, "replay", 7)
    learner = load_learner_checkpoint(kwargs["source_dir"] / "final_state.npz").learner
    learner.model.train()
    before = evaluation._fingerprint(learner)
    evaluation.evaluate_fixed_population(learner, [("a", CLASSES[0]), ("b", CLASSES[1]), ("c", CLASSES[2])],
        dict(zip(["a", "b", "c"], torch.eye(3, 4))), classes=CLASSES)
    assert learner.model.training and before == evaluation._fingerprint(learner)


def test_rejects_state_mutating_prediction(study, tmp_path, monkeypatch):
    kwargs = make_source(study, tmp_path)
    learner = load_learner_checkpoint(kwargs["source_dir"] / "final_state.npz").learner
    predict = learner.predict
    def mutating_predict(vector):
        learner.update(vector, CLASSES[0])
        return predict(vector)
    monkeypatch.setattr(learner, "predict", mutating_predict)
    with pytest.raises(RuntimeError, match="changed learner state or RNG"):
        evaluation.evaluate_fixed_population(learner, [("a", CLASSES[0]), ("b", CLASSES[1]), ("c", CLASSES[2])],
            dict(zip(["a", "b", "c"], torch.eye(3, 4))), classes=CLASSES)


def test_rejects_nonvalidation_path_before_reading(tmp_path):
    with pytest.raises(RuntimeError, match="validation-only"):
        evaluation.plan_validation(source_dir=tmp_path / "missing", embeddings_path=tmp_path / "missing.npz",
                                   validation_csv=tmp_path / "test.csv")


def test_real_scoring_requires_preflight(tmp_path):
    with pytest.raises(RuntimeError, match="manual user launch"):
        evaluation.run_validation(source_dir=tmp_path / "missing", embeddings_path=tmp_path / "missing.npz",
                                  validation_csv=tmp_path / "val.csv", output_dir=tmp_path / "out")


def test_plan_does_not_load_learner_or_write_files(study, tmp_path, monkeypatch):
    kwargs = make_source(study, tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("plan must not load a learner")
    monkeypatch.setattr(evaluation.checkpoints, "load_learner_checkpoint", forbidden)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    evaluation.plan_validation(**kwargs)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("problem", ["corrupt", "overlap", "missing_embedding", "missing_class"])
def test_rejects_invalid_inputs(study, tmp_path, problem):
    kwargs = make_source(study, tmp_path)
    if problem == "corrupt":
        (kwargs["source_dir"] / "final_state.npz").write_bytes(b"corrupt")
    else:
        text = kwargs["validation_csv"].read_text()
        if problem == "overlap":
            text = text.replace("val_0", "image_0")
        elif problem == "missing_embedding":
            text = text.replace("val_0", "missing")
        else:
            text = text.replace(CLASSES[2], CLASSES[0])
        kwargs["validation_csv"].write_text(text)
    with pytest.raises(ValueError):
        evaluation.plan_validation(**kwargs)


def test_rejects_validation_changed_after_parsing(study, tmp_path, monkeypatch):
    kwargs = make_source(study, tmp_path)
    original_open = Path.open
    @contextmanager
    def changing_open(path, *args, **options):
        with original_open(path, *args, **options) as handle:
            yield handle
        if path == kwargs["validation_csv"] and options.get("newline") == "":
            with original_open(path, "a") as handle:
                handle.write("\n")
    monkeypatch.setattr(Path, "open", changing_open)
    with pytest.raises(ValueError, match="input changed: validation"):
        evaluation.plan_validation(**kwargs)
