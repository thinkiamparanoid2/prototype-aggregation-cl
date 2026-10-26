"""Synthetic public-interface checks for approved shared-label development."""
import csv
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import pytest

from ssl_hdc_pilot.scripts.experiment_a import (
    ADDITIONAL_BUDGETS, build_shared_label_schedule, plan_experiment_a, run_experiment_a_cell,
)
from ssl_hdc_pilot.scripts.learner_checkpoint import load_learner_checkpoint
from ssl_hdc_pilot.scripts.stream_runner import LEARNER_NAMES
from ssl_hdc_pilot.tests.test_stream_runner import _StepClock, _write_smoke_study


def test_schedule_has_fixed_reference_selection_and_chronological_delivery() -> None:
    # Independent .NET SHA-256 reference: a=0fea..., d=920c..., b=d41b..., c=fb04...
    schedule = build_shared_label_schedule(["d", "c", "b", "a"], stream_seed=0, budget=2)
    assert schedule["selected_events"] == [{"step": 0, "image_id": "d"}, {"step": 3, "image_id": "a"}]
    assert "labels" not in schedule
    assert build_shared_label_schedule(["d", "c", "b", "a"], stream_seed=0, budget=2) == schedule


def test_all_approved_budgets_are_nested_and_exact() -> None:
    previous = set()
    ids = [str(i) for i in range(976)]
    for budget in ADDITIONAL_BUDGETS:
        schedule = build_shared_label_schedule(ids, stream_seed=1, budget=budget)
        selected = {e["image_id"] for e in schedule["selected_events"]}
        assert len(selected) == budget
        assert previous <= selected
        assert [e["step"] for e in schedule["selected_events"]] == sorted(e["step"] for e in schedule["selected_events"])
        previous = selected


@pytest.mark.parametrize("ids,seed,budget", [
    (["a", "a"], 0, 1), ([" a"], 0, 1), ([1], 0, 1), ([], 0, 0),
    (["a"], True, 0), (["a"], -1, 0), (["a"], 0, True), (["a"], 0, -1),
    (["a"], 0, 2), (["a"], 0, 0.5),
])
def test_schedule_rejects_ambiguous_or_invalid_inputs(ids, seed, budget) -> None:
    with pytest.raises(ValueError):
        build_shared_label_schedule(ids, stream_seed=seed, budget=budget)


@pytest.fixture
def inputs(tmp_path: Path) -> dict:
    embeddings, split, spec = _write_smoke_study(tmp_path, 4)
    return dict(embeddings_path=embeddings, split_csv=split, stream_spec_path=spec,
                stream_seed=17, smoke_mode=True)


@pytest.mark.parametrize("learner", LEARNER_NAMES)
@pytest.mark.parametrize("budget", [0, 1, 3, 7])
def test_every_learner_consumes_only_the_same_scheduled_labels(inputs: dict, tmp_path: Path,
                                                             learner: str, budget: int) -> None:
    before = {p: p.read_bytes() for p in inputs.values() if isinstance(p, Path)}
    output = tmp_path / "output"
    report = run_experiment_a_cell(**inputs, learner_name=learner, budget=budget,
                                   output_dir=output, clock=_StepClock())
    assert report["status"] == "PASS"
    assert report["labels_consumed"] == budget
    assert report["stream_steps"] == 7
    assert report["inputs_changed_during_run"] == report["code_changed_during_run"] == []
    with (output / "history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    selected = [{"step": int(r["step"]), "image_id": r["image_id"]}
                for r in history if r["label_requested"] == "True"]
    schedule = build_shared_label_schedule([r["image_id"] for r in history], stream_seed=17, budget=budget)
    assert selected == schedule["selected_events"]
    assert all(r["label_request_denied_budget"] == "False" for r in history)
    seen = {"known_a", "known_b"}
    for row in history:
        expected_registration = row["label_requested"] == "True" and row["true_label"] not in seen
        assert (row["class_registered"] == "True") == expected_registration
        if row["label_requested"] == "True":
            seen.add(row["true_label"])
    checkpoint = load_learner_checkpoint(output / "final_state.npz")
    assert set(checkpoint.learner.known_classes) == seen
    assert checkpoint.context["progress"]["labels_charged"] == budget
    summary = json.loads((output / "summary.json").read_text())
    assert summary["supervision"]["bootstrap_labels_supplied"] == 2
    assert summary["supervision"]["continual_learner_labels_supplied"] == 2 + budget
    assert summary["label_budget"]["bootstrap_labels_charged"] == 0
    assert summary["policy"]["acquisition_threshold"] is None
    assert summary["final_evaluation_samples"] == 9
    assert {p: p.read_bytes() for p in before} == before


def test_plan_is_read_only_and_schedules_do_not_depend_on_embedding_arm(inputs: dict, tmp_path: Path) -> None:
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    plan1 = plan_experiment_a(**inputs, learner_name="hdc", budget=3)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    with np.load(inputs["embeddings_path"]) as payload:
        image_ids, vectors = payload["image_ids"], payload["embeddings"]
    other_arm = tmp_path / "other_arm.npz"
    np.savez_compressed(other_arm, image_ids=image_ids, embeddings=(-vectors).astype(np.float32))
    other = {**inputs, "embeddings_path": other_arm}
    plan2 = plan_experiment_a(**other, learner_name="replay", budget=3)
    assert plan1["schedule_sha256"] == plan2["schedule_sha256"]


def test_refuses_existing_output_and_keeps_its_files(inputs: dict, tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("user content")
    with pytest.raises(FileExistsError, match="overwrite/reuse"):
        run_experiment_a_cell(**inputs, learner_name="ncm", budget=0, output_dir=output)
    assert (output / "keep.txt").read_text() == "user content"


def test_nontraining_path_rejected_before_any_input_read(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="canonical approved train"):
        plan_experiment_a(embeddings_path=tmp_path / "missing.npz", split_csv=tmp_path / "test.csv",
                          stream_spec_path=tmp_path / "missing.json", learner_name="ncm",
                          stream_seed=0, budget=25, smoke_mode=False)


def test_official_execution_requires_manual_preflight(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="manual user launch"):
        run_experiment_a_cell(embeddings_path=tmp_path / "missing.npz", split_csv=tmp_path / "missing.csv",
                              stream_spec_path=tmp_path / "missing.json", learner_name="ncm",
                              stream_seed=0, budget=25, output_dir=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_plan_refuses_input_replaced_after_parsing(inputs: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    original_open = Path.open

    @contextmanager
    def changing_open(path: Path, *args, **kwargs):
        with original_open(path, *args, **kwargs) as handle:
            yield handle
        if path == inputs["split_csv"] and kwargs.get("newline") == "":
            with original_open(path, "a", encoding="utf-8") as handle:
                handle.write("\n")

    monkeypatch.setattr(Path, "open", changing_open)
    with pytest.raises(ValueError, match="inputs changed during preparation"):
        plan_experiment_a(**inputs, learner_name="ncm", budget=3)
