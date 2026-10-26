"""Acceptance tests for the user-owned comparative forgetting-grid preparation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ssl_hdc_pilot.scripts import uptake_repair


def history(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "true_label", "prediction"))
        writer.writeheader()
        for step, (truth, prediction) in enumerate(rows):
            writer.writerow({"step": step, "true_label": truth, "prediction": prediction})


def test_all_four_uptake_criteria_and_null_handling():
    label = "3_a9_2"
    # Exactly three 10-wide qualifying windows, but fewer than 22 observations:
    # the original criterion must remain an explicit null/not-reached result.
    rows = [{"true_label": label, "prediction": label} for _ in range(12)]
    result = uptake_repair.class_uptake(rows, label)
    assert result["N_available"] == 12
    assert result["original_three_x_20_at_080"] == {
        "examples_to_threshold": None, "status": "not_reached", "reachable_by_support": False}
    assert result["repaired_three_x_10_at_080"]["examples_to_threshold"] == 12
    assert result["terminal_accuracy"] == {"window_examples": 12, "accuracy": 1.0}
    assert result["examples_to_threshold"] == 12


def test_original_criterion_requires_22_examples_not_three_windows_of_20():
    label = "5_a7_2"
    result = uptake_repair.class_uptake(
        [{"true_label": label, "prediction": label} for _ in range(22)], label)
    assert result["original_three_x_20_at_080"]["examples_to_threshold"] == 22
    assert result["original_three_x_20_at_080"]["reachable_by_support"] is True


def test_repair_pins_saved_predictions_and_refuses_stream_support_mismatch(tmp_path):
    source, evaluation = tmp_path / "source", tmp_path / "evaluation"
    source.mkdir(); (evaluation / "predictions").mkdir(parents=True)
    all_rows = []
    for label, count in zip(uptake_repair.NOVEL_CLASSES, (67, 59, 26, 24)):
        all_rows.extend((label, label) for _ in range(count))
    history(source / "history.csv", all_rows)
    (evaluation / "integrity_report.json").write_text(json.dumps({"status": "PASS"}))
    (evaluation / "predictions" / "step_000976.json").write_text(json.dumps({"rows": [{"image_id": "v", "true_label": "x", "predicted_label": "x"}]}))
    train = tmp_path / "train.csv"
    with train.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "label")); writer.writeheader()
        for label, count in zip(uptake_repair.NOVEL_CLASSES, (67, 59, 26, 24)):
            writer.writerows({"image": f"{label}_{i}", "label": label} for i in range(count))
    assert set(uptake_repair.repair_uptake(source_dir=source, evaluation_dir=evaluation, train_csv=train)["classes"]) == set(uptake_repair.NOVEL_CLASSES)
    with train.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=("image", "label")).writerow({"image": "extra", "label": "3_a9_2"})
    with pytest.raises(ValueError, match="stream support"):
        uptake_repair.repair_uptake(source_dir=source, evaluation_dir=evaluation, train_csv=train)


def test_repair_has_no_test_partition_access(tmp_path, monkeypatch):
    original_open = Path.open
    def guarded_open(self, *args, **kwargs):
        if self.name == "test.csv":
            raise AssertionError("trusted test must never be opened")
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    assert uptake_repair.class_uptake([], "3_a9_2")["N_available"] == 0


def test_launcher_has_required_non_overwrite_and_completed_cell_guards():
    script = (Path(__file__).parents[2] / "scripts" / "run-forgetting-grid.ps1").read_text(encoding="utf-8")
    for required in ("PlanOnly", "RUN FORGETTING GRID", "Target already exists",
                     "budget_longitudinal_development", "--execute", "benchmark_environment"):
        assert required in script
    assert "test.csv" not in script


def test_grid_plan_starts_with_hdc_measurement_cell_and_has_36_cells():
    plan = (Path(__file__).parents[1] / "results" / "forgetting_grid_development" / "GRID_PLAN.md").read_text(encoding="utf-8")
    commands = [line for line in plan.splitlines() if line.startswith("rtk proxy powershell")
                and "-Learner" in line and "-PlanOnly" not in line]
    assert len(commands) == 36
    assert commands[0].endswith("-Learner hdc -Arm frozen -Seed 0")


def test_batch_runner_declares_powercut_safe_attempt_and_progress_rules():
    script = (Path(__file__).parents[2] / "scripts" / "run-forgetting-grid-all.ps1").read_text(encoding="utf-8")
    for required in ("RUN FORGETTING GRID ALL", "batch_progress.jsonl", "Attempt cap reached",
                     "attempt{0:D2}", "SKIPPED (already complete)", "Add-Cell1Measurement",
                     "benchmark_environment", "8GB"):
        assert required in script
    assert "test.csv" not in script
