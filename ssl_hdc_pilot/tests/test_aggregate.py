"""Targeted tests for aggregate.py against a tiny fake result set -- no real
cell is read from the actual results tree unless a test explicitly points at
it, and this suite must succeed even when the real "Original" validation
root does not exist (partial results)."""
from __future__ import annotations

import json
from pathlib import Path

from ssl_hdc_pilot.replication import aggregate as agg


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_new_cell(cells_root: Path, replication_root: Path, *, dataset: str, arm: str, learner: str,
                    seed: int, macro_f1: float, accuracy: float, ece: float, mean_forgetting: float,
                    labels_consumed: int, label_budget: int, classes: list[str], novel_classes: list[str],
                    rows: list[tuple[str, str]]) -> Path:
    output_dir = agg.cell_output_dir(cells_root, dataset, arm, learner, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "snapshots").mkdir(exist_ok=True)

    per_class_recall = {cls: 1.0 for cls in classes}
    metrics = {
        "final_validation": {"macro_f1": macro_f1, "accuracy": accuracy, "ece": ece,
                              "per_class_recall": per_class_recall},
        "forgetting": {"mean_forgetting": mean_forgetting},
        "labels_consumed": labels_consumed, "label_budget": label_budget, "label_budget_match": True,
    }
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "integrity_report.json", {"status": "PASS", "artifacts": {}})

    predictions_rows = [{"image_id": f"img{i}", "true_label": t, "predicted_label": p, "top_probability": 0.9}
                        for i, (t, p) in enumerate(rows)]
    _write_json(output_dir / "snapshots" / "step_000010_predictions.json", {"rows": predictions_rows})
    _write_json(output_dir / "snapshot_index.json", {
        "snapshots": [{"processed_stream_events": 10,
                       "predictions": "snapshots/step_000010_predictions.json"}]})

    feasibility_path = replication_root / "specs" / f"{dataset}_stream_feasibility.json"
    if not feasibility_path.is_file():
        _write_json(feasibility_path, {"novel_classes_in_entry_order": novel_classes,
                                        "introductions_at": [5], "stream_length": 10})
    return output_dir


def test_load_new_cell_parses_metrics_predictions_and_novel_terminal_accuracy(tmp_path: Path) -> None:
    cells_root = tmp_path / "cells"
    replication_root = tmp_path / "replication"
    _make_new_cell(cells_root, replication_root, dataset="PlantDoc", arm="frozen_vits16", learner="hdc", seed=0,
                    macro_f1=0.7, accuracy=0.72, ece=0.05, mean_forgetting=0.1, labels_consumed=100,
                    label_budget=164, classes=["a", "b", "c"], novel_classes=["c"],
                    rows=[("a", "a"), ("a", "b"), ("b", "b"), ("c", "c"), ("c", "a")])

    record = agg.load_new_cell(cells_root=cells_root, replication_root=replication_root,
                               dataset="PlantDoc", arm="frozen_vits16", learner="hdc", seed=0)
    assert record is not None
    assert record["macro_f1"] == 0.7
    assert record["labels_consumed"] == 100
    assert record["n_classes"] == 3
    assert set(record["per_class_f1"]) == {"a", "b", "c"}
    # class "a": TP=1 (a->a), FN=1 (a->b), FP=1 (c->a) => F1 = 2/ (2+1+1) = 0.5
    assert abs(record["per_class_f1"]["a"] - 0.5) < 1e-9
    assert record["novel_terminal_accuracy"] == {"c": 1.0}
    assert record["novel_terminal_accuracy_method"] == "validation_population_final_checkpoint_recall_proxy"


def test_load_new_cell_returns_none_when_absent(tmp_path: Path) -> None:
    record = agg.load_new_cell(cells_root=tmp_path / "cells", replication_root=tmp_path / "replication",
                               dataset="PlantDoc", arm="frozen_vits16", learner="hdc", seed=0)
    assert record is None


def test_load_all_new_cells_is_partial_by_default(tmp_path: Path) -> None:
    cells_root = tmp_path / "cells"
    replication_root = tmp_path / "replication"
    _make_new_cell(cells_root, replication_root, dataset="PlantDoc", arm="frozen_vits16", learner="ncm", seed=0,
                    macro_f1=0.5, accuracy=0.5, ece=0.1, mean_forgetting=0.0, labels_consumed=10,
                    label_budget=164, classes=["a", "b"], novel_classes=[], rows=[("a", "a"), ("b", "b")])
    records = agg.load_all_new_cells(cells_root=cells_root, replication_root=replication_root)
    assert len(records) == 1
    assert records[0]["dataset"] == "PlantDoc"
    assert records[0]["learner"] == "ncm"


def test_build_group_rows_flags_incomplete_seeds_and_class_count_mismatch(tmp_path: Path) -> None:
    records = [
        {"dataset": "PlantDoc", "arm": "frozen_vits16", "learner": "hdc", "seed": 0, "macro_f1": 0.6,
         "accuracy": 0.6, "ece": 0.1, "mean_forgetting": 0.1, "labels_consumed": 100, "n_classes": 3,
         "classes": ["a", "b", "c"], "novel_terminal_accuracy": {"c": 0.5}},
        {"dataset": "PlantDoc", "arm": "frozen_vits16", "learner": "hdc", "seed": 1, "macro_f1": 0.62,
         "accuracy": 0.61, "ece": 0.09, "mean_forgetting": 0.11, "labels_consumed": 101, "n_classes": 4,
         "classes": ["a", "b", "c", "d"], "novel_terminal_accuracy": {"c": 0.6}},
    ]
    rows = agg.build_group_rows(records)
    assert len(rows) == 1
    row = rows[0]
    assert row["n_seeds"] == 2
    assert row["incomplete_seeds"] is True  # only 2 of the expected 5 seeds
    assert row["class_count_flag"] is True  # 3 vs 4 classes across seeds
    assert row["macro_f1_mean"] is not None and row["macro_f1_sd"] is not None


def test_build_label_count_flags_detects_mismatched_labels_consumed() -> None:
    records = [
        {"dataset": "PlantDoc", "arm": "frozen_vits16", "learner": "hdc", "seed": 0, "labels_consumed": 100},
        {"dataset": "PlantDoc", "arm": "frozen_vits16", "learner": "ncm", "seed": 0, "labels_consumed": 90},
        {"dataset": "PlantDoc", "arm": "frozen_vits16", "learner": "hdc", "seed": 1, "labels_consumed": 100},
        {"dataset": "PlantDoc", "arm": "frozen_vits16", "learner": "ncm", "seed": 1, "labels_consumed": 100},
    ]
    flags = agg.build_label_count_flags(records)
    assert len(flags) == 1
    assert flags[0]["seed"] == 0
    assert flags[0]["labels_consumed_by_learner"] == {"hdc": 100, "ncm": 90}


def test_win_tie_loss_counts_wins_ties_and_losses() -> None:
    hdc = {0: 0.80, 1: 0.50, 2: 0.60}
    controls = {"ncm": {0: 0.70, 1: 0.50, 2: 0.65}, "ranpac": {0: 0.60, 1: 0.40, 2: 0.61}}
    outcome = agg.win_tie_loss(hdc, controls)
    # seed 0: best control = 0.70, hdc 0.80 -> win
    # seed 1: best control = 0.50, hdc 0.50 -> tie
    # seed 2: best control = 0.65, hdc 0.60 -> loss
    assert outcome["seeds_considered"] == 3
    assert outcome["wins"] == 1
    assert outcome["ties"] == 1
    assert outcome["losses"] == 1
    assert outcome["diffs"] == [0.80 - 0.70, 0.0, 0.60 - 0.65]


def test_paired_bootstrap_interval_is_deterministic_and_reasonable() -> None:
    diffs = [0.05, 0.06, 0.04, 0.05, 0.07]
    first = agg.paired_bootstrap_interval(diffs, resamples=2000, seed=1)
    second = agg.paired_bootstrap_interval(diffs, resamples=2000, seed=1)
    assert first == second
    assert first["ci_low"] <= first["mean_diff"] <= first["ci_high"]
    assert first["excludes_zero"] is True  # all diffs strictly positive

    mixed = agg.paired_bootstrap_interval([0.1, -0.1, 0.05, -0.05, 0.0], resamples=2000, seed=1)
    assert mixed["excludes_zero"] is False

    empty = agg.paired_bootstrap_interval([])
    assert empty["mean_diff"] is None and empty["resamples"] == 0


def test_run_produces_output_with_partial_results_and_missing_original(tmp_path: Path) -> None:
    cells_root = tmp_path / "cells"
    replication_root = tmp_path / "replication"
    _make_new_cell(cells_root, replication_root, dataset="PlantDoc", arm="frozen_vits16", learner="hdc", seed=0,
                    macro_f1=0.7, accuracy=0.72, ece=0.05, mean_forgetting=0.1, labels_consumed=100,
                    label_budget=164, classes=["a", "b", "c"], novel_classes=["c"],
                    rows=[("a", "a"), ("b", "b"), ("c", "c")])

    output_dir = tmp_path / "comparison"
    missing_validation_root = tmp_path / "no_such_original_root"
    report = agg.run(cells_root=cells_root, replication_root=replication_root,
                     validation_root=missing_validation_root, capture_root=tmp_path / "no_capture",
                     uptake_root=tmp_path / "no_uptake", output_dir=output_dir)

    assert report["new_cells_read"] == 1
    assert report["original_cells_read"] == 0
    assert (output_dir / "COMPARISON.md").is_file()
    assert (output_dir / "macro_f1.csv").is_file()
    assert (output_dir / "win_tie_loss.csv").is_file()
    assert (output_dir / "label_count_flags.csv").is_file()
    comparison_text = (output_dir / "COMPARISON.md").read_text(encoding="utf-8")
    assert "PlantDoc" in comparison_text
    assert "seeds=1/5" in comparison_text

    # rerunning (idempotent aggregation, unlike aggregate_grid's refuse-to-overwrite) must not fail
    agg.run(cells_root=cells_root, replication_root=replication_root, validation_root=missing_validation_root,
            capture_root=tmp_path / "no_capture", uptake_root=tmp_path / "no_uptake", output_dir=output_dir)
