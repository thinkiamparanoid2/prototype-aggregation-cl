"""Synthetic guards for the new-dataset replication pipeline only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ssl_hdc_pilot.replication import pipeline


def _split(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "label"))
        writer.writeheader()
        writer.writerows({"image": image, "label": label} for image, label in rows)


def _spec(path: Path) -> None:
    path.write_text(json.dumps({"schema_version": 1, "tag": "synthetic_replication", "smoke": False,
        "execution": {"device": "cpu"}, "seeds": {"stream_order": [0], "projection": 3, "learner": 4, "replay": 5},
        "protocol": {"initially_known_classes": ["a", "b"],
            "class_introductions": [{"class_label": "c", "at_step": 2}],
            "bootstrap": {"confirmed_per_class": 1, "counts_toward_budget": False},
            "confidence_thresholds": [0.1], "label_budget": {"max_total_requests": 3,
                "count_low_confidence": True, "count_novel_registration": True}, "rolling_window": 2},
        "hdc": {"hypervector_dim": 10000},
        "baseline": {"head_type": "linear", "hidden_dim": None, "learning_rate": 0.1, "weight_decay": 0.0,
            "update_steps": 1, "ewc_lambda": 1.0, "ewc_decay": 0.9, "replay_batch_size": 1, "replay_buffer_size": 1},
        "metrics": {"stabilization_window": 1, "stabilization_min_accuracy": 0.8,
            "stabilization_consecutive_windows": 1, "ece_bins": 3}}, indent=2), encoding="utf-8")


def _inputs(tmp_path: Path) -> dict[str, Path]:
    train = tmp_path / "train.csv"; validation = tmp_path / "val.csv"; spec = tmp_path / "stream.json"
    rows = [(f"{label}{number}", label) for label in ("a", "b", "c") for number in range(3)]
    _split(train, rows); _split(validation, [(f"v{image}", label) for image, label in rows]); _spec(spec)
    ids = [image for image, _ in rows] + [f"v{image}" for image, _ in rows]
    vectors = np.asarray([[index + 1, (index % 3) - 1] for index in range(len(ids))], dtype=np.float32)
    embeddings = tmp_path / "saved_embeddings.npz"; np.savez_compressed(embeddings, image_ids=np.asarray(ids), embeddings=vectors)
    return {"train_csv": train, "validation_csv": validation, "stream_spec_path": spec, "embeddings_path": embeddings}


def test_pipeline_writes_immutable_receipt_and_shared_projection_snapshots(tmp_path: Path):
    inputs = _inputs(tmp_path)
    output = tmp_path / "cell"
    receipt = pipeline.run_cell(dataset="synthetic", arm="frozen-s", learner="hdc", seed=0,
        checkpoint_steps=[0, 7], output_dir=output, preflight_confirmed=True, **inputs)
    assert receipt["status"] == "PASS"
    assert (output / "metrics.json").is_file()
    assert (output / "projection_blobs").is_dir()
    with pytest.raises(FileExistsError):
        pipeline.run_cell(dataset="synthetic", arm="frozen-s", learner="hdc", seed=0,
            checkpoint_steps=[0, 7], output_dir=output, preflight_confirmed=True, **inputs)


def test_test_split_is_rejected_before_opening(tmp_path: Path, monkeypatch):
    inputs = _inputs(tmp_path)
    sealed = tmp_path / "test.csv"
    original_open = Path.open
    def guarded_open(self, *args, **kwargs):
        if self.name == "test.csv":
            raise AssertionError("sealed split must not be opened")
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(RuntimeError, match="sealed test split"):
        pipeline.plan_cell(dataset="synthetic", arm="frozen-s", learner="ncm", seed=0,
            checkpoint_steps=[0, 7], train_csv=inputs["train_csv"], validation_csv=sealed,
            stream_spec_path=inputs["stream_spec_path"], embeddings_path=inputs["embeddings_path"])


def test_plan_has_no_write_and_refuses_gpu_spec(tmp_path: Path):
    inputs = _inputs(tmp_path)
    before = sorted(path.name for path in tmp_path.iterdir())
    plan = pipeline.plan_cell(dataset="synthetic", arm="frozen-s", learner="ranpac", seed=0,
        checkpoint_steps=[0, 7], **inputs)
    assert plan["device"] == "cpu"
    assert sorted(path.name for path in tmp_path.iterdir()) == before
    payload = json.loads(inputs["stream_spec_path"].read_text(encoding="utf-8"))
    payload["execution"]["device"] = "cuda"
    inputs["stream_spec_path"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="CPU-only"):
        pipeline.plan_cell(dataset="synthetic", arm="frozen-s", learner="ncm", seed=0,
            checkpoint_steps=[0, 7], **inputs)
