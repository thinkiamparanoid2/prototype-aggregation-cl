import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ssl_hdc_pilot.scripts.probe_gate import (
    ApprovedSplit,
    evaluate_embedding_gate,
    load_checkpoint_run,
    load_approved_split,
    validate_approved_split_pair,
)
from ssl_hdc_pilot.scripts.extract_embeddings import (
    HEADLINE_MODEL_ID,
    LADDER_MODEL_SPECS,
    ImageRecord,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_probe_gate_refuses_split_without_approved_provenance(tmp_path: Path) -> None:
    split = tmp_path / "train.csv"
    _write_csv(split, ["image", "label"], [{"image": "x", "label": "a"}])

    with pytest.raises(RuntimeError, match="provenance"):
        load_approved_split(split, expected_partition="train")


def test_probe_gate_refuses_truncated_class_complete_splits() -> None:
    labels = np.asarray([f"class_{index}" for index in range(19)])
    train = ApprovedSplit(
        records=[ImageRecord(f"train_{index}", Path("unused")) for index in range(19)],
        labels=labels,
    )
    val = ApprovedSplit(
        records=[ImageRecord(f"val_{index}", Path("unused")) for index in range(19)],
        labels=labels.copy(),
    )

    with pytest.raises(RuntimeError, match="3,396"):
        validate_approved_split_pair(train, val)


def test_probe_gate_refuses_tail_class_in_19_class_composition() -> None:
    labels = ["6_a11_2", *[f"class_{index}" for index in range(18)]]
    train_labels = np.asarray([labels[index % 19] for index in range(3_396)])
    val_labels = np.asarray([labels[index % 19] for index in range(1_132)])
    train = ApprovedSplit(
        records=[ImageRecord(f"train_{index}", Path("unused")) for index in range(3_396)],
        labels=train_labels,
    )
    val = ApprovedSplit(
        records=[ImageRecord(f"val_{index}", Path("unused")) for index in range(1_132)],
        labels=val_labels,
    )

    with pytest.raises(RuntimeError, match="tail class"):
        validate_approved_split_pair(train, val)


def _write_complete_checkpoint(root: Path, epoch: int, model_id: str = HEADLINE_MODEL_ID) -> None:
    checkpoint = root / f"checkpoint_epoch{epoch}"
    checkpoint.mkdir()
    (checkpoint / "encoder.npz").write_bytes(b"ema-encoder")
    if epoch > 0:
        (checkpoint / "training_state.npz").write_bytes(b"resume-state")
    (checkpoint / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "complete": True,
                "epoch": epoch,
                "model_id": model_id,
                "labels_used": False,
                "trusted_pixels_used": False,
                "exported_encoder": "ema_teacher",
                "frozen_baseline": epoch == 0,
            }
        ),
        encoding="utf-8",
    )


def _write_run_config(
    root: Path,
    *,
    epochs: int,
    smoke: bool,
    model_id: str = HEADLINE_MODEL_ID,
) -> None:
    root.mkdir()
    (root / "run_config.json").write_text(
        json.dumps(
            {
                "training_config": {
                    "epochs": epochs,
                    "smoke": smoke,
                    "model_id": model_id,
                }
            }
        ),
        encoding="utf-8",
    )


def test_probe_gate_requires_full_configured_checkpoint_cadence(tmp_path: Path) -> None:
    root = tmp_path / "arm_d_vitb16"
    _write_run_config(root, epochs=2, smoke=False)
    _write_complete_checkpoint(root, 0)
    _write_complete_checkpoint(root, 1)

    with pytest.raises(RuntimeError, match="complete checkpoint cadence"):
        load_checkpoint_run(root, results_root=tmp_path)

    _write_complete_checkpoint(root, 2)
    assert [
        path.name
        for path in load_checkpoint_run(root, results_root=tmp_path).checkpoints
    ] == [
        "checkpoint_epoch0",
        "checkpoint_epoch1",
        "checkpoint_epoch2",
    ]


def test_probe_gate_refuses_smoke_checkpoint_directory(tmp_path: Path) -> None:
    root = tmp_path / "arm_d_vitb16"
    _write_run_config(root, epochs=1, smoke=True)
    _write_complete_checkpoint(root, 0)
    _write_complete_checkpoint(root, 1)

    with pytest.raises(RuntimeError, match="smoke checkpoints"):
        load_checkpoint_run(root, results_root=tmp_path)


def _separable(labels: np.ndarray) -> np.ndarray:
    sign = np.where(labels == "a", -1.0, 1.0).astype(np.float32)
    embeddings = np.zeros((len(labels), 384), dtype=np.float32)
    embeddings[:, 0] = sign
    embeddings[:, 1] = sign * 0.5
    return embeddings


def _uninformative(size: int) -> np.ndarray:
    return np.zeros((size, 384), dtype=np.float32)


def test_gate_fails_and_selects_frozen_when_adaptation_is_worse() -> None:
    train_labels = np.asarray(["a", "b"] * 20)
    val_labels = np.asarray(["a", "b"] * 10)
    embeddings = {
        "checkpoint_epoch0": {
            "train": _separable(train_labels),
            "val": _separable(val_labels),
        },
        "checkpoint_epoch1": {
            "train": _uninformative(len(train_labels)),
            "val": _uninformative(len(val_labels)),
        },
    }

    vits = LADDER_MODEL_SPECS["facebook/dinov3-vits16-pretrain-lvd1689m"]
    verdict = evaluate_embedding_gate(embeddings, train_labels, val_labels, model_spec=vits)

    assert verdict["pass"] is False
    assert verdict["selected_checkpoint"] == "checkpoint_epoch0"
    assert verdict["best_adapted_checkpoint"] == "checkpoint_epoch1"
    assert verdict["rung"] == "vits16"


def test_gate_passes_and_selects_better_adapted_checkpoint() -> None:
    train_labels = np.asarray(["a", "b"] * 20)
    val_labels = np.asarray(["a", "b"] * 10)
    embeddings = {
        "checkpoint_epoch0": {
            "train": _uninformative(len(train_labels)),
            "val": _uninformative(len(val_labels)),
        },
        "checkpoint_epoch1": {
            "train": _separable(train_labels),
            "val": _separable(val_labels),
        },
        "checkpoint_epoch2": {
            "train": _uninformative(len(train_labels)),
            "val": _uninformative(len(val_labels)),
        },
    }

    vits = LADDER_MODEL_SPECS["facebook/dinov3-vits16-pretrain-lvd1689m"]
    verdict = evaluate_embedding_gate(embeddings, train_labels, val_labels, model_spec=vits)

    assert verdict["pass"] is True
    assert verdict["selected_checkpoint"] == "checkpoint_epoch1"
    assert verdict["best_adapted_checkpoint"] == "checkpoint_epoch1"
    assert verdict["tolerance"] == 0.005


def test_vitb_checkpoint_run_refuses_vits_trusted_embeddings(tmp_path: Path) -> None:
    root = tmp_path / "arm_d_vitb16"
    _write_run_config(root, epochs=1, smoke=False)
    _write_complete_checkpoint(root, 0)
    _write_complete_checkpoint(root, 1)
    run = load_checkpoint_run(root, results_root=tmp_path)
    train_labels = np.asarray(["a", "b"] * 20)
    val_labels = np.asarray(["a", "b"] * 10)
    vits_embeddings = {
        "checkpoint_epoch0": {
            "train": _separable(train_labels),
            "val": _separable(val_labels),
        },
        "checkpoint_epoch1": {
            "train": _separable(train_labels),
            "val": _separable(val_labels),
        },
    }

    with pytest.raises(ValueError, match="raw 768-d features"):
        evaluate_embedding_gate(
            vits_embeddings,
            train_labels,
            val_labels,
            model_spec=run.model_spec,
        )
