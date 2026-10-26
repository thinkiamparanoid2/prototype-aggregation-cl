"""Apply the fixed post-split linear-probe gate to Arm-D checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from ssl_hdc_pilot.scripts.checkpoint_encoder import (
    inspect_checkpoint_export,
    load_checkpoint_export,
)
from ssl_hdc_pilot.scripts.build_trusted_splits import (
    CANONICAL_SOURCE_CSV,
    EXPECTED_HEADLINE_CLASS_COUNT,
    EXPECTED_TAIL_CLASSES,
    OFFICIAL_SEED,
    PROVENANCE_FIELDS,
    PROVENANCE_VERSION,
)
from ssl_hdc_pilot.scripts.extract_embeddings import (
    EncoderModelSpec,
    get_model_spec,
    ImageRecord,
    extract_records,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TRUSTED_IMAGE_ROOT = REPO_ROOT / "Data" / "test"
RESULTS_ROOT = REPO_ROOT / "ssl_hdc_pilot" / "results"
DEFAULT_CHECKPOINT_ROOT = RESULTS_ROOT / "arm_d_vitb16"
DEFAULT_SPLIT_ROOT = REPO_ROOT / "ssl_hdc_pilot" / "trusted_splits"
DEFAULT_OUTPUT = (
    REPO_ROOT / "ssl_hdc_pilot" / "results" / "arm_d_vitb16_probe_gate.json"
)
GATE_TOLERANCE = 0.005
PROBE_RANDOM_STATE = OFFICIAL_SEED
PROBE_MAX_ITER = 2_000
APPROVED_POLICY = "option-a-19class"
EXPECTED_TRAIN_COUNT = 3_396
EXPECTED_VAL_COUNT = 1_132


@dataclass(frozen=True)
class ApprovedSplit:
    records: list[ImageRecord]
    labels: np.ndarray


@dataclass(frozen=True)
class CheckpointRun:
    checkpoints: list[Path]
    model_spec: EncoderModelSpec


def load_approved_split(
    path: Path,
    *,
    expected_partition: str,
) -> ApprovedSplit:
    if not path.is_file():
        raise FileNotFoundError(f"official trusted split is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"image", "label", *PROVENANCE_FIELDS}
        if not required <= fields:
            missing = sorted(required - fields)
            raise RuntimeError(f"split lacks supervisor-approved provenance fields: {missing}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"approved split is empty: {path}")

    with (REPO_ROOT / CANONICAL_SOURCE_CSV).open(newline="", encoding="utf-8-sig") as handle:
        canonical_rows = list(csv.DictReader(handle))
    canonical_labels = {
        (row.get("image") or "").strip(): (row.get("label") or "").strip()
        for row in canonical_rows
    }

    trusted_root = TRUSTED_IMAGE_ROOT.resolve()
    records: list[ImageRecord] = []
    labels: list[str] = []
    seen: set[str] = set()
    expected = {
        "split_partition": expected_partition,
        "split_policy": APPROVED_POLICY,
        "split_seed": str(OFFICIAL_SEED),
        "supervisor_approved": "true",
        "provenance_version": PROVENANCE_VERSION,
        "source_csv": CANONICAL_SOURCE_CSV,
    }
    for row in rows:
        for field, value in expected.items():
            if row[field].strip() != value:
                raise RuntimeError(
                    f"split lacks approved provenance value {field}={value!r}"
                )
        image_id = row["image"].strip()
        label = row["label"].strip()
        if not image_id or not label or image_id in seen:
            raise RuntimeError(f"invalid or duplicate approved split row: {image_id!r}")
        if canonical_labels.get(image_id) != label:
            raise RuntimeError(f"split label does not match canonical trusted source: {image_id}")
        relative_path = Path(row["image_path"].strip())
        expected_path = Path("Data") / "test" / image_id / f"{image_id}.jpg"
        if relative_path != expected_path:
            raise RuntimeError(f"split image provenance path mismatch for {image_id}")
        image_path = (REPO_ROOT / relative_path).resolve()
        if not image_path.is_relative_to(trusted_root):
            raise RuntimeError(f"split image path escapes the trusted image root: {image_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"approved split image is missing: {image_path}")
        seen.add(image_id)
        records.append(ImageRecord(image_id=image_id, image_path=image_path))
        labels.append(label)
    return ApprovedSplit(records=records, labels=np.asarray(labels))


def validate_approved_split_pair(train_split: ApprovedSplit, val_split: ApprovedSplit) -> None:
    if len(train_split.records) != EXPECTED_TRAIN_COUNT:
        raise RuntimeError("approved Option-A training split must contain 3,396 rows")
    if len(val_split.records) != EXPECTED_VAL_COUNT:
        raise RuntimeError("approved Option-A validation split must contain 1,132 rows")
    if len(train_split.labels) != len(train_split.records) or len(val_split.labels) != len(val_split.records):
        raise RuntimeError("approved split record/label lengths differ")
    train_ids = {record.image_id for record in train_split.records}
    val_ids = {record.image_id for record in val_split.records}
    if train_ids & val_ids:
        raise RuntimeError("approved train and validation splits overlap")
    train_classes = set(train_split.labels)
    val_classes = set(val_split.labels)
    if train_classes != val_classes:
        raise RuntimeError("approved train and validation label sets differ")
    if len(train_classes) != EXPECTED_HEADLINE_CLASS_COUNT:
        raise RuntimeError("approved Option-A train/validation splits must contain 19 classes")
    if train_classes & set(EXPECTED_TAIL_CLASSES):
        raise RuntimeError("approved Option-A splits contain a preregistered tail class")


def _fit_score(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> float:
    probe = LogisticRegression(
        max_iter=PROBE_MAX_ITER,
        class_weight="balanced",
        random_state=PROBE_RANDOM_STATE,
    )
    probe.fit(train_x, train_y)
    predictions = probe.predict(val_x)
    return float(f1_score(val_y, predictions, average="macro"))


def evaluate_embedding_gate(
    checkpoint_embeddings: Mapping[str, Mapping[str, np.ndarray]],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    *,
    model_spec: EncoderModelSpec,
) -> dict[str, Any]:
    if "checkpoint_epoch0" not in checkpoint_embeddings:
        raise ValueError("frozen baseline checkpoint_epoch0 is required")
    adapted_names = [name for name in checkpoint_embeddings if name != "checkpoint_epoch0"]
    if not adapted_names:
        raise ValueError("at least one adapted checkpoint is required")

    table: list[dict[str, Any]] = []
    for name, split_embeddings in checkpoint_embeddings.items():
        train_x = np.asarray(split_embeddings["train"], dtype=np.float32)
        val_x = np.asarray(split_embeddings["val"], dtype=np.float32)
        if train_x.ndim != 2 or val_x.ndim != 2:
            raise ValueError(f"checkpoint embeddings must be matrices: {name}")
        if (
            train_x.shape[1] != model_spec.embedding_dim
            or val_x.shape[1] != model_spec.embedding_dim
        ):
            raise ValueError(
                "checkpoint embeddings must be raw "
                f"{model_spec.embedding_dim}-d features: {name}"
            )
        if train_x.shape[0] != len(train_labels) or val_x.shape[0] != len(val_labels):
            raise ValueError(f"checkpoint embedding/label length mismatch: {name}")
        score = _fit_score(train_x, train_labels, val_x, val_labels)
        table.append({"checkpoint": name, "val_macro_f1": score})

    baseline = next(row for row in table if row["checkpoint"] == "checkpoint_epoch0")
    adapted = [row for row in table if row["checkpoint"] != "checkpoint_epoch0"]
    best_adapted = max(adapted, key=lambda row: row["val_macro_f1"])
    selected = max(table, key=lambda row: row["val_macro_f1"])
    threshold = baseline["val_macro_f1"] - GATE_TOLERANCE
    passed = best_adapted["val_macro_f1"] >= threshold
    return {
        "rung": model_spec.rung,
        "checkpoint_table": table,
        "frozen_baseline_checkpoint": "checkpoint_epoch0",
        "frozen_baseline_val_macro_f1": baseline["val_macro_f1"],
        "best_adapted_checkpoint": best_adapted["checkpoint"],
        "best_adapted_val_macro_f1": best_adapted["val_macro_f1"],
        "tolerance": GATE_TOLERANCE,
        "pass_threshold": threshold,
        "pass": bool(passed),
        "selected_checkpoint": selected["checkpoint"],
    }


def load_checkpoint_run(
    root: Path,
    *,
    results_root: Path = RESULTS_ROOT,
) -> CheckpointRun:
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory is missing: {root}")
    config_path = root / "run_config.json"
    if not config_path.is_file():
        raise RuntimeError("checkpoint directory lacks its immutable run_config.json")
    try:
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
        training_config = run_config["training_config"]
        expected_epochs = int(training_config["epochs"])
        model_id = str(training_config["model_id"])
        model_spec = get_model_spec(model_id)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "checkpoint run config lacks a valid model id or epoch cadence"
        ) from error
    expected_root = (results_root / model_spec.arm_d_directory).resolve()
    if root.resolve() != expected_root:
        raise RuntimeError(
            f"{model_id} checkpoint run must use directory {expected_root}"
        )
    if not 1 <= expected_epochs <= 10:
        raise RuntimeError("checkpoint run config has an invalid epoch cadence")
    if training_config.get("smoke") is not False:
        raise RuntimeError("official probe gate refuses smoke checkpoints")
    checkpoints: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint_epoch*"):
        suffix = path.name.removeprefix("checkpoint_epoch")
        if not path.is_dir() or not suffix.isdigit():
            continue
        epoch = int(suffix)
        provenance = inspect_checkpoint_export(path, results_root=results_root)
        if provenance.model_id != model_id or provenance.epoch != epoch:
            raise RuntimeError("checkpoint export identity does not match probe run")
        checkpoints.append((epoch, path))
    checkpoints.sort(key=lambda item: item[0])
    actual_epochs = [epoch for epoch, _ in checkpoints]
    required_epochs = list(range(expected_epochs + 1))
    if actual_epochs != required_epochs:
        raise RuntimeError(
            "complete checkpoint cadence is required before gating: "
            f"expected {required_epochs}, found {actual_epochs}"
        )
    return CheckpointRun(
        checkpoints=[path for _, path in checkpoints],
        model_spec=model_spec,
    )
def extract_checkpoint_embeddings(
    checkpoints: Sequence[Path],
    train_split: ApprovedSplit,
    val_split: ApprovedSplit,
    *,
    model_spec: EncoderModelSpec,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        model_spec.model_id,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_spec.model_id,
        local_files_only=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    extracted: dict[str, dict[str, np.ndarray]] = {}
    for checkpoint in checkpoints:
        provenance = load_checkpoint_export(checkpoint, model, device)
        if provenance.model_id != model_spec.model_id:
            raise RuntimeError("checkpoint loader returned a different encoder model")
        _, train_embeddings = extract_records(
            records=train_split.records,
            model=model,
            processor=processor,
            device=device,
            batch_size=batch_size,
        )
        _, val_embeddings = extract_records(
            records=val_split.records,
            model=model,
            processor=processor,
            device=device,
            batch_size=batch_size,
        )
        for partition, values in (
            ("trusted_train", train_embeddings),
            ("trusted_val", val_embeddings),
        ):
            if values.shape[1] != model_spec.embedding_dim:
                raise RuntimeError(
                    f"{partition} embeddings for {model_spec.rung} must be "
                    f"{model_spec.embedding_dim}-d, got {values.shape[1]}"
                )
        extracted[checkpoint.name] = {
            "train": train_embeddings,
            "val": val_embeddings,
        }
        print(f"Extracted trusted train/val embeddings for {checkpoint.name}.", flush=True)
    return extracted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--trusted-train-csv", type=Path, default=DEFAULT_SPLIT_ROOT / "train.csv")
    parser.add_argument("--trusted-val-csv", type=Path, default=DEFAULT_SPLIT_ROOT / "val.csv")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    train_split = load_approved_split(args.trusted_train_csv, expected_partition="train")
    val_split = load_approved_split(args.trusted_val_csv, expected_partition="val")
    validate_approved_split_pair(train_split, val_split)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    checkpoint_run = load_checkpoint_run(args.checkpoint_dir)
    embeddings = extract_checkpoint_embeddings(
        checkpoint_run.checkpoints,
        train_split,
        val_split,
        model_spec=checkpoint_run.model_spec,
        device=device,
        batch_size=args.batch_size,
    )
    verdict = evaluate_embedding_gate(
        embeddings,
        train_split.labels,
        val_split.labels,
        model_spec=checkpoint_run.model_spec,
    )
    payload = {
        **verdict,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "trusted_train_csv": str(args.trusted_train_csv.resolve()),
        "trusted_val_csv": str(args.trusted_val_csv.resolve()),
        "model_id": checkpoint_run.model_spec.model_id,
        "embedding_dim": checkpoint_run.model_spec.embedding_dim,
        "metric": "macro_f1",
        "probe": {
            "class": "sklearn.linear_model.LogisticRegression",
            "max_iter": PROBE_MAX_ITER,
            "class_weight": "balanced",
            "random_state": PROBE_RANDOM_STATE,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
