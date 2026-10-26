"""Evaluate a saved Experiment-A final state on validation only, never test.

Defaults to read-only preparation. Real scoring is manually user-launched.
This is development evidence, not longitudinal evaluation or a benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from common.thesis_eval import (
    classification_metrics_from_labels, expected_calibration_error_from_confidence, stage_recall,
)
from ssl_hdc_pilot.scripts import learner_checkpoint as checkpoints
from ssl_hdc_pilot.scripts import stream_runner as runner
from ssl_hdc_pilot.scripts.build_stream_spec import _load_approved_partition
from ssl_hdc_pilot.scripts.experiment_a import _json_hash
from ssl_hdc_pilot.scripts.reconstruct_run import PROTOCOL_PATH, _git_provenance, _write_json


SOURCE_FILES = frozenset({"base_stream_spec.json", "final_state.npz", "final_training_predictions.json",
                          "history.csv", "manifest.json", "shared_label_schedule.json", "summary.json"})


def _fingerprint(learner: Any) -> str:
    arrays: dict[str, np.ndarray] = {}
    state = checkpoints._export_learner(learner, arrays)
    rng = checkpoints._capture_random_state(arrays)
    return _json_hash({"state": state, "rng": rng, "arrays": {
        key: {"shape": list(value.shape), "dtype": str(value.dtype),
              "sha256": checkpoints._array_digest(value)} for key, value in arrays.items()}})


def evaluate_fixed_population(learner: Any, rows: Sequence[tuple[str, str]],
                              embeddings: Mapping[str, torch.Tensor], *,
                              classes: Sequence[str]) -> dict[str, Any]:
    """Predict only, with explicit macro denominators and state/RNG purity checks."""
    classes = list(classes)
    ids = [image_id for image_id, _ in rows]
    if (not rows or len(ids) != len(set(ids)) or any(not x for x in ids)
            or not classes or len(classes) != len(set(classes))
            or any(len(x.split("_")) != 3 for x in classes)):
        raise ValueError("invalid fixed population or crop_disease_stage class universe")
    if set(label for _, label in rows) != set(classes):
        raise ValueError("fixed population must contain every declared class and no others")
    if not set(learner.known_classes) <= set(classes):
        raise ValueError("learner registry is outside the fixed class universe")
    if any(image_id not in embeddings for image_id in ids):
        raise ValueError("missing fixed-population embedding")
    before = _fingerprint(learner)
    model = getattr(learner, "model", None)
    modes = [(module, module.training) for module in model.modules()] if model is not None else []
    predictions = []
    try:
        with torch.inference_mode():
            for image_id, label in rows:
                prediction = learner.predict(embeddings[image_id])
                if prediction.label not in classes or not np.isfinite(prediction.top_probability):
                    raise ValueError("invalid prediction for fixed population")
                if not 0 <= prediction.top_probability <= 1:
                    raise ValueError("invalid top probability")
                predictions.append({"image_id": image_id, "true_label": label,
                                    "predicted_label": prediction.label,
                                    "top_probability": prediction.top_probability})
    finally:
        for module, training in modes:
            module.training = training
    after = _fingerprint(learner)
    if before != after:
        raise RuntimeError("evaluation changed learner state or RNG; results rejected")
    truth = [r["true_label"] for r in predictions]
    predicted = [r["predicted_label"] for r in predictions]
    metrics = classification_metrics_from_labels(truth, predicted, classes=classes)
    metrics.update(per_stage_recall=stage_recall(truth, predicted),
                   ece=expected_calibration_error_from_confidence(
                       [r["top_probability"] for r in predictions],
                       [a == b for a, b in zip(truth, predicted)], n_bins=15),
                   confusion_matrix=confusion_matrix(truth, predicted, labels=classes).tolist())
    return {"classes": classes, "samples": len(rows), "metrics": metrics, "rows": predictions,
            "unregistered_classes": sorted(set(classes) - set(learner.known_classes)),
            "learner_rng_fingerprint_before": before, "learner_rng_fingerprint_after": after}


def _unchanged(pins: dict[str, dict[str, str]]) -> None:
    for name, record in pins.items():
        if runner._file_sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"input changed: {name}")


def _prepare(source_dir: Path, embeddings_path: Path, validation_csv: Path,
             smoke_mode: bool) -> tuple:
    canonical = runner.DEFAULT_APPROVED_SPLIT_DIR
    # Reject test/custom real partitions before reading any inputs.
    if not smoke_mode and validation_csv.resolve() != (canonical / "val.csv").resolve():
        raise RuntimeError("validation-only: requires canonical val.csv; rejected before input access")
    if smoke_mode and validation_csv.resolve().is_relative_to(canonical.resolve()):
        raise RuntimeError("synthetic smoke cannot open canonical partitions")
    pins: dict[str, dict[str, str]] = {}

    def pin(name: str, path: Path) -> None:
        pins[name] = {"path": str(path.resolve()), "sha256": runner._file_sha256(path)}

    pin("source_report", source_dir / "integrity_report.json")
    report = json.loads((source_dir / "integrity_report.json").read_text(encoding="utf-8"))
    if (report.get("status") != "PASS" or report.get("experiment") != "A"
            or report.get("stage") != "development_only" or report.get("failures") != []
            or report.get("inputs_changed_during_run") != [] or report.get("code_changed_during_run") != []
            or set(report.get("artifacts", {})) != SOURCE_FILES):
        raise ValueError("source requires a complete PASS Experiment-A development receipt")
    for name, digest in report["artifacts"].items():
        pin("source/" + name, source_dir / name)
        if pins["source/" + name]["sha256"] != digest:
            raise ValueError(f"source artifact hash mismatch: {name}")
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_role = "synthetic_smoke" if smoke_mode else "trusted_train"
    if manifest.get("data_role") != expected_role:
        raise ValueError("source data role differs from requested mode")
    if (manifest.get("stage") != "development_only" or manifest["policy"].get("experiment") != "A"
            or manifest["learner"] != report.get("learner")
            or manifest["stream_seed"] != report.get("stream_seed")
            or manifest["policy"]["additional_label_budget"] != report.get("budget")
            or report.get("labels_consumed") != report.get("budget")
            or manifest["policy"]["schedule_sha256"] != report.get("schedule_sha256")
            or manifest["policy_sha256"] != _json_hash(manifest["policy"])):
        raise ValueError("source manifest and receipt disagree")
    train_path = Path(manifest["inputs"]["train"]["path"])
    if not smoke_mode:
        if train_path.resolve() != (canonical / "train.csv").resolve():
            raise ValueError("source training path is not canonical")
        if (embeddings_path.name not in {"arm_f_vitb16_trusted_all.npz", "arm_d_vitb16_ep3_trusted_all.npz"}
                or embeddings_path.resolve() != (runner.REPO_ROOT / "ssl_hdc_pilot/results" / embeddings_path.name).resolve()):
            raise ValueError("validation requires a canonical ViT-B embedding arm")
    elif train_path.resolve().is_relative_to(canonical.resolve()):
        raise ValueError("synthetic source cannot use canonical training data")
    pin("train", train_path)
    pin("embeddings", embeddings_path)
    pin("validation", validation_csv)
    for name in ("train", "embeddings"):
        if pins[name]["sha256"] != manifest["inputs"][name]["sha256"]:
            raise ValueError(f"{name} differs from source training input")
    current_code = runner._checkpoint_code_hashes()
    if not set(current_code) <= set(manifest["code_hashes"]):
        raise ValueError("source lacks required code provenance")
    for name, digest in manifest["code_hashes"].items():
        path = (runner.REPO_ROOT / name).resolve()
        if not path.is_relative_to(runner.REPO_ROOT.resolve()):
            raise ValueError("source code path escapes repository")
        pin("code/" + name, path)
        if pins["code/" + name]["sha256"] != digest:
            raise ValueError(f"source code changed; compatibility review required: {name}")
    name = "ssl_hdc_pilot/scripts/evaluate_validation.py"
    pin("code/" + name, runner.REPO_ROOT / name)
    pin("protocol", PROTOCOL_PATH)
    if smoke_mode:
        train_rows = runner._load_split_rows(train_path, smoke_mode=True)
        rows = runner._load_split_rows(validation_csv, smoke_mode=True)
    else:
        train_rows = _load_approved_partition(canonical, "train")
        rows = _load_approved_partition(canonical, "val")
        if (len(train_rows), len(rows)) != (3396, 1132):
            raise ValueError("approved train/validation sizes differ")
    classes = sorted({label for _, label in train_rows})
    if set(label for _, label in rows) != set(classes):
        raise ValueError("train and validation classes differ")
    if {x for x, _ in rows} & {x for x, _ in train_rows}:
        raise ValueError("training/validation IDs overlap")
    embeddings, dimension = runner._load_embedding_map(embeddings_path)
    if any(image_id not in embeddings for image_id, _ in rows):
        raise ValueError("missing validation embedding")
    _unchanged(pins)
    return manifest, report, rows, classes, embeddings, dimension, pins


def plan_validation(*, source_dir: Path, embeddings_path: Path, validation_csv: Path,
                    smoke_mode: bool = False) -> dict[str, Any]:
    manifest, report, rows, classes, _, dimension, pins = _prepare(
        source_dir, embeddings_path, validation_csv, smoke_mode)
    return {"stage": "development_only", "data_role": "synthetic_smoke" if smoke_mode else "trusted_val",
            "learner": manifest["learner"], "budget": report["budget"], "samples": len(rows),
            "classes": classes, "embedding_dim": dimension, "inputs": pins,
            "scope": "preparation only; no learner construction or scoring; no test access"}


def run_validation(*, source_dir: Path, embeddings_path: Path, validation_csv: Path,
                   output_dir: Path, smoke_mode: bool = False,
                   preflight_confirmed: bool = False) -> dict[str, Any]:
    if not smoke_mode and not preflight_confirmed:
        raise RuntimeError("validation scoring requires manual user launch and preflight")
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite/reuse validation output")
    manifest, report, rows, classes, embeddings, dimension, pins = _prepare(
        source_dir, embeddings_path, validation_csv, smoke_mode)
    checkpoint = checkpoints.load_learner_checkpoint(source_dir / "final_state.npz",
        device=manifest["environment"]["requested_device"], restore_rng=False)
    context = checkpoint.context
    for field in ("data_role", "learner", "stream_seed", "policy", "inputs", "code_hashes", "bootstrap_image_ids"):
        if context.get(field) != manifest[field]:
            raise ValueError(f"checkpoint and manifest disagree: {field}")
    training_ids = context["bootstrap_image_ids"] + context["stream_image_ids"]
    if set(training_ids) & {x for x, _ in rows}:
        raise ValueError("checkpoint training IDs overlap validation")
    if checkpoint.learner.input_dim != dimension:
        raise ValueError("checkpoint/embedding dimension mismatch")
    output_dir.mkdir(parents=True, exist_ok=False)
    result = evaluate_fixed_population(checkpoint.learner, rows, embeddings, classes=classes)
    _unchanged(pins)
    role = "synthetic_smoke" if smoke_mode else "trusted_val"
    _write_json(output_dir / "predictions.json", {"data_role": role, "rows": result.pop("rows")})
    _write_json(output_dir / "metrics.json", {"data_role": role, **result})
    _write_json(output_dir / "manifest.json", {"stage": "development_only", "data_role": role,
        "learner": manifest["learner"], "budget": report["budget"], "stream_seed": manifest["stream_seed"],
        "inputs": pins, "git": _git_provenance(), "device": str(checkpoint.learner.device),
        "population_sha256": _json_hash(rows), "classes": classes,
        "probability_semantics": "existing learner softmax over registered classes; ECE 15 bins",
        "selection": "saved final state; no model selection, updates or calibration fitting"})
    _unchanged(pins)
    integrity = {"status": "PASS", "stage": "development_only", "data_role": role,
        "samples": len(rows), "class_count": len(classes), "learner": manifest["learner"],
        "budget": report["budget"], "state_and_rng_unchanged": True,
        "artifacts": {p.name: runner._file_sha256(p) for p in output_dir.iterdir() if p.is_file()},
        "scope": "fixed validation final-state development only; not test, forgetting, uptake or benchmark"}
    _write_json(output_dir / "integrity_report.json", integrity)
    return integrity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--validation-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-confirmed", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    kwargs = dict(source_dir=args.source_dir, embeddings_path=args.embeddings,
                  validation_csv=args.validation_split, smoke_mode=args.smoke)
    if args.execute:
        if args.output_dir is None:
            parser.error("--execute requires a fresh --output-dir")
        result = run_validation(**kwargs, output_dir=args.output_dir,
                                preflight_confirmed=args.preflight_confirmed)
    else:
        result = plan_validation(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
