"""CPU-only replication-cell runner built from the registered learner interfaces.

The module is intentionally narrow: it consumes already-saved embeddings plus
pre-built train/validation splits and stream specifications.  It never opens a
test split, trains an encoder, or overwrites an output directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.thesis_eval import (
    classification_metrics_from_labels,
    expected_calibration_error_from_confidence,
)
from ssl_hdc_pilot.scripts import learner_checkpoint as checkpoints
from ssl_hdc_pilot.scripts import longitudinal_metrics
from ssl_hdc_pilot.scripts import stream_runner as runner


LEARNERS = frozenset({"hdc", "ncm", "ranpac"})


def _sha256(path: Path) -> str:
    return runner._file_sha256(path)


def _reject_test_path(path: Path, *, role: str) -> None:
    """Reject sealed partitions before any existence, hash, or CSV operation."""
    normalized = path.name.casefold()
    if "test" in normalized:
        raise RuntimeError(f"sealed test split rejected for {role}: {path}")


def _require_cpu(spec: runner.StreamSpec) -> None:
    if spec.device != "cpu":
        raise RuntimeError("replication pipeline is CPU-only; stream spec execution.device must be cpu")


def _load_replication_spec(path: Path) -> runner.StreamSpec:
    """Parse a schema-conforming new-dataset spec without canonical-task gates."""
    if not path.is_file():
        raise FileNotFoundError(f"stream spec not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid stream spec JSON: {path}") from error
    spec = runner._parse_stream_spec(path, payload)
    if spec.smoke:
        raise RuntimeError("replication pipeline refuses smoke stream specifications")
    _require_cpu(spec)
    return spec


def _validate_steps(steps: Sequence[int], stream_length: int) -> tuple[int, ...]:
    checked = tuple(steps)
    if (len(checked) < 2 or checked[0] != 0 or checked[-1] != stream_length
            or checked != tuple(sorted(set(checked)))
            or any(type(step) is not int or step < 0 for step in checked)):
        raise ValueError("checkpoint_steps must be increasing, start at 0, and end at stream length")
    return checked


def _fixed_population_predictions(learner: Any, rows: Sequence[tuple[str, str]], embeddings: Mapping[str, Any],
                                  classes: Sequence[str]) -> list[dict[str, Any]]:
    result = []
    allowed = set(classes)
    for image_id, label in rows:
        prediction = learner.predict(embeddings[image_id])
        if prediction.label not in allowed:
            raise RuntimeError("fixed validation prediction is outside the class universe")
        result.append({"image_id": image_id, "true_label": label,
                       "predicted_label": prediction.label,
                       "top_probability": float(prediction.top_probability)})
    return result


def _validation_metrics(rows: Sequence[Mapping[str, Any]], classes: Sequence[str]) -> dict[str, Any]:
    truth = [str(row["true_label"]) for row in rows]
    predicted = [str(row["predicted_label"]) for row in rows]
    metrics = classification_metrics_from_labels(truth, predicted, classes=list(classes))
    metrics["ece"] = expected_calibration_error_from_confidence(
        [float(row["top_probability"]) for row in rows],
        [left == right for left, right in zip(truth, predicted)], n_bins=15,
    )
    return metrics


def plan_cell(*, dataset: str, arm: str, learner: str, seed: int, embeddings_path: Path,
              train_csv: Path, validation_csv: Path, stream_spec_path: Path,
              checkpoint_steps: Sequence[int]) -> dict[str, Any]:
    """Validate the CPU cell inputs without constructing a learner or writing output."""
    if learner not in LEARNERS:
        raise ValueError(f"replication learner must be one of {sorted(LEARNERS)}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    train_csv, validation_csv = Path(train_csv), Path(validation_csv)
    _reject_test_path(train_csv, role="training")
    _reject_test_path(validation_csv, role="validation")
    spec = _load_replication_spec(Path(stream_spec_path))
    if seed not in spec.stream_order_seeds:
        raise ValueError("requested seed is absent from stream spec")
    train_rows = runner._load_split_rows(train_csv, smoke_mode=True)
    validation_rows = runner._load_split_rows(validation_csv, smoke_mode=True)
    train_ids, validation_ids = {row[0] for row in train_rows}, {row[0] for row in validation_rows}
    if train_ids & validation_ids:
        raise ValueError("train and validation splits overlap")
    classes = tuple(sorted({label for _, label in train_rows}))
    if set(label for _, label in validation_rows) != set(classes):
        raise ValueError("validation class universe must equal training class universe")
    embeddings, dimension = runner._load_embedding_map(Path(embeddings_path))
    if any(image_id not in embeddings for image_id in train_ids | validation_ids):
        raise ValueError("embedding archive is missing split image IDs")
    bootstrap, stream = runner._prepare_stream(
        split_rows=train_rows, embedding_map=embeddings, spec=spec, stream_seed=seed,
    )
    steps = _validate_steps(checkpoint_steps, len(stream))
    return {
        "dataset": dataset, "arm": arm, "learner": learner, "seed": seed,
        "device": "cpu", "embedding_dim": dimension, "classes": list(classes),
        "bootstrap_samples": len(bootstrap), "stream_samples": len(stream),
        "checkpoint_steps": list(steps), "label_budget": spec.max_total_requests,
        "inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in {
            "embeddings": Path(embeddings_path), "train": train_csv,
            "validation": validation_csv, "stream_spec": Path(stream_spec_path),
        }.items()},
        "scope": "prepared saved-embedding CPU cell; no test access, extraction, adaptation, or capture",
    }


def run_cell(*, dataset: str, arm: str, learner: str, seed: int, embeddings_path: Path,
             train_csv: Path, validation_csv: Path, stream_spec_path: Path,
             checkpoint_steps: Sequence[int], output_dir: Path,
             preflight_confirmed: bool = False, equivalence_gate: bool = False) -> dict[str, Any]:
    """Run one saved-embedding CPU cell and write immutable capture/evaluation receipts.

    ``equivalence_gate`` is solely for the explicitly authorized Gate 0 replay
    against the already-saved local crop embeddings.  Ordinary replication cells
    require the caller's preflight confirmation.
    """
    if not preflight_confirmed and not equivalence_gate:
        raise RuntimeError("replication execution requires manual user preflight confirmation")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite replication output: {output_dir}")
    plan = plan_cell(dataset=dataset, arm=arm, learner=learner, seed=seed,
                     embeddings_path=embeddings_path, train_csv=train_csv,
                     validation_csv=validation_csv, stream_spec_path=stream_spec_path,
                     checkpoint_steps=checkpoint_steps)
    # Re-load only after all no-write validation completed, then pin those inputs.
    spec = _load_replication_spec(Path(stream_spec_path))
    embeddings, dimension = runner._load_embedding_map(Path(embeddings_path))
    train_rows = runner._load_split_rows(Path(train_csv), smoke_mode=True)
    validation_rows = runner._load_split_rows(Path(validation_csv), smoke_mode=True)
    bootstrap, stream = runner._prepare_stream(split_rows=train_rows, embedding_map=embeddings,
                                                spec=spec, stream_seed=seed)
    steps = tuple(plan["checkpoint_steps"])
    classes = tuple(plan["classes"])
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir()
    ledgers: dict[int, list[dict[str, Any]]] = {}
    snapshot_index: list[dict[str, Any]] = []

    def observe(current: Any, progress: Mapping[str, Any]) -> None:
        processed = int(progress["processed_stream_events"])
        if processed not in steps:
            return
        path = snapshots_dir / f"step_{processed:06d}.npz"
        checkpoints.save_learner_checkpoint(path, current, context={
            "dataset": dataset, "arm": arm, "learner": learner, "seed": seed,
            "processed_stream_events": processed, "stream_spec_sha256": spec.sha256,
        }, shared_projection_root=output_dir)
        rows = _fixed_population_predictions(current, validation_rows, embeddings, classes)
        ledgers[processed] = rows
        prediction_path = snapshots_dir / f"step_{processed:06d}_predictions.json"
        prediction_path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        snapshot_index.append({"processed_stream_events": processed,
                               "checkpoint": path.relative_to(output_dir).as_posix(),
                               "predictions": prediction_path.relative_to(output_dir).as_posix(),
                               "checkpoint_sha256": _sha256(path)})

    history, stream_summary = runner._run_one(learner_name=learner, threshold=spec.confidence_thresholds[0],
        bootstrap=bootstrap, stream=stream, spec=spec, input_dim=dimension, clock=__import__("time").perf_counter,
        on_observation=observe)
    if set(ledgers) != set(steps):
        raise RuntimeError("missing required fixed-population snapshots")
    final_rows = ledgers[steps[-1]]
    final_metrics = _validation_metrics(final_rows, classes)
    forgetting = longitudinal_metrics.summarize_forgetting(
        ledgers, classes=classes, old_classes=spec.initially_known_classes, checkpoint_steps=steps,
    )
    runner._write_history(output_dir / "history.csv", history)
    (output_dir / "snapshot_index.json").write_text(json.dumps({"snapshots": snapshot_index}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"dataset": dataset, "arm": arm, "learner": learner, "seed": seed,
              "data_role": "replication_validation", "final_validation": final_metrics,
              "forgetting": forgetting, "labels_consumed": stream_summary["label_budget"]["stream_label_requests"],
              "label_budget": spec.max_total_requests,
              "label_budget_match": stream_summary["label_budget"]["stream_label_requests"] == spec.max_total_requests,
              "stream_summary": stream_summary,
              "scope": "saved-embedding CPU replication cell; validation development only; no test access"}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "status": "PASS", "plan": plan,
                "code_hashes": {**runner._checkpoint_code_hashes(),
                                "ssl_hdc_pilot/replication/pipeline.py": _sha256(Path(__file__))},
                "equivalence_gate": equivalence_gate,
                "scope": "immutable saved-embedding receipt; no test, extraction, adaptation, or GPU work"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = {path.relative_to(output_dir).as_posix(): _sha256(path)
                 for path in output_dir.rglob("*") if path.is_file()}
    receipt = {"status": "PASS", "artifacts": artifacts, "inputs": plan["inputs"],
               "labels_consumed": result["labels_consumed"], "label_budget": result["label_budget"],
               "label_budget_match": result["label_budget_match"],
               "scope": result["scope"]}
    (output_dir / "integrity_report.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
