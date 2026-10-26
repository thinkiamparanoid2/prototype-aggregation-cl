"""Offline fidelity check: does the bounded-memory streaming-median control
reproduce the exact median-NCM validation result on the saved mechanism
ablation cells?

Background
----------
``MedianNCMPrototypeLearner`` (``ssl_hdc_pilot/scripts/aggregation_prototypes.py``)
computes an *exact* coordinate-wise median class prototype by storing every
confirmed embedding. ``StreamingMedianNCMPrototypeLearner``
(``ssl_hdc_pilot/scripts/streaming_median_prototypes.py``) is a bounded-memory
P^2 control with the same public interface. Neither learner is modified here;
both are imported and exercised exactly as written.

This script rebuilds per-class prototypes for the 6 saved ``median_ncm``
cells on the adapted arms (directory names containing
``_p2_b_adapted_median_ncm_`` or ``_p2_l_adapted_median_ncm_`` under
``ssl_hdc_pilot/results/mechanism_ablation/capture/``) directly from already
-saved capture artifacts:

- the bootstrap examples, in the order recorded in the capture
  ``manifest.json``'s ``bootstrap_image_ids`` (labels come from
  ``ssl_hdc_pilot/trusted_splits/train.csv``), then
- the purchased ("confirmed") stream events, in ``label_events.json`` row
  step order,

with embeddings drawn from the arm's ``.npz`` named in the capture
manifest's ``inputs``. This mirrors the order ``stream_runner.py``'s
training loop uses to call ``learner.update()`` (bootstrap loop unconditional
update per sample, then per-stream-step update only when
``label_requested``/``purchased`` is true) -- see ``stream_runner.py`` around
lines 795-880, which this script reads for reference but never imports.

Sanity gate
-----------
Before any streaming-vs-exact comparison is reported, the exact rebuild's
macro-F1 on ``trusted_splits/val.csv`` must reproduce the saved
``snapshots[-1]["metrics"]["macro_f1"]`` in
``ssl_hdc_pilot/results/mechanism_ablation/validation/<same cell name>/metrics.json``
to within 1e-6, for every one of the 6 cells. If any cell fails this gate,
``run()`` raises before writing anything: a mismatch means the confirmation
-order/label-source rebuild is wrong, and any streaming-vs-exact comparison
built on it would be meaningless.

This script never launches a capture, batch run, training, or GPU work; it is
strictly read-only and post-hoc. It does not import ``evaluate_snapshots.py``,
``stream_runner.py``, ``budget_experiment.py``, or ``experiment_a.py``, and it
never opens ``ssl_hdc_pilot/trusted_splits/test.csv`` (every file read goes
through the guarded helpers below). Model: ``aggregate_grid.py`` and
``uptake_repair.py``, which establish the same read-only pattern for the same
artifact set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score

from common.thesis_eval import classification_metrics_from_labels
from ssl_hdc_pilot.scripts.aggregation_prototypes import MedianNCMPrototypeLearner
from ssl_hdc_pilot.scripts.streaming_median_prototypes import (
    StreamingMedianNCMPrototypeLearner,
)


DEFAULT_CAPTURE_ROOT = Path("ssl_hdc_pilot/results/mechanism_ablation/capture")
DEFAULT_VALIDATION_ROOT = Path("ssl_hdc_pilot/results/mechanism_ablation/validation")
DEFAULT_TRAIN_CSV = Path("ssl_hdc_pilot/trusted_splits/train.csv")
DEFAULT_VAL_CSV = Path("ssl_hdc_pilot/trusted_splits/val.csv")
DEFAULT_EMBEDDINGS_ROOT = Path("ssl_hdc_pilot/results")

# Directory-name markers for the 6 saved adapted-arm median_ncm cells.
CELL_MARKERS = ("_p2_b_adapted_median_ncm_", "_p2_l_adapted_median_ncm_")
EXPECTED_CELL_COUNT = 6

FOCUS_CLASSES = ("3_00_0", "2_00_0")
SANITY_ABS_TOLERANCE = 1e-6

# Hard prohibition: never read the trusted test split, from this script or any
# path it is handed.
FORBIDDEN_TEST_SPLIT_PARTS = ("trusted_splits", "test.csv")


class TrustedTestSplitAccessError(RuntimeError):
    """Raised if anything in this module is asked to open the trusted test split."""


def _guard_no_test_split(path: Path) -> None:
    path = Path(path)
    parts = path.parts
    if path.name == "test.csv" or any(
        parts[i:i + 2] == FORBIDDEN_TEST_SPLIT_PARTS for i in range(len(parts) - 1)
    ):
        raise TrustedTestSplitAccessError(f"refusing to open trusted test split: {path}")


def _read_json(path: Path) -> Any:
    path = Path(path)
    _guard_no_test_split(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    _guard_no_test_split(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256_file(path: Path) -> str:
    path = Path(path)
    _guard_no_test_split(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_embeddings_npz(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    _guard_no_test_split(path)
    with np.load(path, allow_pickle=False) as archive:
        image_ids = archive["image_ids"]
        embeddings = archive["embeddings"]
    if embeddings.dtype != np.float32 or embeddings.ndim != 2 or len(image_ids) != len(embeddings):
        raise ValueError(f"unexpected embeddings archive layout: {path}")
    return {str(image_id): embeddings[i] for i, image_id in enumerate(image_ids)}


def discover_cells(capture_root: Path) -> list[Path]:
    if not capture_root.is_dir():
        return []
    return sorted(
        p for p in capture_root.iterdir()
        if p.is_dir() and any(marker in p.name for marker in CELL_MARKERS)
    )


def _resolve_recorded_input(record: dict[str, Any], *, root: Path) -> Path:
    """Resolve a manifest-recorded input by filename under `root`, verified by
    its recorded sha256. The manifest's own absolute path is historical
    provenance for the machine that produced it, not a live dependency path
    (see AGENTS.md); this script never trusts it directly.
    """
    name = Path(record["path"]).name
    candidate = root / name
    _guard_no_test_split(candidate)
    if not candidate.is_file():
        raise FileNotFoundError(f"expected recorded input not found: {candidate}")
    digest = _sha256_file(candidate)
    if digest != record["sha256"]:
        raise ValueError(
            f"sha256 mismatch resolving recorded input {candidate}: "
            f"recorded {record['sha256']} != actual {digest}")
    return candidate


def _confirmation_sequence(
    *, bootstrap_image_ids: list[str], label_events_rows: list[dict[str, Any]],
    train_labels: dict[str, str],
) -> list[tuple[str, str]]:
    """(image_id, label) pairs in the exact order ``stream_runner.py``'s
    training loop calls ``learner.update()``: every bootstrap sample first
    (in manifest order), then every purchased stream event, in step order.
    """
    sequence: list[tuple[str, str]] = []
    for image_id in bootstrap_image_ids:
        if image_id not in train_labels:
            raise KeyError(f"bootstrap image_id missing from train.csv: {image_id}")
        sequence.append((image_id, train_labels[image_id]))

    purchased = [row for row in label_events_rows if row["purchased"]]
    purchased.sort(key=lambda row: row["step"])
    for row in purchased:
        image_id, label = row["image_id"], row["true_label"]
        recorded = train_labels.get(image_id)
        if recorded is not None and recorded != label:
            raise ValueError(
                f"label_events true_label disagrees with train.csv for {image_id}: "
                f"{label!r} != {recorded!r}")
        sequence.append((image_id, label))
    return sequence


def _state_bytes(learner: Any, label: str) -> int:
    if isinstance(learner, MedianNCMPrototypeLearner):
        vectors = learner.get_vectors(label)
        return sum(v.element_size() * v.numel() for v in vectors)
    if isinstance(learner, StreamingMedianNCMPrototypeLearner):
        return learner.state_nbytes(label)
    raise TypeError(f"unsupported learner type: {type(learner)!r}")


def _replay(
    learner: Any, sequence: list[tuple[str, str]], embeddings: dict[str, np.ndarray],
) -> list[float]:
    """Apply every (image_id, label) pair to `learner` in order; return the
    per-call wall-clock duration of each ``update()`` call, seconds.
    """
    durations: list[float] = []
    for image_id, label in sequence:
        vector = torch.from_numpy(embeddings[image_id])
        started = time.perf_counter()
        learner.update(vector, label)
        durations.append(time.perf_counter() - started)
    return durations


def _score(
    learner: Any, val_rows: list[tuple[str, str]], embeddings: dict[str, np.ndarray],
) -> tuple[list[str], list[str], list[float]]:
    """Predict on every validation row; return (true_labels, predicted_labels,
    per-call predict() wall-clock durations, seconds).
    """
    truth: list[str] = []
    predicted: list[str] = []
    durations: list[float] = []
    for image_id, label in val_rows:
        vector = torch.from_numpy(embeddings[image_id])
        started = time.perf_counter()
        prediction = learner.predict(vector)
        durations.append(time.perf_counter() - started)
        truth.append(label)
        predicted.append(prediction.label)
    return truth, predicted, durations


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)[0])


def process_cell(
    *, capture_dir: Path, validation_root: Path, train_csv: Path, val_csv: Path,
    embeddings_root: Path,
) -> dict[str, Any]:
    capture_manifest = _read_json(capture_dir / "manifest.json")
    capture_integrity = _read_json(capture_dir / "integrity_report.json")
    if capture_integrity.get("status") != "PASS":
        raise ValueError(f"{capture_dir}: capture integrity_report status is not PASS")
    if capture_manifest.get("learner") != "median_ncm":
        raise ValueError(f"{capture_dir}: manifest learner is not median_ncm")
    if "adapted" not in str(capture_manifest.get("ablation_arm")):
        raise ValueError(f"{capture_dir}: manifest ablation_arm is not an adapted arm")

    validation_dir = validation_root / capture_dir.name
    if not validation_dir.is_dir():
        raise ValueError(f"{capture_dir}: no matching validation directory at {validation_dir}")
    validation_manifest = _read_json(validation_dir / "manifest.json")
    validation_integrity = _read_json(validation_dir / "integrity_report.json")
    if validation_integrity.get("status") != "PASS":
        raise ValueError(f"{validation_dir}: validation integrity_report status is not PASS")
    metrics = _read_json(validation_dir / "metrics.json")
    if metrics.get("data_role") != "trusted_val":
        raise ValueError(f"{validation_dir}: metrics.json data_role is not trusted_val")
    snapshot = metrics["snapshots"][-1]
    saved_macro_f1 = float(snapshot["metrics"]["macro_f1"])
    classes = list(snapshot["classes"])

    embeddings_path = _resolve_recorded_input(
        capture_manifest["inputs"]["embeddings"], root=embeddings_root)
    validation_embeddings_record = validation_manifest["inputs"]["embeddings"]
    if validation_embeddings_record["sha256"] != capture_manifest["inputs"]["embeddings"]["sha256"]:
        raise ValueError(f"{capture_dir}: capture and validation manifests disagree on embeddings input")
    train_path = _resolve_recorded_input(capture_manifest["inputs"]["train"], root=train_csv.parent)
    if train_path.name != train_csv.name:
        raise ValueError(f"{capture_dir}: recorded train input is not {train_csv.name}: {train_path}")
    val_path = _resolve_recorded_input(validation_manifest["inputs"]["validation"], root=val_csv.parent)
    if val_path.name != val_csv.name:
        raise ValueError(f"{validation_dir}: recorded validation input is not {val_csv.name}: {val_path}")

    embeddings = _load_embeddings_npz(embeddings_path)
    input_dim = next(iter(embeddings.values())).shape[0]

    train_rows = _read_csv_rows(train_path)
    train_labels = {row["image"]: row["label"] for row in train_rows}
    derived_classes = sorted(set(train_labels.values()))
    if derived_classes != classes:
        raise ValueError(
            f"{capture_dir}: class universe derived from train.csv disagrees with saved snapshot classes")

    label_events = _read_json(capture_dir / "label_events.json")
    rows = label_events["rows"]
    purchased_count = sum(1 for row in rows if row["purchased"])
    if purchased_count != label_events["accounting"]["labels_consumed"]:
        raise ValueError(f"{capture_dir}: purchased row count disagrees with accounting.labels_consumed")

    bootstrap_image_ids = capture_manifest["bootstrap_image_ids"]
    if set(bootstrap_image_ids) & {row["image_id"] for row in rows}:
        raise ValueError(f"{capture_dir}: bootstrap and stream image ids overlap")

    sequence = _confirmation_sequence(
        bootstrap_image_ids=bootstrap_image_ids, label_events_rows=rows, train_labels=train_labels)
    missing = [image_id for image_id, _ in sequence if image_id not in embeddings]
    if missing:
        raise KeyError(f"{capture_dir}: {len(missing)} confirmation-order image ids missing an embedding")

    val_rows_raw = _read_csv_rows(val_path)
    val_rows = [(row["image"], row["label"]) for row in val_rows_raw]
    if not set(label for _, label in val_rows) <= set(classes):
        raise ValueError(f"{validation_dir}: val.csv contains a label outside the saved class universe")
    missing_val = [image_id for image_id, _ in val_rows if image_id not in embeddings]
    if missing_val:
        raise KeyError(f"{validation_dir}: {len(missing_val)} validation image ids missing an embedding")

    exact = MedianNCMPrototypeLearner(input_dim=input_dim, device="cpu")
    streaming = StreamingMedianNCMPrototypeLearner(input_dim=input_dim, device="cpu")

    exact_update_seconds = _replay(exact, sequence, embeddings)
    streaming_update_seconds = _replay(streaming, sequence, embeddings)

    exact_truth, exact_pred, exact_predict_seconds = _score(exact, val_rows, embeddings)
    streaming_truth, streaming_pred, streaming_predict_seconds = _score(streaming, val_rows, embeddings)
    if exact_truth != streaming_truth:
        raise AssertionError("internal error: exact/streaming truth vectors diverged")

    exact_metrics = classification_metrics_from_labels(exact_truth, exact_pred, classes=classes)
    streaming_metrics = classification_metrics_from_labels(streaming_truth, streaming_pred, classes=classes)

    sanity_diff = abs(exact_metrics["macro_f1"] - saved_macro_f1)
    sanity_passed = sanity_diff <= SANITY_ABS_TOLERANCE

    exact_per_class_f1 = dict(zip(
        classes, f1_score(exact_truth, exact_pred, labels=classes, average=None, zero_division=0)))
    streaming_per_class_f1 = dict(zip(
        classes, f1_score(streaming_truth, streaming_pred, labels=classes, average=None, zero_division=0)))

    if set(exact.known_classes) != set(streaming.known_classes):
        raise AssertionError("internal error: exact/streaming registered class sets diverged")
    common_classes = sorted(exact.known_classes)
    prototype_cosines = {
        label: _cosine(exact.get_prototype(label), streaming.get_prototype(label))
        for label in common_classes
    }

    agreement_rate = float(sum(1 for a, b in zip(exact_pred, streaming_pred) if a == b) / len(exact_pred))

    per_class_state_bytes = {
        label: {"exact_bytes": _state_bytes(exact, label), "streaming_bytes": _state_bytes(streaming, label),
                "exact_confirmed_examples": len(exact.get_vectors(label))}
        for label in common_classes
    }
    exact_total_bytes = sum(v["exact_bytes"] for v in per_class_state_bytes.values())
    streaming_total_bytes = sum(v["streaming_bytes"] for v in per_class_state_bytes.values())

    return {
        "cell": capture_dir.name,
        "ablation_arm": capture_manifest["ablation_arm"],
        "stream_seed": capture_manifest["stream_seed"],
        "input_dim": input_dim,
        "classes": classes,
        "confirmed_examples": len(sequence),
        "bootstrap_examples": len(bootstrap_image_ids),
        "purchased_examples": purchased_count,
        "validation_samples": len(val_rows),
        "sanity_check": {
            "saved_macro_f1": saved_macro_f1,
            "rebuilt_exact_macro_f1": exact_metrics["macro_f1"],
            "abs_diff": sanity_diff,
            "tolerance": SANITY_ABS_TOLERANCE,
            "passed": sanity_passed,
        },
        "comparison": {
            "macro_f1": {"exact": exact_metrics["macro_f1"], "streaming": streaming_metrics["macro_f1"]},
            "accuracy": {"exact": exact_metrics["accuracy"], "streaming": streaming_metrics["accuracy"]},
            "per_class_f1_focus": {
                cls: {"exact": exact_per_class_f1[cls], "streaming": streaming_per_class_f1[cls]}
                for cls in FOCUS_CLASSES if cls in exact_per_class_f1
            },
            "per_class_f1_all": {
                cls: {"exact": exact_per_class_f1[cls], "streaming": streaming_per_class_f1[cls]}
                for cls in classes
            },
            "prediction_agreement_rate": agreement_rate,
            "prototype_cosine": {
                "mean": statistics.mean(prototype_cosines.values()) if prototype_cosines else None,
                "per_class": prototype_cosines,
            },
            "state_bytes": {
                "exact_total": exact_total_bytes,
                "streaming_total": streaming_total_bytes,
                "per_class": per_class_state_bytes,
            },
            "timing_seconds_indicative_only": {
                "note": "single desktop, not a benchmark; predict() timed over the "
                        "validation scoring pass (this script's only predict calls), "
                        "not the confirmation replay itself.",
                "update_time_median": {
                    "exact": statistics.median(exact_update_seconds),
                    "streaming": statistics.median(streaming_update_seconds),
                },
                "predict_time_median": {
                    "exact": statistics.median(exact_predict_seconds),
                    "streaming": statistics.median(streaming_predict_seconds),
                },
                "update_calls": len(sequence),
                "predict_calls": len(val_rows),
            },
        },
    }


def plan(*, capture_root: Path, validation_root: Path, output_dir: Path | None) -> dict[str, Any]:
    cell_dirs = discover_cells(capture_root)
    return {
        "mode": "plan_only",
        "would_read": {
            "capture_root": str(capture_root),
            "validation_root": str(validation_root),
            "cell_directories": [d.name for d in cell_dirs],
            "cell_count": len(cell_dirs),
        },
        "would_write": {
            "output_dir": str(output_dir) if output_dir else None,
            "files": ["cells.json", "cells.csv", "state_bytes.csv", "summary.json", "PROVENANCE.md"],
        },
        "note": "no files were read beyond directory listing, and nothing was written; pass --execute to run",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def build_cells_table(cells: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    header = [
        "cell", "ablation_arm", "stream_seed", "input_dim",
        "sanity_saved_macro_f1", "sanity_rebuilt_macro_f1", "sanity_abs_diff", "sanity_passed",
        "macro_f1_exact", "macro_f1_streaming",
        "f1_3_00_0_exact", "f1_3_00_0_streaming", "f1_2_00_0_exact", "f1_2_00_0_streaming",
        "prediction_agreement_rate", "prototype_cosine_mean",
        "state_bytes_exact_total", "state_bytes_streaming_total",
        "update_time_median_exact_s", "update_time_median_streaming_s",
        "predict_time_median_exact_s", "predict_time_median_streaming_s",
    ]
    rows = []
    for cell in cells:
        comparison = cell["comparison"]
        focus = comparison["per_class_f1_focus"]
        timing = comparison["timing_seconds_indicative_only"]
        rows.append([
            cell["cell"], cell["ablation_arm"], cell["stream_seed"], cell["input_dim"],
            cell["sanity_check"]["saved_macro_f1"], cell["sanity_check"]["rebuilt_exact_macro_f1"],
            cell["sanity_check"]["abs_diff"], cell["sanity_check"]["passed"],
            comparison["macro_f1"]["exact"], comparison["macro_f1"]["streaming"],
            focus.get("3_00_0", {}).get("exact"), focus.get("3_00_0", {}).get("streaming"),
            focus.get("2_00_0", {}).get("exact"), focus.get("2_00_0", {}).get("streaming"),
            comparison["prediction_agreement_rate"], comparison["prototype_cosine"]["mean"],
            comparison["state_bytes"]["exact_total"], comparison["state_bytes"]["streaming_total"],
            timing["update_time_median"]["exact"], timing["update_time_median"]["streaming"],
            timing["predict_time_median"]["exact"], timing["predict_time_median"]["streaming"],
        ])
    return header, rows


def build_state_bytes_table(cells: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    header = ["cell", "class", "exact_confirmed_examples", "exact_bytes", "streaming_bytes"]
    rows = []
    for cell in cells:
        per_class = cell["comparison"]["state_bytes"]["per_class"]
        for cls in sorted(per_class):
            record = per_class[cls]
            rows.append([cell["cell"], cls, record["exact_confirmed_examples"],
                        record["exact_bytes"], record["streaming_bytes"]])
    return header, rows


def build_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    all_passed = all(cell["sanity_check"]["passed"] for cell in cells)
    macro_f1_diffs = [
        cell["comparison"]["macro_f1"]["streaming"] - cell["comparison"]["macro_f1"]["exact"]
        for cell in cells
    ]
    agreement_rates = [cell["comparison"]["prediction_agreement_rate"] for cell in cells]
    cosine_means = [cell["comparison"]["prototype_cosine"]["mean"] for cell in cells if cell["comparison"]["prototype_cosine"]["mean"] is not None]
    byte_ratios = [
        cell["comparison"]["state_bytes"]["streaming_total"] / cell["comparison"]["state_bytes"]["exact_total"]
        for cell in cells if cell["comparison"]["state_bytes"]["exact_total"]
    ]
    return {
        "cell_count": len(cells),
        "all_sanity_checks_passed": all_passed,
        "per_cell_sanity": [
            {"cell": cell["cell"], "passed": cell["sanity_check"]["passed"],
             "abs_diff": cell["sanity_check"]["abs_diff"]}
            for cell in cells
        ],
        "macro_f1_streaming_minus_exact": {
            "median": statistics.median(macro_f1_diffs) if macro_f1_diffs else None,
            "min": min(macro_f1_diffs) if macro_f1_diffs else None,
            "max": max(macro_f1_diffs) if macro_f1_diffs else None,
        },
        "prediction_agreement_rate": {
            "median": statistics.median(agreement_rates) if agreement_rates else None,
            "min": min(agreement_rates) if agreement_rates else None,
        },
        "prototype_cosine_mean": {
            "median": statistics.median(cosine_means) if cosine_means else None,
            "min": min(cosine_means) if cosine_means else None,
        },
        "streaming_bytes_over_exact_bytes_ratio": {
            "median": statistics.median(byte_ratios) if byte_ratios else None,
            "max": max(byte_ratios) if byte_ratios else None,
        },
    }


def build_provenance(*, capture_root: Path, validation_root: Path, cells: list[dict[str, Any]]) -> str:
    lines = [
        "# streaming_median_fidelity.py provenance",
        "",
        "Read-only, post-hoc offline fidelity check. No capture, evaluation,",
        "training, or GPU work was launched by this script; it never imports",
        "evaluate_snapshots.py, stream_runner.py, budget_experiment.py, or",
        "experiment_a.py, and it never opens ssl_hdc_pilot/trusted_splits/test.csv.",
        "",
        f"- capture root: `{capture_root}`",
        f"- validation root: `{validation_root}`",
        f"- cells checked: {len(cells)}",
        "",
        "## Sanity gate",
        "",
        "For each cell, the exact-median rebuild's macro-F1 on trusted_splits/val.csv",
        "must reproduce the saved snapshots[-1][\"metrics\"][\"macro_f1\"] to within 1e-6.",
        "",
    ]
    for cell in cells:
        sanity = cell["sanity_check"]
        lines.append(
            f"- `{cell['cell']}`: saved={sanity['saved_macro_f1']:.9f} "
            f"rebuilt={sanity['rebuilt_exact_macro_f1']:.9f} "
            f"abs_diff={sanity['abs_diff']:.3e} passed={sanity['passed']}"
        )
    lines.append("")
    return "\n".join(lines)


def run(
    *, capture_root: Path, validation_root: Path, train_csv: Path, val_csv: Path,
    embeddings_root: Path, output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    cell_dirs = discover_cells(capture_root)
    if len(cell_dirs) != EXPECTED_CELL_COUNT:
        raise ValueError(
            f"expected exactly {EXPECTED_CELL_COUNT} adapted median_ncm cells under {capture_root}, "
            f"found {len(cell_dirs)}: {[d.name for d in cell_dirs]}")

    cells = [
        process_cell(
            capture_dir=cell_dir, validation_root=validation_root, train_csv=train_csv,
            val_csv=val_csv, embeddings_root=embeddings_root)
        for cell_dir in cell_dirs
    ]

    failing = [cell for cell in cells if not cell["sanity_check"]["passed"]]
    if failing:
        details = "; ".join(
            f"{cell['cell']}: saved={cell['sanity_check']['saved_macro_f1']!r} "
            f"rebuilt={cell['sanity_check']['rebuilt_exact_macro_f1']!r} "
            f"abs_diff={cell['sanity_check']['abs_diff']!r}"
            for cell in failing
        )
        raise RuntimeError(
            f"sanity check failed for {len(failing)}/{len(cells)} cells; refusing to write a "
            f"streaming-vs-exact comparison built on an unverified rebuild: {details}")

    cells_header, cells_rows = build_cells_table(cells)
    state_bytes_header, state_bytes_rows = build_state_bytes_table(cells)
    summary = build_summary(cells)
    provenance = build_provenance(capture_root=capture_root, validation_root=validation_root, cells=cells)

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "cells.json", cells)
    _write_csv(output_dir / "cells.csv", cells_header, cells_rows)
    _write_csv(output_dir / "state_bytes.csv", state_bytes_header, state_bytes_rows)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "PROVENANCE.md").write_text(provenance, encoding="utf-8")

    return {
        "mode": "executed",
        "output_dir": str(output_dir),
        "cell_count": len(cells),
        "all_sanity_checks_passed": summary["all_sanity_checks_passed"],
        "files_written": ["cells.json", "cells.csv", "state_bytes.csv", "summary.json", "PROVENANCE.md"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true", help="otherwise print the plan and write nothing")
    args = parser.parse_args()

    if not args.execute:
        print(json.dumps(
            plan(capture_root=args.capture_root, validation_root=args.validation_root,
                 output_dir=args.output_dir),
            indent=2, sort_keys=True))
        return

    if args.output_dir is None:
        parser.error("--execute requires --output-dir")
    report = run(
        capture_root=args.capture_root, validation_root=args.validation_root,
        train_csv=args.train_csv, val_csv=args.val_csv, embeddings_root=args.embeddings_root,
        output_dir=args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
