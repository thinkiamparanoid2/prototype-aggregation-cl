"""Post-hoc, read-only novel-class uptake metrics for saved grid artifacts.

This intentionally does not import or alter evaluate_snapshots.py and never
loads a learner.  The original validation-snapshot uptake remains intact.  The
new window metrics use the authenticated chronological training history because
only that ledger records one prediction per encountered class example; snapshot
prediction files are read and pinned as the accompanying fixed-validation
evidence, not re-scored.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


NOVEL_CLASSES = ("3_a9_2", "5_a7_2", "3_a9_3", "6_a11_1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _windows(values: list[bool], width: int) -> Iterable[tuple[int, float]]:
    for end in range(width, len(values) + 1):
        yield end, sum(values[end - width:end]) / width


def _first_consecutive(values: list[bool], *, width: int, target: float, consecutive: int) -> int | None:
    run = 0
    for end, accuracy in _windows(values, width):
        run = run + 1 if accuracy >= target else 0
        if run >= consecutive:
            return end
    return None


def class_uptake(rows: list[dict[str, str]], class_label: str) -> dict[str, Any]:
    hits = [row for row in rows if row["true_label"] == class_label]
    correct = [row["prediction"] == class_label for row in hits]
    available = len(correct)
    original = _first_consecutive(correct, width=20, target=.8, consecutive=3)
    repaired = _first_consecutive(correct, width=10, target=.8, consecutive=3)
    terminal_count = min(20, available)
    terminal = None if not terminal_count else sum(correct[-terminal_count:]) / terminal_count
    return {
        "class_label": class_label,
        "N_available": available,
        "original_three_x_20_at_080": {"examples_to_threshold": original,
            "status": "reached" if original is not None else "not_reached",
            "reachable_by_support": available >= 22},
        "repaired_three_x_10_at_080": {"examples_to_threshold": repaired,
            "status": "reached" if repaired is not None else "not_reached",
            "reachable_by_support": available >= 12},
        "terminal_accuracy": {"window_examples": terminal_count, "accuracy": terminal},
        "examples_to_threshold": repaired,
    }


def _read_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) < {"step", "true_label", "prediction"}:
        raise ValueError("history lacks the authenticated chronological predictions")
    if [int(row["step"]) for row in rows] != list(range(len(rows))):
        raise ValueError("history steps are not a complete zero-based stream")
    return rows


def _read_prediction_receipt(evaluation_dir: Path) -> dict[str, str]:
    report = json.loads((evaluation_dir / "integrity_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise ValueError("evaluation directory lacks a PASS receipt")
    files = sorted((evaluation_dir / "predictions").glob("step_*.json"))
    if not files:
        raise ValueError("evaluation directory has no saved prediction ledgers")
    pins = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("rows"), list) or not payload["rows"]:
            raise ValueError(f"invalid saved prediction ledger: {path.name}")
        pins[path.relative_to(evaluation_dir).as_posix()] = _sha256(path)
    return pins


def repair_uptake(*, source_dir: Path, evaluation_dir: Path, train_csv: Path) -> dict[str, Any]:
    """Recompute support-aware uptake without prediction or model execution."""
    history = _read_history(source_dir / "history.csv")
    with train_csv.open(newline="", encoding="utf-8") as handle:
        train_rows = list(csv.DictReader(handle))
    if not train_rows or "label" not in train_rows[0]:
        raise ValueError("trusted train CSV lacks label column")
    pool_counts = Counter(row["label"] for row in train_rows)
    result = {label: class_uptake(history, label) for label in NOVEL_CLASSES}
    for label, record in result.items():
        if record["N_available"] != pool_counts[label]:
            raise ValueError(f"stream support differs from train pool for {label}")
    return {
        "schema_version": 1,
        "source": {"history": str(source_dir / "history.csv"), "history_sha256": _sha256(source_dir / "history.csv"),
                   "train_csv": str(train_csv), "train_csv_sha256": _sha256(train_csv),
                   "saved_validation_prediction_ledgers": _read_prediction_receipt(evaluation_dir)},
        "classes": result,
        "original_metric_preserved": "Validation-snapshot uptake in metrics.json is unchanged; this is a post-hoc stream-prediction supplement.",
        "scope": "development-only post-hoc arithmetic; no model loading, state mutation, validation rescoring, or test access",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite uptake-repair output")
    result = repair_uptake(source_dir=args.source_dir, evaluation_dir=args.evaluation_dir, train_csv=args.train_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
