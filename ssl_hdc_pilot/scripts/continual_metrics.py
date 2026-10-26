"""Metrics for continual-learning streams; never emit accuracy alone."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from common.thesis_eval import (
    classification_metrics_from_labels,
    expected_calibration_error_from_confidence,
    stage_recall,
)

def rolling_per_class_accuracy(
    history: Sequence[dict[str, Any]], *, window: int
) -> dict[str, float]:
    """Accuracy over each class's most recent ``window`` stream encounters."""
    if window < 1:
        raise ValueError("rolling window must be positive")
    outcomes: dict[str, list[bool]] = defaultdict(list)
    for row in history:
        outcomes[str(row["true_label"])].append(bool(row["correct"]))
    return {
        label: float(np.mean(values[-window:]))
        for label, values in sorted(outcomes.items())
    }


def final_predictive_metrics(
    history: Sequence[dict[str, Any]], *, ece_bins: int
) -> dict[str, Any]:
    if not history:
        raise ValueError("cannot score an empty stream")
    true_labels = [str(row["true_label"]) for row in history]
    predicted_labels = [
        str(row["prediction"]) if row["prediction"] is not None else "__unclassified__"
        for row in history
    ]
    classes = sorted(set(true_labels))
    metrics = classification_metrics_from_labels(
        true_labels,
        predicted_labels,
        classes=classes,
    )
    stage_compatible = all(
        len(label.split("_")) >= 3
        for label in [*true_labels, *predicted_labels]
        if label != "__unclassified__"
    ) and "__unclassified__" not in predicted_labels
    metrics["per_stage_recall"] = (
        stage_recall(true_labels, predicted_labels) if stage_compatible else {}
    )
    metrics["ece"] = expected_calibration_error_from_confidence(
        [float(row["top_probability"]) for row in history],
        [bool(row["correct"]) for row in history],
        n_bins=ece_bins,
    )
    return metrics


def forgetting_metrics(
    history: Sequence[dict[str, Any]],
    *,
    initially_known_classes: Sequence[str],
    rolling_window: int,
) -> dict[str, Any]:
    initial = set(initially_known_classes)
    curve: list[dict[str, Any]] = []
    prefix: list[dict[str, Any]] = []
    for row in history:
        prefix.append(row)
        rolling = rolling_per_class_accuracy(prefix, window=rolling_window)
        values = [rolling[label] for label in initially_known_classes if label in rolling]
        curve.append(
            {
                "step": int(row["step"]),
                "initial_known_macro_accuracy": float(np.mean(values)) if values else None,
            }
        )
    observed = [
        float(point["initial_known_macro_accuracy"])
        for point in curve
        if point["initial_known_macro_accuracy"] is not None
    ]
    peak = max(observed) if observed else None
    final = observed[-1] if observed else None
    return {
        "definition": (
            "At each stream step, macro-average the per-class accuracy over each "
            "initially-known class's most recent rolling_window encounters. "
            "Forgetting is peak minus final rolling macro accuracy."
        ),
        "initially_known_classes": sorted(initial),
        "rolling_window": rolling_window,
        "curve": curve,
        "peak_accuracy": peak,
        "final_accuracy": final,
        "forgetting_from_peak": None if peak is None or final is None else peak - final,
    }


def new_class_uptake_metrics(
    history: Sequence[dict[str, Any]],
    *,
    novel_classes: Sequence[str],
    window: int,
    min_accuracy: float,
    consecutive_windows: int,
) -> dict[str, Any]:
    if window < 1 or consecutive_windows < 1 or not 0 <= min_accuracy <= 1:
        raise ValueError("invalid new-class stabilization settings")
    results: dict[str, Any] = {}
    for label in novel_classes:
        rows = [row for row in history if row["true_label"] == label]
        outcomes = [bool(row["correct"]) for row in rows]
        consecutive = 0
        stable_examples: int | None = None
        stable_step: int | None = None
        for index in range(window - 1, len(outcomes)):
            accuracy = float(np.mean(outcomes[index - window + 1 : index + 1]))
            consecutive = consecutive + 1 if accuracy >= min_accuracy else 0
            if consecutive >= consecutive_windows:
                stable_examples = index + 1
                stable_step = int(rows[index]["step"])
                break
        results[label] = {
            "examples_until_stabilization": stable_examples,
            "stream_step_at_stabilization": stable_step,
            "examples_observed": len(rows),
        }
    return {
        "definition": (
            "Class-specific examples encountered through the first point where "
            f"{consecutive_windows} consecutive sliding window(s) of {window} "
            f"examples reach accuracy >= {min_accuracy:.6g}; null means not stabilized."
        ),
        "classes": results,
    }


def update_compute_metrics(update_seconds: Iterable[float]) -> dict[str, Any]:
    values = np.asarray([float(value) for value in update_seconds], dtype=np.float64)
    if values.size == 0:
        return {
            "update_count": 0,
            "total": 0.0,
            "mean": None,
            "median": None,
            "p95": None,
        }
    return {
        "update_count": int(values.size),
        "total": float(np.sum(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def build_summary(
    history: Sequence[dict[str, Any]],
    *,
    final_evaluation: Sequence[dict[str, Any]] | None = None,
    initially_known_classes: Sequence[str],
    novel_classes: Sequence[str],
    rolling_window: int,
    stabilization_window: int,
    stabilization_min_accuracy: float,
    stabilization_consecutive_windows: int,
    ece_bins: int,
    bootstrap_update_seconds: Sequence[float],
    bootstrap_labels_charged: int,
    label_budget_max: int,
) -> dict[str, Any]:
    stream_update_seconds = [
        float(row["update_wall_time_seconds"])
        for row in history
        if bool(row["label_requested"])
    ]
    stream_requests = sum(bool(row["label_requested"]) for row in history)
    total_charged = bootstrap_labels_charged + stream_requests
    final_rows = history if final_evaluation is None else final_evaluation
    return {
        "prequential_metrics": final_predictive_metrics(history, ece_bins=ece_bins),
        "final_metrics": final_predictive_metrics(final_rows, ece_bins=ece_bins),
        "final_evaluation_samples": len(final_rows),
        "forgetting": forgetting_metrics(
            history,
            initially_known_classes=initially_known_classes,
            rolling_window=rolling_window,
        ),
        "new_class_uptake": new_class_uptake_metrics(
            history,
            novel_classes=novel_classes,
            window=stabilization_window,
            min_accuracy=stabilization_min_accuracy,
            consecutive_windows=stabilization_consecutive_windows,
        ),
        "update_compute_seconds": update_compute_metrics(
            [*bootstrap_update_seconds, *stream_update_seconds]
        ),
        "label_budget": {
            "max_total_requests": label_budget_max,
            "bootstrap_labels_charged": bootstrap_labels_charged,
            "stream_label_requests": stream_requests,
            "total_charged": total_charged,
            "remaining": label_budget_max - total_charged,
            "exhausted": total_charged >= label_budget_max,
        },
    }
