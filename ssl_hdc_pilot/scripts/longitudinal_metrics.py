"""Pure development metrics over supplied ID-aligned prediction ledgers.

No file access, learner execution or protocol defaults. Callers must supply the
approved population, checkpoint grid, purchased-label history and uptake targets.
This implements arithmetic, not authorization to collect new states or score test.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from statistics import mean, median, stdev
from typing import Any

from common.thesis_eval import classification_metrics_from_labels


Ledger = Sequence[Mapping[str, Any]]


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return float(value)


def _classes(values: Sequence[str]) -> list[str]:
    values = list(values)
    if not values or any(type(x) is not str or not x for x in values) or len(values) != len(set(values)):
        raise ValueError("classes must be unique nonempty strings")
    return values


def _aligned(ledgers: Mapping[int, Ledger], classes: Sequence[str]) -> dict[int, list[tuple[str, str, str]]]:
    classes = _classes(classes)
    if not ledgers:
        raise ValueError("at least one prediction ledger is required")
    result = {}
    reference = None
    for step, rows in ledgers.items():
        _integer(step, "processed-event count")
        population = {}
        for row in rows:
            image_id, truth, predicted = row["image_id"], row["true_label"], row["predicted_label"]
            if type(image_id) is not str or not image_id or image_id in population:
                raise ValueError("invalid or duplicate prediction ID")
            if truth not in classes or predicted not in classes:
                raise ValueError("prediction ledger label outside fixed class universe")
            population[image_id] = (truth, predicted)
        identity = {image_id: value[0] for image_id, value in population.items()}
        if set(identity.values()) != set(classes):
            raise ValueError("every fixed class must have evaluation support")
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("evaluation IDs or true labels changed between checkpoints")
        result[step] = [(image_id, *population[image_id]) for image_id in sorted(population)]
    return result


def summarize_forgetting(ledgers: Mapping[int, Ledger], *, classes: Sequence[str],
                         old_classes: Sequence[str], checkpoint_steps: Sequence[int]) -> dict[str, Any]:
    """Peak-minus-final recall on ONLY the explicit retention checkpoint grid."""
    classes, old = _classes(classes), _classes(old_classes)
    steps = list(checkpoint_steps)
    if (len(steps) < 2 or any(type(x) is not int or x < 0 for x in steps)
            or steps != sorted(set(steps)) or steps[0] != 0):
        raise ValueError("retention grid must be increasing, unique, and start at bootstrap t=0")
    if not set(old) <= set(classes):
        raise ValueError("old classes must belong to the fixed universe")
    aligned = _aligned(ledgers, classes)
    if any(step not in aligned for step in steps):
        raise ValueError("missing required retention checkpoint")
    if max(aligned) > steps[-1]:
        raise ValueError("retention final step precedes supplied observations")
    curve = []
    for step in steps:
        subset = [(truth, predicted) for _, truth, predicted in aligned[step] if truth in old]
        metrics = classification_metrics_from_labels(
            [r[0] for r in subset], [r[1] for r in subset], classes=old)
        curve.append({"processed_events": step, "samples": len(subset), **metrics})
    per_class = {}
    for label in old:
        recalls = [point["per_class_recall"][label] for point in curve]
        per_class[label] = {"initial_recall": recalls[0], "peak_recall": max(recalls),
                            "final_recall": recalls[-1], "forgetting": max(recalls) - recalls[-1]}
    return {"old_classes": old, "checkpoint_steps": steps, "curve": curve, "per_class": per_class,
            "mean_forgetting": mean(record["forgetting"] for record in per_class.values()),
            "scope": "fixed old-class population; peak on declared retention grid only"}


def summarize_uptake(ledgers: Mapping[int, Ledger], *, classes: Sequence[str], class_label: str,
                     purchased_label_steps: Sequence[int], checkpoint_counts: Sequence[int],
                     final_step: int, relative_fraction: float, absolute_target: float) -> dict[str, Any]:
    """First observed crossing at purchased-label milestones plus final state.

    Purchased steps are processed-event counts immediately after each purchase
    for this novel class (convert zero-based history event i to i+1).
    """
    classes = _classes(classes)
    if class_label not in classes:
        raise ValueError("novel class is outside the fixed universe")
    _integer(final_step, "final step")
    relative_fraction = _unit(relative_fraction, "relative fraction")
    absolute_target = _unit(absolute_target, "absolute target")
    if relative_fraction == 0 or absolute_target == 0:
        raise ValueError("uptake targets must be positive")
    purchases, counts = list(purchased_label_steps), list(checkpoint_counts)
    for values, name in ((purchases, "purchase steps"), (counts, "checkpoint counts")):
        if any(type(x) is not int or x <= 0 for x in values) or values != sorted(set(values)):
            raise ValueError(f"{name} must be strictly increasing positive integers")
    if not counts or any(step > final_step for step in purchases):
        raise ValueError("empty uptake grid or purchases beyond final step")
    aligned = _aligned(ledgers, classes)
    if final_step not in aligned or max(aligned) > final_step:
        raise ValueError("missing or inconsistent final uptake checkpoint")
    for step, rows in aligned.items():
        if (not purchases or step < purchases[0]) and any(predicted == class_label for _, _, predicted in rows):
            raise ValueError("unpurchased novel class cannot be predicted under paid-registration policy")
    selected = [(purchases[count - 1], count) for count in counts if count <= len(purchases)]
    if not selected or selected[-1][0] != final_step:
        selected.append((final_step, len(purchases)))
    if any(step not in aligned for step, _ in selected):
        raise ValueError("missing required purchased-label checkpoint; cannot infer crossing")
    curve = []
    for step, count in selected:
        subset = [(truth, predicted) for _, truth, predicted in aligned[step] if truth == class_label]
        recall = sum(truth == predicted for truth, predicted in subset) / len(subset)
        curve.append({"processed_events": step, "labels_consumed_for_class": count,
                      "recall": recall, "is_final": step == final_step})
    final_recall = curve[-1]["recall"]

    def crossing(target: float | None) -> dict[str, Any]:
        if target is None:
            return {"target": None, "status": "undefined_final_recall_zero", "first_observed": None,
                    "later_below_target": None}
        hits = [i for i, point in enumerate(curve) if point["recall"] >= target]
        if not hits:
            return {"target": target, "status": "never_reached", "first_observed": None,
                    "later_below_target": None}
        first = hits[0]
        return {"target": target, "status": "reached", "first_observed": curve[first],
                "later_below_target": any(point["recall"] < target for point in curve[first + 1:])}

    return {"class_label": class_label, "final_recall": final_recall,
            "registration_status": "registered" if purchases else "never_registered",
            "checkpoint_counts": counts, "unavailable_counts": [x for x in counts if x > len(purchases)],
            "curve": curve, "relative": crossing(relative_fraction * final_recall if final_recall > 0 else None),
            "absolute": crossing(absolute_target),
            "scope": "first observed crossing, not exact crossing time or sustained stabilization"}


def normalized_aulc(scores: Mapping[int, float], *, budget_grid: Sequence[int]) -> float:
    """Trapezoidal area over an explicit complete budget grid, normalized by width."""
    grid = list(budget_grid)
    if (len(grid) < 2 or any(type(x) is not int or x < 0 for x in grid)
            or grid != sorted(set(grid)) or grid[0] != 0 or set(scores) != set(grid)):
        raise ValueError("AULC requires the complete increasing budget grid starting at zero")
    for budget, score in scores.items():
        _integer(budget, "budget")
        _unit(score, "score")
    return sum((b - a) * (scores[a] + scores[b]) / 2 for a, b in zip(grid, grid[1:])) / grid[-1]


def paired_seed_summary(left: Mapping[int, float], right: Mapping[int, float], *,
                        expected_seeds: Sequence[int], scale: float = 1.0) -> dict[str, Any]:
    """Explicit left-minus-right descriptive contrasts; no inferential CI or p-value."""
    seeds = list(expected_seeds)
    if (not seeds or len(seeds) != len(set(seeds)) or any(type(x) is not int or x < 0 for x in seeds)
            or set(left) != set(seeds) or set(right) != set(seeds)):
        raise ValueError("paired summaries require exactly the declared seed set")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("contrast scale must be positive and finite")
    for mapping in (left, right):
        for seed, value in mapping.items():
            _integer(seed, "seed")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("paired values must be finite")
    contrasts = {seed: scale * (left[seed] - right[seed]) for seed in seeds}
    if any(not math.isfinite(x) for x in contrasts.values()):
        raise ValueError("nonfinite paired contrast")
    values = list(contrasts.values())
    return {"direction": "left_minus_right", "scale": scale, "per_seed": contrasts,
            "mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else None,
            "median": median(values), "n_streams": len(values),
            "scope": "descriptive paired stream values, not independent dataset replications"}
