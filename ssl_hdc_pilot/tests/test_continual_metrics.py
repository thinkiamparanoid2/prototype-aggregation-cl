from __future__ import annotations

from ssl_hdc_pilot.scripts.continual_metrics import build_summary


def _row(step: int, truth: str, prediction: str, requested: bool) -> dict[str, object]:
    return {
        "step": step,
        "true_label": truth,
        "prediction": prediction,
        "correct": truth == prediction,
        "top_probability": 0.8,
        "label_requested": requested,
        "update_wall_time_seconds": 0.01 if requested else 0.0,
    }


def test_forgetting_uptake_compute_and_budget_are_reported_together() -> None:
    history = [
        _row(0, "known_a", "known_a", False),
        _row(1, "known_b", "known_b", False),
        _row(2, "novel_c", "known_a", True),
        _row(3, "known_a", "known_b", True),
        _row(4, "novel_c", "novel_c", False),
        _row(5, "known_b", "known_a", False),
        _row(6, "novel_c", "novel_c", False),
    ]

    summary = build_summary(
        history,
        final_evaluation=[
            _row(0, "known_a", "known_a", False),
            _row(1, "known_b", "known_b", False),
            _row(2, "novel_c", "novel_c", False),
        ],
        initially_known_classes=["known_a", "known_b"],
        novel_classes=["novel_c"],
        rolling_window=2,
        stabilization_window=2,
        stabilization_min_accuracy=1.0,
        stabilization_consecutive_windows=1,
        ece_bins=5,
        bootstrap_update_seconds=[0.02, 0.02],
        bootstrap_labels_charged=2,
        label_budget_max=6,
    )

    assert summary["forgetting"]["peak_accuracy"] == 1.0
    assert summary["forgetting"]["final_accuracy"] == 0.5
    assert summary["forgetting"]["forgetting_from_peak"] == 0.5
    uptake = summary["new_class_uptake"]["classes"]["novel_c"]
    assert uptake["examples_until_stabilization"] == 3
    assert uptake["stream_step_at_stabilization"] == 6
    assert summary["update_compute_seconds"]["update_count"] == 4
    assert summary["label_budget"]["total_charged"] == 4
    assert summary["prequential_metrics"]["accuracy"] < 1.0
    assert summary["final_metrics"]["accuracy"] == 1.0
    assert summary["final_evaluation_samples"] == 3
    assert "macro_f1" in summary["final_metrics"]
    assert "macro_recall" in summary["final_metrics"]
