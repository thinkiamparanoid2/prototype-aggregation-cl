"""Hand-checkable synthetic arithmetic; no model runs or dataset access."""
from copy import deepcopy
import math

import pytest

from ssl_hdc_pilot.scripts.longitudinal_metrics import (
    normalized_aulc, paired_seed_summary, summarize_forgetting, summarize_uptake,
)


CLASSES = ["old_a", "old_b", "novel"]


def ledger(predictions):
    return [{"image_id": str(i), "true_label": truth, "predicted_label": predicted}
            for i, (truth, predicted) in enumerate(zip(
                ["old_a", "old_a", "old_b", "old_b", "novel", "novel"], predictions))]


@pytest.fixture
def ledgers():
    return {
        0: ledger(["old_a", "old_b", "old_b", "old_b", "old_a", "old_a"]),
        1: ledger(["old_a", "old_a", "old_b", "old_b", "novel", "novel"]),
        2: ledger(["old_b", "old_b", "old_b", "novel", "novel", "old_a"]),
        4: ledger(["old_a", "novel", "old_a", "novel", "novel", "novel"]),
    }


def forgetting(values):
    return summarize_forgetting(values, classes=CLASSES, old_classes=CLASSES[:2], checkpoint_steps=[0, 2, 4])


def uptake(values, **kwargs):
    return summarize_uptake(values, **{
        "classes": CLASSES, "class_label": "novel", "purchased_label_steps": [1, 2, 3],
        "checkpoint_counts": [1, 2, 5], "final_step": 4,
        "relative_fraction": .8, "absolute_target": .8, **kwargs})


def test_retention_uses_only_declared_grid_and_fixed_old_class_denominator(ledgers):
    before = deepcopy(ledgers)
    result = forgetting(ledgers)
    assert result["mean_forgetting"] == .5
    assert result["per_class"]["old_a"] == {
        "initial_recall": .5, "peak_recall": .5, "final_recall": .5, "forgetting": 0.}
    assert result["per_class"]["old_b"]["forgetting"] == 1.
    assert result["curve"][-1]["accuracy"] == .25
    assert result["curve"][-1]["macro_f1"] == .25
    assert result["curve"][-1]["samples"] == 4
    assert result["checkpoint_steps"] == [0, 2, 4]
    assert ledgers == before


def test_row_reordering_does_not_change_identity_or_scores(ledgers):
    expected = forgetting(ledgers)
    ledgers[2].reverse()
    assert forgetting(ledgers) == expected


@pytest.mark.parametrize("problem", ["id_change", "truth_change", "duplicate", "missing_class", "unknown_prediction"])
def test_rejects_population_drift_and_invalid_ledgers(ledgers, problem):
    if problem == "id_change":
        ledgers[2][0]["image_id"] = "different"
    elif problem == "truth_change":
        ledgers[2][0]["true_label"] = "old_b"
    elif problem == "duplicate":
        ledgers[2].append(ledgers[2][0])
    elif problem == "missing_class":
        ledgers[2] = ledgers[2][:4]
    else:
        ledgers[2][0]["predicted_label"] = "outside"
    with pytest.raises(ValueError):
        forgetting(ledgers)


@pytest.mark.parametrize("steps", [[0, 3, 4], [0, 2], [2, 4], [0, 2, 2, 4], [0, True, 4], [0]])
def test_rejects_invalid_or_missing_retention_grid(ledgers, steps):
    with pytest.raises(ValueError):
        summarize_forgetting(ledgers, classes=CLASSES, old_classes=CLASSES[:2], checkpoint_steps=steps)


def test_uptake_first_observed_crossing_reports_later_drop_and_missing_milestones(ledgers):
    before = deepcopy(ledgers)
    result = uptake(ledgers)
    assert result["registration_status"] == "registered"
    assert result["final_recall"] == 1.
    assert result["relative"]["first_observed"]["labels_consumed_for_class"] == 1
    assert result["absolute"]["first_observed"]["processed_events"] == 1
    assert result["absolute"]["later_below_target"]
    assert result["unavailable_counts"] == [5]
    assert result["curve"][-1] == {
        "processed_events": 4, "labels_consumed_for_class": 3, "recall": 1., "is_final": True}
    assert ledgers == before


def test_zero_final_recall_relative_is_undefined_even_if_earlier_recall_was_positive(ledgers):
    ledgers[4][-1]["predicted_label"] = "old_a"
    ledgers[4][-2]["predicted_label"] = "old_a"
    result = uptake(ledgers)
    assert result["relative"]["status"] == "undefined_final_recall_zero"
    assert result["relative"]["first_observed"] is None
    assert result["absolute"]["status"] == "reached"
    assert result["absolute"]["later_below_target"]


def test_never_registered_is_explicit_not_zero_label_success(ledgers):
    result = uptake({4: ledgers[0]}, purchased_label_steps=[])
    assert result["registration_status"] == "never_registered"
    assert result["relative"]["status"] == "undefined_final_recall_zero"
    assert result["absolute"]["status"] == "never_reached"
    assert result["absolute"]["first_observed"] is None
    assert result["unavailable_counts"] == [1, 2, 5]


def test_unpurchased_class_cannot_have_predictions(ledgers):
    with pytest.raises(ValueError, match="unpurchased novel"):
        uptake({4: ledgers[4]}, purchased_label_steps=[])


def test_novel_prediction_before_first_purchase_is_rejected(ledgers):
    ledgers[0][-1]["predicted_label"] = "novel"
    with pytest.raises(ValueError, match="unpurchased novel"):
        uptake(ledgers)


def test_weak_final_recall_can_reach_relative_but_not_absolute_target(ledgers):
    for rows in ledgers.values():
        rows[-1]["predicted_label"] = "old_a"
    result = uptake(ledgers)
    assert result["final_recall"] == .5
    assert result["relative"]["target"] == .4
    assert result["relative"]["status"] == "reached"
    assert result["absolute"]["status"] == "never_reached"


def test_final_milestone_is_not_duplicated(ledgers):
    result = uptake(ledgers, purchased_label_steps=[1, 4], checkpoint_counts=[1, 2, 5])
    assert [x["processed_events"] for x in result["curve"]] == [1, 4]
    assert result["curve"][-1]["is_final"]


@pytest.mark.parametrize("kwargs", [
    {"purchased_label_steps": [1, 1]}, {"purchased_label_steps": [0]},
    {"purchased_label_steps": [5]}, {"checkpoint_counts": []},
    {"checkpoint_counts": [1, 3]}, {"relative_fraction": 0},
    {"absolute_target": float("nan")}, {"final_step": 3},
])
def test_rejects_invalid_uptake_policy_or_missing_observation(ledgers, kwargs):
    with pytest.raises(ValueError):
        uptake(ledgers, **kwargs)


def test_aulc_hand_calculated_irregular_grid():
    assert normalized_aulc({0: .2, 25: .4, 100: .8}, budget_grid=[0, 25, 100]) == pytest.approx(.525)
    grid = [0, 25, 50, 100, 200, 400]
    assert normalized_aulc({b: .6 for b in grid}, budget_grid=grid) == pytest.approx(.6)


@pytest.mark.parametrize("scores,grid", [
    ({0: .1}, [0, 25]), ({0: .1, 25: .2, 50: .3}, [0, 25]),
    ({0: .1, 25: math.nan}, [0, 25]), ({0: .1, 25: 1.1}, [0, 25]),
    ({0: .1, 25: .2}, [25, 0]), ({0: .1, 25: .2}, [0, 25, 25]),
    ({0: .1, 25: True}, [0, 25]), ({False: .1, 25: .2}, [0, 25]),
])
def test_aulc_rejects_incomplete_or_invalid_curves(scores, grid):
    with pytest.raises(ValueError):
        normalized_aulc(scores, budget_grid=grid)


def test_paired_contrasts_keep_sign_and_sample_sd():
    result = paired_seed_summary({0: .4, 1: .5, 2: .6}, {0: .3, 1: .3, 2: .3},
                                 expected_seeds=[0, 1, 2], scale=100)
    assert list(result["per_seed"].values()) == pytest.approx([10, 20, 30])
    assert result["mean"] == pytest.approx(20)
    assert result["median"] == pytest.approx(20)
    assert result["sample_sd"] == pytest.approx(10)
    assert result["direction"] == "left_minus_right"
    single = paired_seed_summary({0: .1}, {0: .2}, expected_seeds=[0])
    assert single["mean"] == pytest.approx(-.1)
    assert single["sample_sd"] is None


@pytest.mark.parametrize("left,right,seeds,scale", [
    ({0: .1}, {1: .2}, [0, 1], 1), ({0: math.inf}, {0: .2}, [0], 1),
    ({0: .1}, {0: .2}, [0, 0], 1), ({0: .1}, {0: .2}, [0], -1),
    ({False: .1}, {0: .2}, [0], 1), ({0: .1}, {0: .2}, [0], True),
])
def test_rejects_unpaired_or_invalid_seed_summaries(left, right, seeds, scale):
    with pytest.raises(ValueError):
        paired_seed_summary(left, right, expected_seeds=seeds, scale=scale)
