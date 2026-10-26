"""Tests for the read-only, post-hoc forgetting-grid aggregator.

The acceptance test at the bottom of this file is the one that matters: it
runs the aggregator against the real saved artifacts on disk and checks that
every number it produces matches what RESULTS_LOG.md:1408-1485 claims. If any
of those assertions fail, that is a finding about the prose, not a bug to be
papered over by adjusting the aggregator's formulas to match.
"""
from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from ssl_hdc_pilot.scripts import aggregate_grid as agg


# ---------------------------------------------------------------------------
# per_class_f1_from_confusion
# ---------------------------------------------------------------------------

def test_per_class_f1_matches_hand_computed_values():
    # 3 classes. Class "a": TP=2,FN=1,FP=1 -> F1=2*2/(2*2+1+1)=4/6.
    # Class "b": TP=3,FN=1,FP=1 -> F1=2*3/(2*3+1+1)=6/8=0.75.
    # Class "c": never predicted and never true -> denom 0 -> F1=0.0.
    classes = ["a", "b", "c"]
    cm = [
        [2, 1, 0],
        [1, 3, 0],
        [0, 0, 0],
    ]
    result = agg.per_class_f1_from_confusion(classes, cm)
    assert result["a"] == pytest.approx(4 / 6)
    assert result["b"] == pytest.approx(0.75)
    assert result["c"] == 0.0


def test_per_class_f1_zero_denominator_is_explicit_zero_not_nan():
    classes = ["only"]
    cm = [[0]]
    result = agg.per_class_f1_from_confusion(classes, cm)
    assert result["only"] == 0.0


# ---------------------------------------------------------------------------
# parse_cell_identity
# ---------------------------------------------------------------------------

def test_parse_cell_identity_splits_arm_and_learner_using_manifest():
    parsed = agg.parse_cell_identity(
        "2026-09-15_grid_b400_adapted_hdc_s0_t010_attempt01",
        manifest_learner="hdc", manifest_seed=0)
    assert parsed == {
        "arm": "adapted", "learner": "hdc", "seed": 0,
        "budget": 400, "threshold": 10, "attempt": 1,
    }


def test_parse_cell_identity_handles_underscore_containing_learner():
    # A hypothetical mechanism-ablation learner name containing an underscore
    # (e.g. "sign_ncm") must not be split apart, because the split point is
    # resolved against the manifest's learner string, not guessed from the name.
    parsed = agg.parse_cell_identity(
        "2026-09-15_ablation_b400_s_frozen_sign_ncm_s2_t010_attempt01",
        manifest_learner="sign_ncm", manifest_seed=2)
    assert parsed["arm"] == "s_frozen"
    assert parsed["learner"] == "sign_ncm"
    assert parsed["seed"] == 2


def test_parse_cell_identity_seed_mismatch_is_hard_error():
    with pytest.raises(ValueError, match="does not match manifest stream_seed"):
        agg.parse_cell_identity(
            "2026-09-15_grid_b400_adapted_hdc_s0_t010_attempt01",
            manifest_learner="hdc", manifest_seed=1)


def test_parse_cell_identity_learner_mismatch_is_hard_error():
    with pytest.raises(ValueError, match="does not end with manifest learner"):
        agg.parse_cell_identity(
            "2026-09-15_grid_b400_adapted_hdc_s0_t010_attempt01",
            manifest_learner="ncm", manifest_seed=0)


# ---------------------------------------------------------------------------
# discover_cells: only well-formed grid-cell directories are picked up
# ---------------------------------------------------------------------------

def test_discover_cells_excludes_non_grid_and_archived_shapes(tmp_path):
    (tmp_path / "2026-09-15_grid_b400_adapted_hdc_s0_t010_attempt01").mkdir()
    # Archived pre-grid capture: budget token appears after the seed token.
    (tmp_path / "2026-09-12_b_frozen_ncm_s0_b400_t010_attempt01").mkdir()
    (tmp_path / "GRID_PLAN.md").write_text("not a directory match target", encoding="utf-8")
    (tmp_path / "preflight_run_abc123.log").write_text("", encoding="utf-8")

    found = agg.discover_cells(tmp_path)
    assert [p.name for p in found] == ["2026-09-15_grid_b400_adapted_hdc_s0_t010_attempt01"]


def test_discover_cells_on_missing_root_returns_empty():
    assert agg.discover_cells(Path("this/does/not/exist")) == []


# ---------------------------------------------------------------------------
# Trusted-test-split guard
# ---------------------------------------------------------------------------

def test_guard_refuses_trusted_splits_test_csv():
    with pytest.raises(agg.TrustedTestSplitAccessError):
        agg._read_json(Path("ssl_hdc_pilot/trusted_splits/test.csv"))
    with pytest.raises(agg.TrustedTestSplitAccessError):
        agg._sha256(Path("ssl_hdc_pilot/trusted_splits/test.csv"))


@pytest.mark.skipif(not agg.DEFAULT_CAPTURE_ROOT.exists(), reason="saved grid artifacts not present in this checkout")
def test_full_aggregation_never_opens_trusted_test_csv(monkeypatch):
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        if self.name == "test.csv":
            raise AssertionError("trusted test split must never be opened")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    cells = agg.load_all_cells(
        validation_root=agg.DEFAULT_VALIDATION_ROOT,
        capture_root=agg.DEFAULT_CAPTURE_ROOT,
        uptake_root=agg.DEFAULT_UPTAKE_ROOT,
    )
    assert len(cells) == 36
    agg.build_summary(cells)
    agg.build_concentration(cells)


# ---------------------------------------------------------------------------
# --execute refuses to overwrite an existing output directory
# ---------------------------------------------------------------------------

def test_run_refuses_to_overwrite_existing_output_dir(tmp_path):
    existing = tmp_path / "already_here"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        agg.run(
            validation_root=agg.DEFAULT_VALIDATION_ROOT,
            capture_root=agg.DEFAULT_CAPTURE_ROOT,
            uptake_root=agg.DEFAULT_UPTAKE_ROOT,
            output_dir=existing,
        )


def test_plan_mode_writes_nothing(tmp_path):
    target = tmp_path / "would_be_output"
    result = agg.plan(
        validation_root=agg.DEFAULT_VALIDATION_ROOT,
        capture_root=agg.DEFAULT_CAPTURE_ROOT,
        uptake_root=agg.DEFAULT_UPTAKE_ROOT,
        output_dir=target,
    )
    assert result["mode"] == "plan_only"
    assert not target.exists()


# ---------------------------------------------------------------------------
# Synthetic end-to-end wiring check for build_concentration's top-N exclusion
# ---------------------------------------------------------------------------

def _fake_cell(arm, learner, seed, per_class_f1):
    classes = list(per_class_f1)
    return {
        "arm": arm, "learner": learner, "seed": seed,
        "classes": classes, "per_class_f1": per_class_f1,
    }


def test_build_concentration_excludes_top_n_by_delta_not_by_name():
    cells = []
    for seed in (0, 1, 2):
        cells.append(_fake_cell("adapted", "hdc", seed, {"x": 0.9, "y": 0.8, "z": 0.5}))
        cells.append(_fake_cell("adapted", "ncm", seed, {"x": 0.1, "y": 0.1, "z": 0.5}))
        cells.append(_fake_cell("adapted", "ranpac", seed, {"x": 0.2, "y": 0.2, "z": 0.4}))
    result = agg.build_concentration(cells)
    assert result["status"] == "ok"
    # deltas: x=0.9-0.2=0.7, y=0.8-0.2=0.6, z=0.5-0.5=0.0
    assert result["top_contributing_classes"] == ["x", "y"]
    assert result["macro_delta_including_top"] == pytest.approx((0.7 + 0.6 + 0.0) / 3)
    assert result["macro_delta_excluding_top"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# THE ACCEPTANCE TEST THAT MATTERS: reproduce RESULTS_LOG.md:1408-1485
# from the real, saved artifacts on disk.
# ---------------------------------------------------------------------------

def _median_field(cells, arm, learner, field):
    values = [c[field] for c in cells if c["arm"] == arm and c["learner"] == learner]
    assert values, f"no cells found for arm={arm} learner={learner}"
    return statistics.median(values)


def _mean_per_class_f1(cells, arm, learner, class_label):
    values = [c["per_class_f1"][class_label] for c in cells if c["arm"] == arm and c["learner"] == learner]
    assert values, f"no cells found for arm={arm} learner={learner}"
    return statistics.mean(values)


@pytest.fixture(scope="module")
def grid_cells():
    # Raw captures (~600 MB) are kept out of the public repository; see README.
    if not agg.DEFAULT_CAPTURE_ROOT.exists():
        pytest.skip("saved grid artifacts not present in this checkout")
    return agg.load_all_cells(
        validation_root=agg.DEFAULT_VALIDATION_ROOT,
        capture_root=agg.DEFAULT_CAPTURE_ROOT,
        uptake_root=agg.DEFAULT_UPTAKE_ROOT,
    )


class TestGridAcceptance:
    @pytest.fixture
    def cells(self, grid_cells):
        return grid_cells

    def test_all_36_cells_present(self, cells):
        assert len(cells) == 36

    @pytest.mark.parametrize("arm,learner,expected", [
        ("frozen", "hdc", 0.7220),
        ("frozen", "ncm", 0.7313),
        ("frozen", "ranpac", 0.7406),
        ("adapted", "hdc", 0.6001),
        ("adapted", "ncm", 0.5275),
        ("adapted", "ranpac", 0.5243),
    ])
    def test_median_macro_f1(self, cells, arm, learner, expected):
        assert round(_median_field(cells, arm, learner, "macro_f1"), 4) == expected

    def test_adapted_hdc_per_seed_macro_f1(self, cells):
        values = sorted(round(c["macro_f1"], 4) for c in cells if c["arm"] == "adapted" and c["learner"] == "hdc")
        assert values == [0.5943, 0.6001, 0.6040]

    @pytest.mark.parametrize("arm,learner,expected", [
        ("frozen", "hdc", 0.0497),
        ("frozen", "ncm", 0.0435),
        ("frozen", "ranpac", 0.0451),
        ("adapted", "hdc", 0.0395),
        ("adapted", "ncm", 0.0533),
        ("adapted", "ranpac", 0.0651),
    ])
    def test_median_forgetting(self, cells, arm, learner, expected):
        assert round(_median_field(cells, arm, learner, "mean_forgetting"), 4) == expected

    @pytest.mark.parametrize("class_label,learner,expected", [
        ("2_00_0", "hdc", 0.741),
        ("2_00_0", "ncm", 0.062),
        ("2_00_0", "ranpac", 0.059),
        ("3_00_0", "hdc", 0.429),
        ("3_00_0", "ncm", 0.017),
        ("3_00_0", "ranpac", 0.017),
    ])
    def test_adapted_per_class_f1(self, cells, class_label, learner, expected):
        assert round(_mean_per_class_f1(cells, "adapted", learner, class_label), 3) == expected

    def test_concentration_macro_deltas(self, cells):
        concentration = agg.build_concentration(cells)
        assert concentration["status"] == "ok"
        assert round(concentration["macro_delta_including_top"], 4) == 0.0635
        assert round(concentration["macro_delta_excluding_top"], 4) == 0.0069
        assert set(concentration["top_contributing_classes"]) == {"2_00_0", "3_00_0"}
        assert concentration["focus_at_or_above_best_control_count"] == 13
        assert concentration["focus_at_or_above_best_control_total"] == 19

    def test_labels_consumed_400_for_every_prototype_head_cell(self, cells):
        for cell in cells:
            if cell["learner"] in agg.LABEL_MATCHED_LEARNERS:
                assert cell["labels_consumed"] == 400, cell["cell"]
                assert cell["allowance"] == 400, cell["cell"]

    def test_naive_ewc_replay_are_not_label_matched(self, cells):
        for cell in cells:
            if cell["learner"] in agg.NOT_LABEL_MATCHED_LEARNERS:
                assert cell["labels_consumed"] != 400, cell["cell"]

    def test_win_counts_match_prose(self, cells):
        summary = agg.build_summary(cells)
        wins = summary["win_counts"]
        assert wins["frozen"]["wins"]["hdc"] == 0
        assert wins["frozen"]["seeds_considered"] == 3
        assert wins["adapted"]["wins"]["hdc"] == 3
        assert wins["adapted"]["seeds_considered"] == 3
