"""Tests for the read-only streaming-median fidelity check.

Model: test_aggregate_grid.py. These tests are deliberately cheap -- they do
not replay any learner against real capture artifacts (that is exercised
separately by an actual `--execute` run); they only check the three safety
properties the task requires: plan mode writes nothing, execute refuses to
overwrite an existing output directory, and this module never opens the
trusted test split.
"""
from __future__ import annotations

import builtins
import pathlib

import pytest

from ssl_hdc_pilot.scripts import streaming_median_fidelity as fidelity


# ---------------------------------------------------------------------------
# Plan mode: read-only, writes nothing.
# ---------------------------------------------------------------------------


def test_plan_mode_writes_nothing(tmp_path):
    output_dir = tmp_path / "would_be_output"
    result = fidelity.plan(
        capture_root=fidelity.DEFAULT_CAPTURE_ROOT,
        validation_root=fidelity.DEFAULT_VALIDATION_ROOT,
        output_dir=output_dir,
    )
    assert result["mode"] == "plan_only"
    assert not output_dir.exists()
    assert list(tmp_path.iterdir()) == []


def test_plan_mode_reports_cell_directories_without_reading_their_contents(tmp_path):
    # A nonexistent capture root must not raise; discover_cells() degrades to
    # an empty list rather than reading anything.
    result = fidelity.plan(
        capture_root=tmp_path / "does_not_exist",
        validation_root=tmp_path / "also_missing",
        output_dir=None,
    )
    assert result["would_read"]["cell_count"] == 0
    assert result["would_write"]["output_dir"] is None


# ---------------------------------------------------------------------------
# Execute mode: refuse to overwrite an existing output directory.
# ---------------------------------------------------------------------------


def test_execute_refuses_to_overwrite_existing_output_dir(tmp_path):
    existing_output = tmp_path / "already_here"
    existing_output.mkdir()

    with pytest.raises(FileExistsError):
        fidelity.run(
            capture_root=tmp_path / "no_such_capture_root",
            validation_root=tmp_path / "no_such_validation_root",
            train_csv=tmp_path / "no_such_train.csv",
            val_csv=tmp_path / "no_such_val.csv",
            embeddings_root=tmp_path / "no_such_embeddings",
            output_dir=existing_output,
        )
    # The refusal must happen before any cell discovery/reading; the existing
    # directory is untouched (still empty, not deleted or written into).
    assert existing_output.is_dir()
    assert list(existing_output.iterdir()) == []


# ---------------------------------------------------------------------------
# Guard: this module must never open the trusted test split.
# ---------------------------------------------------------------------------


def test_guard_fails_if_trusted_test_split_is_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    forbidden = ("trusted_splits", "test.csv")

    def _is_forbidden(path: object) -> bool:
        parts = pathlib.Path(str(path)).parts
        return any(parts[i:i + 2] == forbidden for i in range(len(parts) - 1))

    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):  # noqa: ANN001
        if _is_forbidden(file):
            raise AssertionError(f"forbidden open of trusted test split: {file}")
        return real_open(file, *args, **kwargs)

    real_path_open = pathlib.Path.open

    def guarded_path_open(self, *args, **kwargs):  # noqa: ANN001
        if _is_forbidden(self):
            raise AssertionError(f"forbidden Path.open of trusted test split: {self}")
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(pathlib.Path, "open", guarded_path_open)

    test_split = pathlib.Path("ssl_hdc_pilot/trusted_splits/test.csv")

    with pytest.raises(fidelity.TrustedTestSplitAccessError):
        fidelity._read_json(test_split)
    with pytest.raises(fidelity.TrustedTestSplitAccessError):
        fidelity._read_csv_rows(test_split)
    with pytest.raises(fidelity.TrustedTestSplitAccessError):
        fidelity._sha256_file(test_split)
    with pytest.raises(fidelity.TrustedTestSplitAccessError):
        fidelity._load_embeddings_npz(test_split)
    with pytest.raises(fidelity.TrustedTestSplitAccessError):
        fidelity._resolve_recorded_input(
            {"path": "whatever/test.csv", "sha256": "0" * 64}, root=pathlib.Path("ssl_hdc_pilot/trusted_splits"))


def test_guard_helper_detects_forbidden_path_directly() -> None:
    with pytest.raises(fidelity.TrustedTestSplitAccessError):
        fidelity._guard_no_test_split(pathlib.Path("ssl_hdc_pilot/trusted_splits/test.csv"))
    # A sibling file with a similar name must not be caught by mistake.
    fidelity._guard_no_test_split(pathlib.Path("ssl_hdc_pilot/trusted_splits/val.csv"))
    fidelity._guard_no_test_split(pathlib.Path("ssl_hdc_pilot/trusted_splits/train.csv"))
