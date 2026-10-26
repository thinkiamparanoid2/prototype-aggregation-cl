"""Targeted tests for run_batch.py cell enumeration, checkpoint-step schedule,
resume detection, and the attempt cap -- no real cell is ever executed here."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import pytest

from ssl_hdc_pilot.replication import run_batch as rb


def test_enumerate_cells_produces_the_locked_270_cell_order() -> None:
    cells = rb.enumerate_cells()
    assert len(cells) == 270
    assert cells[0] == rb.Cell("PlantDoc", "frozen_vits16", "hdc", 0)
    assert cells[-1] == rb.Cell("PlantVillage", "adapted_vitl16", "ranpac", 4)

    # dataset order: PlantDoc -> PlantWild -> PlantVillage, each a contiguous block of 90
    assert {c.dataset for c in cells[:90]} == {"PlantDoc"}
    assert {c.dataset for c in cells[90:180]} == {"PlantWild"}
    assert {c.dataset for c in cells[180:]} == {"PlantVillage"}

    # within a dataset: all 3 frozen arms (45 cells) before all 3 adapted arms (45 cells)
    first_dataset_arms = [c.arm for c in cells[:90]]
    assert first_dataset_arms[:45] == ["frozen_vits16"] * 15 + ["frozen_vitb16"] * 15 + ["frozen_vitl16"] * 15
    assert first_dataset_arms[45:] == ["adapted_vits16"] * 15 + ["adapted_vitb16"] * 15 + ["adapted_vitl16"] * 15

    # within one arm: hdc/ncm/ranpac x seeds 0-4, in that order
    first_arm_cells = cells[:15]
    assert [c.learner for c in first_arm_cells] == ["hdc"] * 5 + ["ncm"] * 5 + ["ranpac"] * 5
    assert [c.seed for c in first_arm_cells] == list(range(5)) * 3

    assert len({(c.dataset, c.arm, c.learner, c.seed) for c in cells}) == 270


def test_enumerate_cells_filters_and_limit() -> None:
    filtered = rb.enumerate_cells(dataset="PlantWild", arm="adapted_vitb16", learner="ncm")
    assert len(filtered) == 5
    assert {c.dataset for c in filtered} == {"PlantWild"}
    assert {c.arm for c in filtered} == {"adapted_vitb16"}
    assert {c.learner for c in filtered} == {"ncm"}
    assert [c.seed for c in filtered] == list(range(5))

    limited = rb.enumerate_cells()[:3]
    assert len(limited) == 3


def test_checkpoint_steps_schedule_matches_gate0_evidence() -> None:
    # ssl_hdc_pilot/results/replication/gate0_claude_2026-09-17/run_gate0.py used
    # steps=[0,195,196,390,391,585,586,780,781,976] for introductions_at=[195,390,585,780],
    # stream_length=976, and Gate 0 recorded all six macro-F1/forgetting deltas as 0.0.
    assert rb.checkpoint_steps_from_schedule([195, 390, 585, 780], 976) == [
        0, 195, 196, 390, 391, 585, 586, 780, 781, 976]


def test_checkpoint_steps_for_dataset_reads_real_plantdoc_feasibility() -> None:
    steps = rb.checkpoint_steps_for_dataset("PlantDoc")
    assert steps == [0, 100, 101, 200, 201, 300, 301, 401]
    assert steps[0] == 0
    assert steps[-1] == 401
    assert steps == sorted(set(steps))


def test_cell_status_done_missing_incomplete(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"
    assert rb.cell_status(missing_dir) == "missing"

    incomplete_dir = tmp_path / "incomplete"
    incomplete_dir.mkdir()
    assert rb.cell_status(incomplete_dir) == "incomplete"

    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (corrupt_dir / "integrity_report.json").write_text("not json", encoding="utf-8")
    assert rb.cell_status(corrupt_dir) == "incomplete"

    done_dir = tmp_path / "done"
    done_dir.mkdir()
    (done_dir / "metrics.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (done_dir / "extra.txt").write_text("x", encoding="utf-8")
    receipt = {"status": "PASS", "artifacts": {"metrics.json": "sha", "extra.txt": "sha2"}}
    (done_dir / "integrity_report.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert rb.cell_status(done_dir) == "done"

    # a receipt claiming an artifact that is not actually on disk is incomplete
    missing_artifact_dir = tmp_path / "missing_artifact"
    missing_artifact_dir.mkdir()
    (missing_artifact_dir / "metrics.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    bad_receipt = {"status": "PASS", "artifacts": {"metrics.json": "sha", "ghost.txt": "sha2"}}
    (missing_artifact_dir / "integrity_report.json").write_text(json.dumps(bad_receipt), encoding="utf-8")
    assert rb.cell_status(missing_artifact_dir) == "incomplete"


def test_quarantine_incomplete_and_attempt_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rb, "FAILED_CELLS_ROOT", tmp_path / "failed_cells")
    monkeypatch.setattr(rb, "CELLS_ROOT", tmp_path / "cells")
    cell = rb.Cell("PlantDoc", "frozen_vits16", "hdc", 0)
    cell.output_dir.mkdir(parents=True)
    (cell.output_dir / "junk.txt").write_text("partial", encoding="utf-8")

    assert rb.attempt_count(cell) == 0
    rb.quarantine_incomplete(cell)
    assert not cell.output_dir.exists()
    assert rb.attempt_count(cell) == 1

    # quarantining an already-absent directory is a no-op, not an error
    rb.quarantine_incomplete(cell)
    assert rb.attempt_count(cell) == 1


def test_process_cell_stops_the_batch_after_max_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rb, "FAILED_CELLS_ROOT", tmp_path / "failed_cells")
    monkeypatch.setattr(rb, "CELLS_ROOT", tmp_path / "cells")
    fake_embeddings = tmp_path / "frozen_vits16.npz"
    fake_embeddings.write_bytes(b"not a real npz, never opened by this test")
    monkeypatch.setattr(rb, "embeddings_path", lambda dataset, arm: fake_embeddings)
    monkeypatch.setattr(rb, "checkpoint_steps_for_dataset", lambda dataset: [0, 1])
    monkeypatch.setattr(rb, "free_space_ok", lambda **kwargs: (True, "G:\\"))

    call_count = {"n": 0}

    def fake_crash(cell, **kwargs):
        call_count["n"] += 1
        # simulate a subprocess that started writing output then crashed: the
        # directory exists but never gets a finished receipt.
        cell.output_dir.mkdir(parents=True, exist_ok=True)
        (cell.output_dir / "partial.txt").write_text("crash", encoding="utf-8")
        return {"returncode": 1, "elapsed_seconds": 0.01, "peak_rss_bytes": None}

    monkeypatch.setattr(rb, "run_cell_subprocess", fake_crash)

    cell = rb.Cell("PlantDoc", "frozen_vits16", "hdc", 0)
    with pytest.raises(rb.BatchStopped, match="attempt cap"):
        rb.process_cell(cell, python_exe="unused", threads=1, require_adapted=False,
                        stop_event=Event(), log_dir=tmp_path / "logs")
    assert call_count["n"] == rb.MAX_ATTEMPTS
    assert rb.attempt_count(cell) == rb.MAX_ATTEMPTS


def test_process_cell_skips_missing_adapted_embeddings_without_require_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rb, "CELLS_ROOT", tmp_path / "cells")
    monkeypatch.setattr(rb, "embeddings_path", lambda dataset, arm: tmp_path / "does_not_exist.npz")
    cell = rb.Cell("PlantDoc", "adapted_vits16", "hdc", 0)
    outcome = rb.process_cell(cell, python_exe="unused", threads=1, require_adapted=False,
                              stop_event=Event(), log_dir=tmp_path / "logs")
    assert outcome["status"] == "skipped"


def test_process_cell_requires_adapted_embeddings_when_flag_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rb, "CELLS_ROOT", tmp_path / "cells")
    monkeypatch.setattr(rb, "embeddings_path", lambda dataset, arm: tmp_path / "does_not_exist.npz")
    cell = rb.Cell("PlantDoc", "adapted_vits16", "hdc", 0)
    with pytest.raises(rb.BatchStopped, match="required embeddings not found"):
        rb.process_cell(cell, python_exe="unused", threads=1, require_adapted=True,
                        stop_event=Event(), log_dir=tmp_path / "logs")
