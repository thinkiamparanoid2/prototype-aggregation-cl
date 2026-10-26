"""Acceptance tests for the mechanism-ablation batch launcher.

Modeled on ssl_hdc_pilot/tests/test_forgetting_grid.py's launcher checks: static
required-substring assertions on the script text (the same fidelity level that
file uses for scripts/run-forgetting-grid-all.ps1), plus one real -PlanOnly
execution since -PlanOnly is read-only (no learner, no capture/evaluation
subprocess, no directory writes) and safe to run directly.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run-mechanism-ablation-all.ps1"


def test_batch_runner_declares_required_guards_and_phase_order():
    script = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "PlanOnly", "RUN ABLATION", "batch_progress.jsonl", "Attempt cap reached",
        "attempt{0:D2}", "SKIPPED (already complete)", "benchmark_environment", "8GB",
        "--ablation-mode", "--output-dir", "--hypervector-dim", "mechanism_ablation",
        "Get-PackageLearners", "phase_order='2,1,3'",
    ):
        assert required in script, f"missing required token: {required}"
    assert "test.csv" not in script
    # --output-dir must appear on both generated python invocations (capture and
    # evaluation); --ablation-mode is only valid on the capture invocation (the amendment
    # explicitly forbids adding a CLI flag to evaluate_snapshots.py).
    assert script.count("--output-dir") >= 2
    assert "budget_experiment" in script and "evaluate_snapshots" in script


def test_original_single_cell_prober_is_kept_working():
    prober = REPO_ROOT / "scripts" / "run-mechanism-ablation.ps1"
    assert prober.is_file()
    text = prober.read_text(encoding="utf-8")
    assert "PlanOnly" in text and "test.csv" not in text


@pytest.mark.skipif(sys.platform != "win32" or not (REPO_ROOT / ".venv-e1").is_dir(),
                    reason="Windows PowerShell launcher using the project .venv-e1")
def test_plan_only_lists_120_cells_in_phase_2_then_1_then_3_order_and_writes_nothing(tmp_path):
    capture_dir = REPO_ROOT / "ssl_hdc_pilot/results/mechanism_ablation/capture"
    validation_dir = REPO_ROOT / "ssl_hdc_pilot/results/mechanism_ablation/validation"
    progress = REPO_ROOT / "ssl_hdc_pilot/results/mechanism_ablation/batch_progress.jsonl"
    before = {
        "capture": capture_dir.is_dir(),
        "validation": validation_dir.is_dir(),
        "progress": progress.is_file(),
    }
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-File", str(SCRIPT), "-PlanOnly"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    cell_lines = [line for line in lines if line.split(" ", 1)[0].isdigit()]
    assert len(cell_lines) == 120
    # Cell 1 opens Phase 2, cell 30/31 is the Phase 2 -> Phase 1 boundary, cell 75/76 is the
    # Phase 1 -> Phase 3 boundary, matching the 30/45/45 = 120 cell-count reconciliation.
    assert cell_lines[0].split(" ")[1] == "phase2"
    assert cell_lines[29].split(" ")[1] == "phase2"
    assert cell_lines[30].split(" ")[1] == "phase1"
    assert cell_lines[74].split(" ")[1] == "phase1"
    assert cell_lines[75].split(" ")[1] == "phase3"
    assert cell_lines[119].split(" ")[1] == "phase3"
    phases_in_order = [line.split(" ")[1] for line in cell_lines]
    assert phases_in_order == (["phase2"] * 30) + (["phase1"] * 45) + (["phase3"] * 45)
    # -PlanOnly must never write: no capture/validation directories, no progress file.
    assert capture_dir.is_dir() == before["capture"]
    assert validation_dir.is_dir() == before["validation"]
    assert progress.is_file() == before["progress"]
    if not before["capture"]:
        assert not capture_dir.is_dir()
    if not before["validation"]:
        assert not validation_dir.is_dir()


@pytest.mark.skipif(sys.platform != "win32" or not (REPO_ROOT / ".venv-e1").is_dir(),
                    reason="Windows PowerShell launcher using the project .venv-e1")
def test_plan_only_phase_filter_shows_only_that_phase(tmp_path):
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-File", str(SCRIPT), "-PlanOnly", "-Phase", "3"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    cell_lines = [line for line in lines if line.split(" ", 1)[0].isdigit()]
    assert len(cell_lines) == 45
    assert all(line.split(" ")[1] == "phase3" for line in cell_lines)
    assert all("dim=" in line for line in cell_lines)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell launcher")
def test_confirmation_is_required_before_any_execute_path(tmp_path):
    # Non-interactive stdin means Read-Host cannot be answered; the script must fail closed
    # (never fall through to an execute path) rather than silently proceeding unattended.
    capture_dir = REPO_ROOT / "ssl_hdc_pilot/results/mechanism_ablation/capture"
    before = capture_dir.is_dir()
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-File", str(SCRIPT)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
    )
    assert result.returncode != 0
    assert capture_dir.is_dir() == before
