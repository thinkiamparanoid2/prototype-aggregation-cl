"""Static safety contracts for user-owned replication launchers."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))

from ssl_hdc_pilot.replication import run_batch as rb  # noqa: E402


def _text(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_full_run_launcher_has_three_ordered_resumable_stages() -> None:
    text = _text("replication-full-run.ps1")
    assert "replication-adapt-all.ps1" in text
    assert "-m ssl_hdc_pilot.replication.run_batch" in text and "--require-adapted" in text
    assert "-m ssl_hdc_pilot.replication.aggregate" in text
    adapt_index = text.index("& (Join-Path $PSScriptRoot 'replication-adapt-all.ps1')")
    batch_index = text.index("-m ssl_hdc_pilot.replication.run_batch")
    aggregate_index = text.index("-m ssl_hdc_pilot.replication.aggregate")
    assert adapt_index < batch_index < aggregate_index
    # 'Continue', not 'Stop': PS 5.1 turns native stderr into errors; every stage checks its exit code.
    assert "$ErrorActionPreference = 'Continue'" in text
    assert text.count("if ($LASTEXITCODE -ne 0)") >= 3
    assert "resume" in text.lower()


def test_replication_run_all_is_removed() -> None:
    assert not (SCRIPTS / "replication-run-all.ps1").exists()


def test_run_batch_plan_lists_the_locked_270_cells_without_creating_outputs() -> None:
    result_root = ROOT / "ssl_hdc_pilot" / "results" / "replication" / "cells"

    def tree_state() -> dict:
        if not result_root.exists():
            return {}
        return {path.relative_to(result_root): path.stat().st_mtime_ns for path in result_root.rglob("*")}

    before = tree_state()
    completed = subprocess.run(
        [sys.executable, "-m", "ssl_hdc_pilot.replication.run_batch", "--plan"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    cells = [line for line in completed.stdout.splitlines() if line[:1].isdigit()]
    assert len(cells) == 270
    assert cells[0].startswith("1 PlantDoc/frozen_vits16/hdc_s0")
    assert cells[-1].startswith("270 PlantVillage/adapted_vitl16/ranpac_s4")
    assert tree_state() == before


def test_run_batch_enumeration_matches_the_locked_cell_count() -> None:
    assert len(rb.enumerate_cells()) == 270


def test_replication_setup_launchers_are_execution_gated() -> None:
    for name in ("replication-acquire.ps1", "replication-extract.ps1", "replication-adapt.ps1"):
        text = _text(name)
        assert "[switch]$Execute" in text
        assert "if (-not $Execute)" in text
        assert "Prepare-only" in text


def test_extract_and_adapt_cover_every_dataset_and_encoder_size() -> None:
    for name in ("replication-extract.ps1", "replication-adapt.ps1"):
        text = _text(name)
        assert "PlantDoc', 'PlantWild', 'PlantVillage" in text
        assert "vits16', 'vitb16', 'vitl16" in text
