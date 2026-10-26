"""Synthetic acceptance for the authorized --hypervector-dim ablation passthrough.

Amendment CODEX_ABLATION_BATCH_RUNNER_AMENDMENT.md Sec.2 narrowly authorises a
hypervector-dimension override through budget_experiment.py, experiment_a.py and
stream_runner.py, gated on --ablation-mode, locked to {1000, 2000, 10000}, applying
to the hdc learner only, and recorded only in the manifest/report (never in policy
or the checkpoint context). This module is additive coverage; it does not modify
any existing test.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ssl_hdc_pilot.scripts import budget_experiment as budget_module
from ssl_hdc_pilot.scripts import stream_runner as runner
from ssl_hdc_pilot.scripts.budget_experiment import run_budget_cell
from ssl_hdc_pilot.scripts.experiment_a import ABLATION_HYPERVECTOR_DIMS, _prepare
from ssl_hdc_pilot.tests.test_stream_runner import _StepClock, _write_smoke_study


@pytest.fixture
def inputs(tmp_path: Path) -> dict:
    embeddings, train, spec = _write_smoke_study(tmp_path, 4)
    return dict(embeddings_path=embeddings, split_csv=train, stream_spec_path=spec,
                stream_seed=17, smoke_mode=True)


def test_hypervector_dim_requires_ablation_mode(inputs: dict) -> None:
    with pytest.raises(ValueError, match="ablation_mode"):
        _prepare(**inputs, learner_name="hdc", budget=3, ablation_mode=False, hypervector_dim=1000)


@pytest.mark.parametrize("value", [500, 3000, 10001, 0, -1000, True])
def test_hypervector_dim_is_locked_to_the_three_registered_values(inputs: dict, value) -> None:
    with pytest.raises(ValueError, match="1000, 2000, 10000"):
        _prepare(**inputs, learner_name="hdc", budget=3, ablation_mode=True, hypervector_dim=value)


@pytest.mark.parametrize("value", sorted(ABLATION_HYPERVECTOR_DIMS))
def test_hypervector_dim_accepts_exactly_the_three_registered_values(inputs: dict, value) -> None:
    # Must not raise; ABLATION_HYPERVECTOR_DIMS is exactly {1000, 2000, 10000}.
    _prepare(**inputs, learner_name="hdc", budget=3, ablation_mode=True, hypervector_dim=value)


def test_registered_dims_are_exactly_1000_2000_10000() -> None:
    assert ABLATION_HYPERVECTOR_DIMS == frozenset({1000, 2000, 10000})


def test_override_applies_to_hdc_only_ranpac_keeps_spec_dimension(inputs: dict) -> None:
    spec = runner.load_stream_spec(inputs["stream_spec_path"], smoke_mode=True)
    hdc_learner = runner._build_learner("hdc", spec=spec, input_dim=4, ablation_mode=True,
                                        hypervector_dim=1000)
    assert hdc_learner.hypervector_dim == 1000
    ranpac_learner = runner._build_learner("ranpac", spec=spec, input_dim=4, ablation_mode=True,
                                           hypervector_dim=1000)
    assert ranpac_learner.projection_dim == spec.hypervector_dim == 10_000


def test_override_none_falls_back_to_spec_dimension(inputs: dict) -> None:
    spec = runner.load_stream_spec(inputs["stream_spec_path"], smoke_mode=True)
    learner = runner._build_learner("hdc", spec=spec, input_dim=4, ablation_mode=True,
                                    hypervector_dim=None)
    assert learner.hypervector_dim == spec.hypervector_dim == 10_000


def test_manifest_and_report_record_dim_only_when_ablation_mode(inputs: dict, tmp_path: Path) -> None:
    output = tmp_path / "ablation_output"
    report = run_budget_cell(**inputs, experiment="B", threshold=2., learner_name="hdc",
        budget=3, output_dir=output, clock=_StepClock(), ablation_mode=True, hypervector_dim=1000)
    assert report["status"] == "PASS"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["ablation_hypervector_dim"] == 1000
    assert report["ablation_hypervector_dim"] == 1000
    # Never leaked into the pinned policy object or its hash-relevant contents.
    assert "ablation_hypervector_dim" not in manifest["policy"]
    assert "hypervector_dim" not in manifest["policy"]

    canonical_output = tmp_path / "canonical_output"
    canonical_report = run_budget_cell(**inputs, experiment="B", threshold=2., learner_name="ncm",
        budget=3, output_dir=canonical_output, clock=_StepClock())
    canonical_manifest = json.loads((canonical_output / "manifest.json").read_text())
    assert "ablation_hypervector_dim" not in canonical_manifest
    assert "ablation_hypervector_dim" not in canonical_report


def test_checkpoint_context_excludes_hypervector_dim_override(inputs: dict, tmp_path: Path) -> None:
    from ssl_hdc_pilot.scripts.learner_checkpoint import load_learner_checkpoint

    output = tmp_path / "output"
    run_budget_cell(**inputs, experiment="B", threshold=2., learner_name="hdc", budget=3,
                    output_dir=output, clock=_StepClock(), ablation_mode=True, hypervector_dim=1000)
    index = json.loads((output / "snapshot_index.json").read_text())
    snapshot = index["snapshots"][0]
    context = load_learner_checkpoint(output / snapshot["path"], shared_projection_root=output).context
    common_keys = {"schema_version", "artifact_kind", "stage", "data_role", "experiment",
        "learner", "stream_seed", "policy", "policy_sha256", "inputs", "code_hashes",
        "bootstrap_image_ids", "stream_image_ids"}
    assert set(context) == common_keys | {"progress", "snapshot_reasons"}
    assert "ablation_hypervector_dim" not in context
    assert "ablation_hypervector_dim" not in context["policy"]


def test_default_canonical_run_manifest_and_report_keys_unchanged(inputs: dict, tmp_path: Path) -> None:
    output = tmp_path / "output"
    report = run_budget_cell(**inputs, experiment="B", threshold=2., learner_name="ncm",
        budget=3, output_dir=output, clock=_StepClock())
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest) == {"schema_version", "artifact_kind", "stage", "data_role", "experiment",
        "learner", "stream_seed", "policy", "policy_sha256", "inputs", "code_hashes",
        "bootstrap_image_ids", "stream_image_ids", "base_stream_spec_sha256", "ablation_mode",
        "ablation_arm", "checkpoint_storage", "registered_base_configuration", "git", "protocol",
        "environment", "scope"}
    assert set(report) == {"status", "artifact_kind", "experiment", "stage", "learner", "stream_seed",
        "budget", "threshold", "ablation_mode", "ablation_arm", "stream_steps", "labels_consumed",
        "unused_allowance", "snapshots_saved", "retention_checkpoints", "failures",
        "inputs_changed_during_run", "code_changed_during_run", "artifacts", "scope"}


def test_cli_rejects_hypervector_dim_without_ablation_mode(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, "-m", "ssl_hdc_pilot.scripts.budget_experiment",
        "--experiment", "B", "--threshold", "0.1", "--embeddings", str(tmp_path / "missing.npz"),
        "--trusted-split", str(tmp_path / "missing.csv"), "--learner", "hdc", "--seed", "0",
        "--budget", "3", "--hypervector-dim", "1000"],
        capture_output=True, text=True, cwd=str(runner.REPO_ROOT), timeout=60)
    assert result.returncode != 0
    assert "--hypervector-dim" in result.stderr
