"""Synthetic checks for approved budget B and A/B snapshot capture."""
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from ssl_hdc_pilot.scripts import stream_runner as runner
from ssl_hdc_pilot.scripts import learner_checkpoint as checkpoint_io
from ssl_hdc_pilot.scripts import budget_experiment as budget_module
from ssl_hdc_pilot.scripts.budget_experiment import plan_budget_cell, run_budget_cell
from ssl_hdc_pilot.scripts.evaluate_validation import _fingerprint
from ssl_hdc_pilot.scripts.experiment_a import _json_hash, _prepare
from ssl_hdc_pilot.scripts.learner_checkpoint import load_learner_checkpoint
from ssl_hdc_pilot.tests.test_stream_runner import _StepClock, _write_smoke_study


@pytest.fixture
def inputs(tmp_path):
    embeddings, train, spec = _write_smoke_study(tmp_path, 4)
    payload = json.loads(spec.read_text())
    payload["protocol"]["confidence_thresholds"] = [0., 2.]
    spec.write_text(json.dumps(payload))
    return dict(embeddings_path=embeddings, split_csv=train, stream_spec_path=spec,
                stream_seed=17, smoke_mode=True)


@pytest.mark.parametrize("learner", runner.LEARNER_NAMES)
@pytest.mark.parametrize("experiment,budget", [("A", 0), ("A", 3), ("A", 7), ("B", 0), ("B", 1), ("B", 3), ("B", 7)])
def test_snapshots_match_unserialized_reference_and_preserve_accounting(inputs, tmp_path, learner, experiment, budget):
    output = tmp_path / "output"
    before = {path: path.read_bytes() for path in inputs.values() if isinstance(path, Path)}
    threshold = 2. if experiment == "B" else None
    # Classifier expansion consumes caller CPU RNG even though learned weights
    # are deterministic. Both compared passes must start at the same RNG state.
    initial_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    report = run_budget_cell(**inputs, experiment=experiment, threshold=threshold,
        learner_name=learner, budget=budget, output_dir=output, clock=_StepClock())
    assert report["status"] == "PASS"
    assert report["labels_consumed"] == budget
    assert report["retention_checkpoints"] == [0, 2, 3, 7]
    for name, digest in report["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    index = json.loads((output / "snapshot_index.json").read_text())["snapshots"]
    times = [s["processed_stream_events"] for s in index]
    assert times == sorted(set(times)) and set([0, 2, 3, 7]) <= set(times)
    assert index[-1]["path"] == "final_state.npz"
    assert not (output / "snapshots/step_000007.npz").exists()
    with (output / "history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    purchased_steps = [int(r["step"]) + 1 for r in history if r["label_requested"] == "True" and r["true_label"] == "novel_c"]
    expected_times = {0, 2, 3, 7} | {purchased_steps[n-1] for n in [1, 2, 5, 10, 20, 50, 100] if n <= len(purchased_steps)}
    assert set(times) == expected_times
    events = json.loads((output / "label_events.json").read_text())
    assert events["accounting"]["labels_consumed"] == budget
    if experiment == "B":
        assert events["accounting"]["potential_requests"] == 7
        assert events["accounting"]["denied_requests"] == 7 - budget
    assert sum(e["purchased"] for e in events["rows"]) == budget
    assert all(not e["class_registered"] or e["purchased"] for e in events["rows"])

    # Independent unserialized runner pass: same outputs and exact state/RNG at each capture.
    spec, _, bootstrap, stream, dim, schedule, _ = _prepare(**inputs, learner_name=learner, budget=budget)
    effective = replace(spec, max_total_requests=budget, bootstrap_counts_toward_budget=False,
                        count_low_confidence=experiment == "B", count_novel_registration=experiment == "B")
    expected_states = {}
    expected_details = {}
    def remember(model, progress):
        if progress["processed_stream_events"] in times:
            expected_states[progress["processed_stream_events"]] = _fingerprint(model)
            arrays = {}
            state = checkpoint_io._export_learner(model, arrays)
            rng = checkpoint_io._capture_random_state(arrays)
            expected_details[progress["processed_stream_events"]] = {
                "state": state, "rng": rng, "arrays": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256": checkpoint_io._array_digest(value)} for key, value in arrays.items()}}
    shared = frozenset(e["image_id"] for e in schedule["selected_events"]) if experiment == "A" else None
    random.setstate(initial_rng[0])
    np.random.set_state(initial_rng[1])
    torch.set_rng_state(initial_rng[2])
    baseline_history, baseline_summary = runner._run_one(learner_name=learner, threshold=threshold,
        bootstrap=bootstrap, stream=stream, spec=effective, input_dim=dim, clock=_StepClock(),
        shared_label_ids=shared, on_observation=remember)
    baseline_path = tmp_path / "baseline_history.csv"
    runner._write_history(baseline_path, baseline_history)
    assert baseline_path.read_bytes() == (output / "history.csv").read_bytes()
    summary = json.loads((output / "summary.json").read_text())
    for name, value in baseline_summary.items():
        assert summary[name] == value
    for snapshot in index:
        with np.load(output / snapshot["path"], allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata_json"].tobytes().decode())
        actual = _json_hash({key: metadata[key] for key in ("state", "rng", "arrays")})
        detail = expected_details[snapshot["processed_stream_events"]]
        mismatched_arrays = [key for key in metadata["arrays"] if metadata["arrays"][key] != detail["arrays"].get(key)]
        assert actual == expected_states[snapshot["processed_stream_events"]], (
            snapshot["processed_stream_events"], mismatched_arrays,
            _json_hash(metadata["state"]) == _json_hash(detail["state"]),
            _json_hash(metadata["rng"]) == _json_hash(detail["rng"]))
        restored = load_learner_checkpoint(output / snapshot["path"], shared_projection_root=output)
        assert restored.context["progress"]["processed_stream_events"] == snapshot["processed_stream_events"]
        assert restored.context["progress"]["labels_charged"] == snapshot["labels_consumed"]
        assert restored.context["snapshot_reasons"] == snapshot["reasons"]
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("learner", runner.LEARNER_NAMES)
def test_b_allows_underspending_and_registers_only_purchased_novel_class(inputs, tmp_path, learner):
    report = run_budget_cell(**inputs, experiment="B", threshold=0., learner_name=learner,
        budget=7, output_dir=tmp_path / "output", clock=_StepClock())
    assert report["status"] == "PASS"
    assert report["labels_consumed"] == 1 and report["unused_allowance"] == 6
    events = json.loads((tmp_path / "output/label_events.json").read_text())
    assert events["accounting"]["potential_requests"] == 1
    assert events["accounting"]["purchases_by_class"] == {"novel_c": 1}


def test_double_trigger_is_charged_once(inputs, tmp_path):
    report = run_budget_cell(**inputs, experiment="B", threshold=2., learner_name="ncm",
        budget=7, output_dir=tmp_path / "output", clock=_StepClock())
    events = json.loads((tmp_path / "output/label_events.json").read_text())["rows"]
    double = [e for e in events if e["request_reasons"]["low_confidence"] and e["request_reasons"]["novel_registration"]]
    assert len(double) == 1
    assert report["labels_consumed"] == len(events) == 7


def test_default_plan_has_no_model_or_output_side_effects(inputs, tmp_path, monkeypatch):
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError("plan cannot construct a learner")
    monkeypatch.setattr(runner, "_build_learner", forbidden)
    plan = plan_budget_cell(**inputs, experiment="B", threshold=2., learner_name="ncm", budget=3)
    assert plan["policy"]["snapshot_policy"]["retention_processed_events"] == [0, 2, 3, 7]
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("experiment,threshold", [("A", .1), ("B", None), ("B", .2), ("B", float("nan")), ("B", True), ("C", None)])
def test_rejects_unapproved_acquisition_parameters(inputs, experiment, threshold):
    with pytest.raises(ValueError):
        plan_budget_cell(**inputs, experiment=experiment, threshold=threshold, learner_name="ncm", budget=3)


def test_preserves_existing_output_and_requires_real_preflight(inputs, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        run_budget_cell(**inputs, experiment="B", threshold=2., learner_name="ncm", budget=3, output_dir=output)
    assert (output / "keep.txt").read_text() == "preserve"
    with pytest.raises(RuntimeError, match="manual user launch"):
        run_budget_cell(**{**inputs, "smoke_mode": False}, experiment="B", threshold=.1,
                        learner_name="ncm", budget=25, output_dir=tmp_path / "new")


def test_rejects_test_partition_before_opening_inputs(tmp_path):
    with pytest.raises(RuntimeError, match="canonical approved train"):
        plan_budget_cell(experiment="B", threshold=.1, embeddings_path=tmp_path / "missing.npz",
            split_csv=tmp_path / "test.csv", stream_spec_path=tmp_path / "missing.json",
            learner_name="ncm", stream_seed=0, budget=25, smoke_mode=False)


@pytest.mark.parametrize("learner", runner.ABLATION_LEARNER_NAMES)
def test_canonical_default_rejects_ablation_learners(inputs, learner):
    with pytest.raises(ValueError, match="unknown learner"):
        plan_budget_cell(**inputs, experiment="B", threshold=2., learner_name=learner, budget=3)


@pytest.mark.parametrize("ablation_mode,learner", [(False, "ncm"), (True, "sign_ncm")])
def test_checkpoint_context_matches_evaluator_whitelist(inputs, tmp_path, ablation_mode, learner):
    output = tmp_path / "output"
    report = run_budget_cell(**inputs, experiment="B", threshold=2., learner_name=learner,
        budget=3, output_dir=output, clock=_StepClock(), ablation_mode=ablation_mode)
    manifest = json.loads((output / "manifest.json").read_text())
    index = json.loads((output / "snapshot_index.json").read_text())
    snapshot = index["snapshots"][0]
    context = load_learner_checkpoint(output / snapshot["path"], shared_projection_root=output).context
    common_keys = {"schema_version", "artifact_kind", "stage", "data_role", "experiment",
        "learner", "stream_seed", "policy", "policy_sha256", "inputs", "code_hashes",
        "bootstrap_image_ids", "stream_image_ids"}
    assert set(context) == common_keys | {"progress", "snapshot_reasons"}
    assert context == {**{key: manifest[key] for key in common_keys},
        "progress": context["progress"], "snapshot_reasons": snapshot["reasons"]}
    assert "ablation_mode" not in context["policy"]
    assert manifest["ablation_mode"] is ablation_mode
    assert report["ablation_mode"] is ablation_mode


def test_snapshot_changed_after_capture_is_not_accepted(inputs, tmp_path, monkeypatch):
    output = tmp_path / "output"
    original_write = budget_module._write_json
    def corrupt_after_summary(path, payload):
        original_write(path, payload)
        if path.name == "summary.json":
            with (output / "snapshots/step_000000.npz").open("ab") as handle:
                handle.write(b"changed")
    monkeypatch.setattr(budget_module, "_write_json", corrupt_after_summary)
    report = run_budget_cell(**inputs, experiment="B", threshold=2., learner_name="ncm",
                            budget=3, output_dir=output, clock=_StepClock())
    assert report["status"] == "FAIL"
    assert "snapshot_changed_after_capture" in report["failures"]
