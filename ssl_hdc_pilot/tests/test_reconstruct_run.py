"""Reconstruction tests use generated vectors and fake histories, never trusted data."""
from pathlib import Path
import json
import csv
import hashlib
import io

import pytest

from ssl_hdc_pilot.scripts.reconstruct_run import (
    EARLIER_SPEC_SHA, compare_values, plan_reconstruction, reconstruct_run,
    recover_source_spec,
)
from ssl_hdc_pilot.scripts.stream_runner import DEFAULT_STREAM_SPEC, LEARNER_NAMES, run_study
from ssl_hdc_pilot.scripts.learner_checkpoint import load_learner_checkpoint
from ssl_hdc_pilot.scripts import stream_runner as runner
from ssl_hdc_pilot.scripts.continual_metrics import final_predictive_metrics
from ssl_hdc_pilot.tests.test_stream_runner import _StepClock, _write_smoke_study


@pytest.mark.parametrize("learner", LEARNER_NAMES)
def test_reconstructs_one_cell_and_retains_fidelity_evidence(tmp_path: Path, learner: str) -> None:
    embeddings, split, spec = _write_smoke_study(tmp_path, 4)
    source = run_study(
        embeddings_path=embeddings, split_csv=split, stream_spec_path=spec,
        output_dir=tmp_path / "source", smoke_mode=True, learner_names=(learner,),
        clock=_StepClock(),
    )[0]
    before = source.summary_json.read_bytes(), source.history_csv.read_bytes()
    output = tmp_path / "reconstructed"
    report = reconstruct_run(
        source_summary=source.summary_json, embeddings_path=embeddings,
        split_csv=split, stream_spec_path=spec, output_dir=output,
        smoke_mode=True, clock=_StepClock(),
    )
    assert report["status"] == "PASS"
    assert report["history"]["mismatch_count"] == 0
    assert report["summary"]["mismatch_count"] == 0
    assert report["stream_steps"] == 7
    assert len(list(output.glob("*_final_state.npz"))) == 1
    checkpoint = load_learner_checkpoint(next(output.glob("*_final_state.npz")))
    assert checkpoint.context["progress"]["stream_complete"] is True
    assert checkpoint.context["progress"]["processed_stream_events"] == 7
    assert json.loads((output / "fidelity_report.json").read_text())["status"] == "PASS"
    manifest = json.loads((output / "reconstruction_manifest.json").read_text())
    assert len(manifest["event_schedule_sha256"]) == 64
    assert manifest["environment"]["requested_device"] == "cpu"
    assert manifest["data_role"] == "synthetic_smoke"
    assert "head_type" in manifest["registered_configuration"]["baseline"]
    assert report["code_changed_during_run"] == []
    predictions = json.loads((output / "final_training_predictions.json").read_text())
    assert predictions["data_role"] == "synthetic_smoke"
    assert len(predictions["rows"]) == 9
    assert len({row["image_id"] for row in predictions["rows"]}) == 9
    saved_summary = json.loads(next(output.glob("*_summary.json")).read_text())
    assert final_predictive_metrics(predictions["rows"], ece_bins=5) == saved_summary["final_metrics"]
    assert (source.summary_json.read_bytes(), source.history_csv.read_bytes()) == before


@pytest.fixture
def synthetic_source(tmp_path: Path) -> dict:
    embeddings, split, spec = _write_smoke_study(tmp_path, 4)
    source = run_study(
        embeddings_path=embeddings, split_csv=split, stream_spec_path=spec,
        output_dir=tmp_path / "source", smoke_mode=True, learner_names=("ncm",),
        clock=_StepClock(),
    )[0]
    return dict(source_summary=source.summary_json, embeddings_path=embeddings,
                split_csv=split, stream_spec_path=spec, smoke_mode=True)


def test_plan_is_read_only_and_pins_inputs(synthetic_source: dict, tmp_path: Path) -> None:
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    plan = plan_reconstruction(**synthetic_source)
    assert plan["spec_recovery"] == "exact_source_spec"
    assert plan["historical_input_hashes_available"] is False
    assert plan["inputs"]["source_summary"]["sha256"] == hashlib.sha256(
        synthetic_source["source_summary"].read_bytes()).hexdigest()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_refuses_existing_output_directory(synthetic_source: dict, tmp_path: Path) -> None:
    output = tmp_path / "source"
    before = {p: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError, match="overwrite/reuse"):
        reconstruct_run(**synthetic_source, output_dir=output)
    assert {p: p.read_bytes() for p in output.iterdir()} == before


def test_official_execution_requires_preflight_before_reading_inputs(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="user launch and contamination preflight"):
        reconstruct_run(source_summary=tmp_path / "missing.json", embeddings_path=tmp_path / "missing.npz",
                        split_csv=tmp_path / "missing.csv", stream_spec_path=tmp_path / "missing-spec.json",
                        output_dir=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_official_plan_refuses_nontrain_path_before_opening_any_inputs(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="canonical approved Option-A train"):
        plan_reconstruction(source_summary=tmp_path / "missing.json",
                            embeddings_path=tmp_path / "missing.npz", split_csv=tmp_path / "test.csv",
                            stream_spec_path=DEFAULT_STREAM_SPEC, smoke_mode=False)


def test_official_training_validation_never_reads_heldout_partitions(
    synthetic_source: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Virtualize the filesystem boundary only: all CSV labels here are generated.
    spec = json.loads(DEFAULT_STREAM_SPEC.read_text())
    tag = "option_a_section6a_v1__embeddings_4__stream_0__ncm__margin_0p1"
    old = synthetic_source["source_summary"]
    source = tmp_path / f"{tag}_summary.json"
    payload = json.loads(old.read_text())
    payload.update(run_tag=tag, smoke=False, stream_seed=0, confidence_threshold=0.1,
                   stream_spec_sha256=hashlib.sha256(DEFAULT_STREAM_SPEC.read_bytes()).hexdigest())
    source.write_text(json.dumps(payload))
    source.with_name(f"{tag}_history.csv").write_bytes(
        old.with_name(old.name.replace("_summary.json", "_history.csv")).read_bytes())
    classes = spec["protocol"]["initially_known_classes"] + [
        item["class_label"] for item in spec["protocol"]["class_introductions"]]
    buffer = io.StringIO()
    columns = ["image", "label", "split_partition", "split_policy", "split_seed",
               "supervisor_approved", "provenance_version", "source_csv"]
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for index in range(3396):
        writer.writerow(dict(image=f"fabricated_{index}", label=classes[index % 19],
                             split_partition="train", split_policy=runner.APPROVED_POLICY,
                             split_seed=runner.OFFICIAL_SEED, supervisor_approved="true",
                             provenance_version=runner.PROVENANCE_VERSION, source_csv=runner.CANONICAL_SOURCE_CSV))
    canonical_train = runner.DEFAULT_APPROVED_SPLIT_DIR / "train.csv"
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() == canonical_train.resolve():
            return io.StringIO(buffer.getvalue())
        if path.name in {"test.csv", "val.csv"}:
            pytest.fail("reconstruction attempted to read held-out labels")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="training-only holdback derivation"):
        plan_reconstruction(source_summary=source, embeddings_path=synthetic_source["embeddings_path"],
                            split_csv=canonical_train, stream_spec_path=DEFAULT_STREAM_SPEC, smoke_mode=False)


@pytest.mark.parametrize("learner", ["hdc", "naive", "replay"])
def test_recovers_exact_registered_historical_spec_without_mutation(learner: str) -> None:
    original = DEFAULT_STREAM_SPEC.read_bytes()
    recovered, reason = recover_source_spec(original, EARLIER_SPEC_SHA, learner)
    assert hashlib.sha256(recovered).hexdigest() == EARLIER_SPEC_SHA
    assert reason == "exact_hash_recovery_of_pre_ewc_decay_amendment"
    assert DEFAULT_STREAM_SPEC.read_bytes() == original


@pytest.mark.parametrize("learner", ["ewc", "ncm", "ranpac"])
def test_does_not_apply_historical_recovery_to_unregistered_learners(learner: str) -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        recover_source_spec(DEFAULT_STREAM_SPEC.read_bytes(), EARLIER_SPEC_SHA, learner)


def test_unknown_spec_hash_refused_before_outputs(synthetic_source: dict, tmp_path: Path) -> None:
    source = synthetic_source["source_summary"]
    payload = json.loads(source.read_text())
    payload["stream_spec_sha256"] = "0" * 64
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        reconstruct_run(**synthetic_source, output_dir=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_discrete_values_are_exact_and_only_floats_have_tolerance() -> None:
    assert compare_values({"score": 0.5}, {"score": 0.5000005})["mismatch_count"] == 0
    assert compare_values({"score": 0.5}, {"score": 0.500002})["mismatch_count"] == 1
    assert compare_values({"count": 1}, {"count": 1.00000001})["mismatch_count"] == 1
    assert compare_values({"requested": True}, {"requested": 1})["mismatch_count"] == 1
    assert compare_values(["a", "b"], ["b", "a"])["mismatch_count"] == 2
    assert compare_values({"score": float("nan")}, {"score": float("nan")})["mismatch_count"] == 1
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compare_values({}, {}, atol=float("inf"))


@pytest.mark.parametrize("field", ["label_requested", "confidence", "image_id", "timing"])
def test_historical_history_changes_are_not_silently_accepted(
    synthetic_source: dict, tmp_path: Path, field: str,
) -> None:
    summary = synthetic_source["source_summary"]
    history = summary.with_name(summary.name.replace("_summary.json", "_history.csv"))
    with history.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    if field == "timing":
        rows[0]["update_wall_time_seconds"] = "987.0"
    elif field == "label_requested":
        rows[0][field] = "False" if rows[0][field] == "True" else "True"
    elif field == "confidence":
        rows[0][field] = str(float(rows[0][field]) + 0.1)
    else:
        rows[0][field] = "different-id"
    with history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "output"
    if field == "image_id":
        with pytest.raises(ValueError, match="IDs/order/labels"):
            reconstruct_run(**synthetic_source, output_dir=output, clock=_StepClock())
        assert not output.exists()
    else:
        report = reconstruct_run(**synthetic_source, output_dir=output, clock=_StepClock())
        assert report["status"] == ("PASS" if field == "timing" else "FAIL")


@pytest.mark.parametrize("field", ["total", "update_count"])
def test_timing_exclusion_does_not_hide_update_count_changes(
    synthetic_source: dict, tmp_path: Path, field: str,
) -> None:
    source = synthetic_source["source_summary"]
    payload = json.loads(source.read_text())
    payload["update_compute_seconds"][field] += 1
    source.write_text(json.dumps(payload))
    report = reconstruct_run(**synthetic_source, output_dir=tmp_path / "output", clock=_StepClock())
    assert report["status"] == ("PASS" if field == "total" else "FAIL")
