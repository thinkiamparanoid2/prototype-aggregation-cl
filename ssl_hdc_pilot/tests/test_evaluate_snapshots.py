"""Synthetic acceptance for the A/B snapshot validation public entry points."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ssl_hdc_pilot.scripts import evaluate_snapshots as evaluation
from ssl_hdc_pilot.scripts import budget_experiment as capture
from ssl_hdc_pilot.scripts import stream_runner as runner
from ssl_hdc_pilot.tests.test_evaluate_validation import study, CLASSES
from ssl_hdc_pilot.tests.test_stream_runner import _StepClock


def source(study, tmp_path, learner="ncm", experiment="B", budget=7):
    embeddings, train, spec, val = study
    output = tmp_path / "source"
    capture.run_budget_cell(experiment=experiment, threshold=2. if experiment == "B" else None,
        embeddings_path=embeddings, split_csv=train, stream_spec_path=spec,
        learner_name=learner, stream_seed=17, budget=budget, output_dir=output,
        smoke_mode=True, clock=_StepClock())
    return dict(source_dir=output, embeddings_path=embeddings, validation_csv=val, smoke_mode=True)


@pytest.mark.parametrize("learner", runner.LEARNER_NAMES)
@pytest.mark.parametrize("experiment,budget", [("A", 0), ("A", 7), ("B", 0), ("B", 7)])
def test_all_snapshots_are_pure_fixed_population_and_nonoverwriting(study, tmp_path, learner, experiment, budget):
    kwargs = source(study, tmp_path, learner, experiment, budget)
    before = {p: p.read_bytes() for p in kwargs["source_dir"].rglob("*") if p.is_file()}
    plan = evaluation.plan_snapshot_validation(**kwargs)
    assert plan["samples_per_snapshot"] == 3 and plan["classes"] == CLASSES
    output = tmp_path / "evaluation"
    report = evaluation.run_snapshot_validation(**kwargs, output_dir=output)
    assert report["status"] == "PASS" and report["state_and_rng_unchanged"]
    assert report["snapshots_scored"] == plan["snapshots"]
    assert report["samples_per_snapshot"] == 3
    for name, digest in report["artifacts"].items():
        assert runner._file_sha256(output / name) == digest
    result = json.loads((output / "metrics.json").read_text())
    assert result["forgetting"]["checkpoint_steps"] == [0, 2, 3, 7]
    assert result["forgetting"]["old_classes"] == CLASSES[:2]
    assert all(point["samples"] == 2 for point in result["forgetting"]["curve"])
    for snapshot in result["snapshots"]:
        assert snapshot["learner_rng_fingerprint_before"] == snapshot["learner_rng_fingerprint_after"]
        assert snapshot["classes"] == CLASSES
        rows = json.loads((output / snapshot["predictions"]).read_text())["rows"]
        assert [row["image_id"] for row in rows] == ["val_0", "val_1", "val_2"]
    if budget == 0:
        uptake = result["uptake"][CLASSES[2]]
        assert uptake["registration_status"] == "never_registered"
        assert uptake["relative"]["status"] == "undefined_final_recall_zero"
        assert uptake["absolute"]["status"] == "never_reached"
    assert {p: p.read_bytes() for p in before} == before
    with pytest.raises(FileExistsError):
        evaluation.run_snapshot_validation(**kwargs, output_dir=output)


def test_plan_never_loads_a_learner_or_writes(study, tmp_path, monkeypatch):
    kwargs = source(study, tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError("planning must not load any learner")
    monkeypatch.setattr(evaluation.checkpoints, "load_learner_checkpoint", forbidden)
    evaluation.plan_snapshot_validation(**kwargs)
    assert {p: p.read_bytes() for p in before} == before
    assert set(before) == {p for p in tmp_path.rglob("*") if p.is_file()}


def repin(directory, filename):
    path = directory / "integrity_report.json"
    receipt = json.loads(path.read_text())
    receipt["artifacts"][filename] = runner._file_sha256(directory / filename)
    path.write_text(json.dumps(receipt))


@pytest.mark.parametrize("problem", ["missing_blob", "unlisted_blob", "extra_blob"])
def test_shared_blob_receipt_dependency_closure_is_required(study, tmp_path, problem, monkeypatch):
    kwargs = source(study, tmp_path, learner="hdc")
    directory = kwargs["source_dir"]
    blob = next((directory / "projection_blobs").glob("*.npz"))
    receipt_path = directory / "integrity_report.json"
    receipt = json.loads(receipt_path.read_text())
    if problem == "missing_blob":
        blob.unlink()
    elif problem == "unlisted_blob":
        del receipt["artifacts"][blob.relative_to(directory).as_posix()]
        receipt_path.write_text(json.dumps(receipt))
    else:
        extra = directory / "projection_blobs" / ("0" * 64 + ".npz")
        extra.write_bytes(blob.read_bytes())
        repin(directory, extra.relative_to(directory).as_posix())
    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before scoring")
    monkeypatch.setattr(evaluation.validation, "evaluate_fixed_population", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluation.plan_snapshot_validation(**kwargs)


@pytest.mark.parametrize("problem", ["corrupt", "missing_snapshot", "false_event", "false_count", "unsafe_path", "policy", "code"])
def test_source_tampering_is_rejected_before_scoring(study, tmp_path, problem):
    kwargs = source(study, tmp_path)
    directory = kwargs["source_dir"]
    if problem == "corrupt":
        (directory / "final_state.npz").write_bytes(b"bad")
    elif problem in {"missing_snapshot", "false_count"}:
        name = "snapshot_index.json"
        data = json.loads((directory / name).read_text())
        if problem == "missing_snapshot":
            data["snapshots"].pop(0)
        else:
            data["snapshots"][0]["labels_consumed"] = 1
        (directory / name).write_text(json.dumps(data))
        repin(directory, name)
    elif problem == "false_event":
        name = "label_events.json"
        data = json.loads((directory / name).read_text())
        data["rows"][0]["image_id"] = "not-the-stream-id"
        (directory / name).write_text(json.dumps(data))
        repin(directory, name)
    elif problem == "unsafe_path":
        path = directory / "integrity_report.json"
        data = json.loads(path.read_text())
        data["artifacts"]["../outside"] = "0" * 64
        path.write_text(json.dumps(data))
    else:
        name = "manifest.json"
        data = json.loads((directory / name).read_text())
        if problem == "policy":
            data["policy"]["snapshot_policy"]["absolute_recall_target"] = .2
        else:
            data["code_hashes"].pop(next(iter(data["code_hashes"])))
        (directory / name).write_text(json.dumps(data))
        repin(directory, name)
    with pytest.raises(ValueError):
        evaluation.plan_snapshot_validation(**kwargs)


def test_forbidden_partition_and_manual_guards_precede_input_access(tmp_path):
    kwargs = dict(source_dir=tmp_path / "missing", embeddings_path=tmp_path / "missing.npz",
                  validation_csv=tmp_path / "test.csv")
    with pytest.raises(RuntimeError, match="validation-only"):
        evaluation.plan_snapshot_validation(**kwargs)
    with pytest.raises(RuntimeError, match="manual user launch"):
        evaluation.run_snapshot_validation(**kwargs, output_dir=tmp_path / "out")


@pytest.mark.parametrize("problem", ["overlap", "class", "embedding"])
def test_rejects_invalid_validation_population(study, tmp_path, problem):
    kwargs = source(study, tmp_path)
    path = kwargs["validation_csv"]
    text = path.read_text()
    text = text.replace("val_0", "image_0" if problem == "overlap" else "absent") if problem != "class" else text.replace(CLASSES[2], CLASSES[0])
    path.write_text(text)
    with pytest.raises(ValueError):
        evaluation.plan_snapshot_validation(**kwargs)


def test_evaluation_cannot_write_inside_source(study, tmp_path):
    kwargs = source(study, tmp_path)
    with pytest.raises(ValueError, match="inside source"):
        evaluation.run_snapshot_validation(**kwargs, output_dir=kwargs["source_dir"] / "evaluation")


def test_scorer_receives_only_validation_embeddings(study, tmp_path, monkeypatch):
    kwargs = source(study, tmp_path)
    original = evaluation.validation.evaluate_fixed_population
    calls = []
    def checked(learner, rows, embeddings, **options):
        assert set(embeddings) == {"val_0", "val_1", "val_2"}
        calls.append(1)
        return original(learner, rows, embeddings, **options)
    monkeypatch.setattr(evaluation.validation, "evaluate_fixed_population", checked)
    report = evaluation.run_snapshot_validation(**kwargs, output_dir=tmp_path / "eval")
    assert len(calls) == report["snapshots_scored"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell user launcher")
@pytest.mark.parametrize("mode", ["success", "plan", "cancel", "failed_native", "exists"])
def test_launcher_binds_log_path_and_never_launches_without_guards(mode):
    # Exercise the actual script with harmless command mocks. No real subprocess
    # learner, environment audit, log writes or experiment output is permitted.
    script = runner.REPO_ROOT / "scripts/validate-b400-snapshots.ps1"
    harness = r'''
    $ErrorActionPreference = 'Stop'
    $global:Calls = @()
    $global:LogPaths = @()
    $global:Prompts = 0
    $global:LASTEXITCODE = 0
    function rtk {
        $global:Calls += ,@($args)
        $global:LASTEXITCODE = 0
        if ($Mode -eq 'failed_native') { $global:LASTEXITCODE = 8 }
        Write-Output 'harmless mocked command output'
    }
    function Test-Path { param($LiteralPath) return ($Mode -eq 'exists') }
    function New-Item { param($ItemType, [switch]$Force, $Path) }
    function Get-Process { [pscustomobject]@{ ProcessName='synthetic'; Id=1; CPU=0 } }
    function Read-Host { param($Prompt) $global:Prompts++; if ($Mode -eq 'cancel') { return 'NO' }; return 'RUN SNAPSHOT VAL' }
    function Tee-Object {
        param([Parameter(Mandatory=$true)][string]$FilePath, [switch]$Append,
              [Parameter(ValueFromPipeline=$true)]$InputObject)
        process {
            if ([string]::IsNullOrWhiteSpace($FilePath)) { throw 'Missing log path' }
            $global:LogPaths += $FilePath
            $InputObject
        }
    }
    $caught = $false
    try { & $ScriptPath -PlanOnly:($Mode -eq 'plan') | Out-Null } catch { $caught = $true }
    $execs = @($global:Calls | Where-Object { $_ -contains '--execute' }).Count
    if ($Mode -eq 'success') {
        if ($caught -or $execs -ne 1 -or $global:Prompts -ne 1 -or $global:LogPaths.Count -eq 0) {
            throw 'Success path lost arguments or prompted unexpectedly'
        }
        if (@($global:LogPaths | Sort-Object -Unique).Count -ne 1) { throw 'Log paths diverged' }
    } else {
        if ($execs -ne 0) { throw 'Scoring launched despite guard' }
        if (($Mode -eq 'plan') -eq $caught) { throw 'Unexpected guard outcome' }
    }
    Write-Output 'PASS launcher harness'
    '''
    prefix = "$Mode = '" + mode + "'; $ScriptPath = '" + str(script).replace("'", "''") + "';\n"
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", prefix + harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS launcher harness" in result.stdout
