"""Synthetic-only benchmark fidelity, accounting, timing and safety tests."""
from dataclasses import replace
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ssl_hdc_pilot.scripts import benchmark_core as core
from ssl_hdc_pilot.scripts import benchmark_study as study
from ssl_hdc_pilot.scripts import benchmark_environment as environment
from ssl_hdc_pilot.scripts import learner_checkpoint as checkpoints
from ssl_hdc_pilot.scripts import stream_runner as runner
from ssl_hdc_pilot.tests.test_stream_runner import _StepClock, _write_smoke_study


@pytest.fixture
def inputs(tmp_path):
    emb, train, spec = _write_smoke_study(tmp_path, 4)
    return dict(embeddings_path=emb, split_csv=train, stream_spec_path=spec,
                stream_seed=17, smoke_mode=True)


@pytest.mark.parametrize("learner", core.ORDER)
@pytest.mark.parametrize("experiment,budget", [("A", 0), ("A", 3), ("B", 0), ("B", 3)])
def test_measured_pass_matches_registered_reference_and_restores_rng(inputs, tmp_path, learner, experiment, budget):
    threshold = 2. if experiment == "B" else None
    spec, _, bootstrap, stream, dim, schedule, _, _ = study._prepare(**inputs, experiment=experiment, budget=budget, threshold=threshold)
    effective = replace(spec, max_total_requests=budget, bootstrap_counts_toward_budget=False,
                        count_low_confidence=experiment == "B", count_novel_registration=experiment == "B")
    selected = frozenset(e["image_id"] for e in schedule["selected_events"]) if experiment == "A" else None
    checkpoint = tmp_path / "bootstrap.npz"
    def observe(model, progress):
        if progress["processed_stream_events"] == 0:
            checkpoints.save_learner_checkpoint(checkpoint, model)
    predictions = []
    history, _ = runner._run_one(learner_name=learner, threshold=threshold, bootstrap=bootstrap, stream=stream,
        spec=effective, input_dim=dim, clock=_StepClock(), on_observation=observe,
        shared_label_ids=selected, on_final_predictions=predictions.extend)
    for mode in ("throughput", "latency", "untimed"):
        model = checkpoints.load_learner_checkpoint(checkpoint, restore_rng=True).learner
        calls = []
        result = core.stream_pass(model, stream, budget=budget, novel_classes=spec.novel_classes,
            threshold=threshold, selected_ids=selected, mode=mode, clock=_StepClock(), sync=lambda device:calls.append(device))
        expected_trace = [{k:r[k] for k in result["trace"][0]} for r in history]
        assert result["trace"] == expected_trace
        assert core.final_predictions(model, [*bootstrap, *stream]) == predictions
        assert len(calls) == (2 if mode == "throughput" else 2 * budget if mode == "latency" else 0)
        if mode == "throughput":
            assert result["stream_events_per_second"] == pytest.approx(7 / result["elapsed_stream_seconds"])
            assert result["update_latency"] is None and result["update_seconds"] == []
        elif mode == "latency":
            assert result["elapsed_stream_seconds"] is None
            assert result["update_latency"]["count"] == budget
            if budget == 0:
                assert result["update_latency"]["updates_per_second"] is None
        report = core.payload_report(model)
        assert report["total_exported_tensor_bytes"] == sum(report[k] for k in
            ["prediction_tensor_bytes", "extra_resumable_tensor_bytes", "replay_embedding_bytes"])
        if learner == "replay":
            assert report["replay"]["occupancy"] == min(2 + budget, 3)
            assert report["replay_embedding_bytes"] == report["replay"]["occupancy"] * dim * 4


@pytest.mark.parametrize("experiment,threshold", [("A", None), ("B", 2.)])
def test_complete_synthetic_benchmark_five_paired_blocks_preserves_inputs(inputs, tmp_path, experiment, threshold):
    before = {p:p.read_bytes() for p in inputs.values() if isinstance(p, Path)}
    output = tmp_path / "benchmark"
    receipt = study.run_benchmark(**inputs, experiment=experiment, threshold=threshold, budget=3,
                                  output_dir=output, clock=_StepClock())
    assert receipt["status"] == "PASS" and receipt["paired_blocks"] == 5
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["checkpoint_storage"] == study.checkpoint_storage.STORAGE_DESCRIPTION
    blobs = {name for name in receipt["artifacts"] if name.startswith("projection_blobs/")}
    assert blobs
    assert (manifest["experiment"], manifest["budget"], manifest["stream_seed"]) == (experiment, 3, 17)
    for name, digest in receipt["artifacts"].items():
        assert runner._file_sha256(output / name) == digest
    summary = json.loads((output / "summary.json").read_text())
    assert summary["environment_qualification"] == receipt["environment_qualification"]
    assert receipt["environment_qualification"]["qualification"] == "synthetic_no_machine_claim"
    assert receipt["environment_qualification"]["isolated_machine"] is False
    assert len(summary["timings"]) == 60
    assert len(summary["paired_ratios"]) == 5
    assert set(summary["per_learner"]) == set(core.ORDER)
    for block in summary["paired_ratios"]:
        assert all(ratio == 1 for ratio in block["hdc_over_comparator"].values())
    for name, memory in summary["memory"].items():
        assert [s["processed_stream_events"] for s in memory["state_samples"]] == [0, 2, 3, 7]
        for sample in memory["state_samples"]:
            assert sample["prediction_only_npz_bytes"] > 0
            assert sample["full_resume_npz_bytes"] > sample["prediction_only_npz_bytes"]
            assert sample["payload"]["global_rng_tensor_bytes"] > 0
            dependencies = sample["shared_projection_dependencies"]
            assert set(dependencies) <= blobs
            shared_bytes = sum((output / p).stat().st_size for p in dependencies)
            assert sample["shared_projection_bytes"] == shared_bytes
            assert bool(dependencies) == (name in {"hdc", "ranpac"})
            assert sample["full_resume_dependency_closure_bytes"] == sample["full_resume_npz_bytes"] + shared_bytes
            assert sample["prediction_only_dependency_closure_bytes"] == sample["prediction_only_npz_bytes"] + shared_bytes
    assert {p:p.read_bytes() for p in before} == before
    with pytest.raises(FileExistsError):
        study.run_benchmark(**inputs, experiment=experiment, threshold=threshold, budget=3, output_dir=output)


def test_plan_does_not_construct_learner(inputs, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("plan constructed learner")
    monkeypatch.setattr(runner, "_build_learner", forbidden)
    files = set(tmp_path.rglob("*"))
    plan = study.plan_benchmark(**inputs, experiment="A", threshold=None, budget=3)
    assert plan["paired_blocks"] == 5
    assert (plan["experiment"], plan["budget"], plan["stream_seed"]) == ("A", 3, 17)
    assert set(tmp_path.rglob("*")) == files


def test_guard_rejects_real_execution_without_user_preflight(tmp_path):
    with pytest.raises(RuntimeError, match="manual user launch"):
        study.run_benchmark(smoke_mode=False, output_dir=tmp_path / "output")


def test_only_registered_block_order_and_zero_purchase_latency():
    assert core.block_order(1) == ("ncm", "ranpac", "naive", "ewc", "replay", "hdc")
    for value in (-1, 5, True):
        with pytest.raises(ValueError):
            core.block_order(value)
    assert core.latency_summary([])["median_seconds"] is None
    for value in (-1., 0., float("nan")):
        with pytest.raises(ValueError):
            core.latency_summary([value])


@pytest.mark.parametrize("mode", ["throughput", "latency"])
def test_inventory_hook_is_rejected_before_timing(mode):
    def forbidden(*args):
        raise AssertionError("inventory or timing started")
    with pytest.raises(ValueError, match="inventory hooks are forbidden"):
        core.stream_pass(None, [], budget=0, novel_classes=[], selected_ids=frozenset(),
                         threshold=None, mode=mode, clock=forbidden, sync=forbidden,
                         on_observation=forbidden)


def test_memory_sampling_is_separate_and_reports_sampled_peak():
    result, report = environment.memory_pass(lambda: "value", "cpu", sample=lambda: 123)
    assert result == "value"
    assert report["sampled_working_set_peak_bytes"] == 123
    assert report["cuda_peak_allocated_bytes"] is None


def test_fidelity_failure_preserves_partial_attempt(inputs, tmp_path, monkeypatch):
    original = core.final_predictions
    def wrong(*args, **kwargs):
        rows = original(*args, **kwargs)
        rows[0]["prediction"] = "wrong"
        return rows
    monkeypatch.setattr(core, "final_predictions", wrong)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="fidelity mismatch"):
        study.run_benchmark(**inputs, experiment="A", threshold=None, budget=3,
                            output_dir=output, clock=_StepClock())
    assert (output / "hdc/bootstrap.npz").exists()
    assert json.loads((output / "failure.json").read_text())["status"] == "FAIL"
    assert not (output / "integrity_report.json").exists()


def test_storage_distinguishes_content_dedup_from_file_allocation(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    result = environment.storage_report([a, b], runner._file_sha256)
    assert result["logical_file_bytes_before_final_receipt"] == 8
    assert result["content_dedup_potential_bytes_before_final_receipt"] == 4
    assert result["filesystem_allocated_unique_file_bytes_before_final_receipt"] >= 0


@pytest.mark.skipif(os.name != "nt", reason="Windows workload guard")
@pytest.mark.parametrize("problem", [None, "browser", "gpu", "unreadable", "empty_tasks", "invalid_tasks", "invalid_pid"])
def test_contamination_guard_fails_closed(monkeypatch, problem):
    def fake(command, **kwargs):
        if command[0] == "tasklist":
            return SimpleNamespace(stdout={
                "browser": '"msedge.exe","999999"\n', "empty_tasks": '',
                "invalid_tasks": 'unreadable inventory', "invalid_pid": '"System","unknown"\n',
            }.get(problem, '"System","4"\n'))
        return SimpleNamespace(stdout='999999, python.exe\n' if problem == "gpu" else 'unsupported' if problem == "unreadable" else '')
    monkeypatch.setattr(environment.subprocess, "run", fake)
    if problem:
        with pytest.raises(RuntimeError):
            environment.contamination_check()
    else:
        assert environment.contamination_check()["blocked_processes"] == []


@pytest.mark.skipif(os.name != "nt", reason="Windows workload guard")
@pytest.mark.parametrize("path", [r"C:\Windows\explorer.exe",
    r"C:\Windows\System32\ShellHost.exe", r"G:\newshit\Microsoft VS Code\Code.exe",
    r"G:\WindowsApps\Microsoft.WindowsTerminal_1.24.11911.0_x64__8wekyb3d8bbwe\WindowsTerminal.exe"])
def test_approved_desktop_baseline_is_recorded_not_rejected(monkeypatch, path):
    def fake(command, **kwargs):
        return SimpleNamespace(stdout='"System","4"\n' if command[0] == "tasklist" else f'999999, {path}\n')
    monkeypatch.setattr(environment.subprocess, "run", fake)
    result = environment.contamination_check()
    assert result["other_gpu_compute"] == []
    assert result["accepted_desktop_baseline"] == [{"pid": 999999, "name": path}]


@pytest.mark.skipif(os.name != "nt", reason="Windows workload guard")
@pytest.mark.parametrize("path", [r"C:\Temp\explorer.exe", r"C:\Temp\Code.exe",
    r"C:\Temp\WindowsTerminal.exe", r"C:\Temp\python.exe", r"C:\Temp\unknown.exe",
    r"C:\Program Files (x86)\Microsoft\EdgeWebView\Application\152.0.4191.66\msedgewebview2.exe"])
def test_unapproved_gpu_processes_are_not_silently_accepted(monkeypatch, path):
    def fake(command, **kwargs):
        return SimpleNamespace(stdout='"System","4"\n' if command[0] == "tasklist" else f'999999, {path}\n')
    monkeypatch.setattr(environment.subprocess, "run", fake)
    monkeypatch.setattr(environment, "_search_process_snapshot", lambda: {})
    if path.endswith("python.exe"):
        with pytest.raises(RuntimeError, match="competing workloads"):
            environment.contamination_check()
    else:
        result = environment.contamination_check()
        assert result["accepted_desktop_baseline"] == []
        assert result["unclassified_gpu_processes"] == [{"pid": 999999, "name": path}]


@pytest.mark.skipif(os.name != "nt", reason="Windows process ownership")
@pytest.mark.parametrize("owner", ["search", "browser", "wrong_path", "cycle", "reused_pid", "different_child"])
def test_only_verified_windows_search_webview_chain_is_baseline(monkeypatch, owner):
    webview = r"C:\Program Files (x86)\Microsoft\EdgeWebView\Application\152.0.4191.66\msedgewebview2.exe"
    search = r"C:\Windows\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\SearchHost.exe"
    records = {
        999997: {"pid": 999997, "parent_pid": 1, "path": search, "started_ticks": 10},
        999998: {"pid": 999998, "parent_pid": 999997, "path": webview, "started_ticks": 20},
        999999: {"pid": 999999, "parent_pid": 999998, "path": webview, "started_ticks": 30},
    }
    if owner == "browser":
        records.pop(999997)
    elif owner == "wrong_path":
        records[999997]["path"] = r"C:\Temp\SearchHost.exe"
    elif owner == "cycle":
        records[999998]["parent_pid"] = 999999
    elif owner == "reused_pid":
        records[999997]["started_ticks"] = 40
    elif owner == "different_child":
        records[999999]["path"] = r"C:\Temp\msedgewebview2.exe"
    def fake(command, **kwargs):
        return SimpleNamespace(stdout='"System","4"\n' if command[0] == "tasklist" else f'999999, {webview}\n')
    monkeypatch.setattr(environment.subprocess, "run", fake)
    monkeypatch.setattr(environment, "_search_process_snapshot", lambda: records)
    if owner != "search":
        result = environment.contamination_check()
        assert result["accepted_desktop_baseline"] == []
        assert result["unclassified_gpu_processes"] == [{"pid": 999999, "name": webview}]
    else:
        result = environment.contamination_check()
        chain = result["accepted_desktop_baseline"][0]["verified_owner_chain"]
        assert [node["pid"] for node in chain] == [999999, 999998, 999997]
        assert result["other_gpu_compute"] == []


@pytest.mark.parametrize("payload", ["invalid", '{}', '[{"pid": -1}]',
    '[{"pid": 1, "parent_pid": 0, "path": "path", "started_ticks": 1}, {"pid": 1, "parent_pid": 0, "path": "path", "started_ticks": 1}]'])
def test_unreadable_search_ownership_snapshot_fails_closed(monkeypatch, payload):
    monkeypatch.setattr(environment.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=payload))
    with pytest.raises(RuntimeError, match="ownership inventory"):
        environment._search_process_snapshot()


@pytest.mark.skipif(os.name != "nt", reason="Windows workload guard")
@pytest.mark.parametrize("line", [r", C:\Windows\explorer.exe", r"-1, C:\Windows\explorer.exe",
    r"0, C:\Windows\explorer.exe", "999999,", "999999", "999999,path,extra"])
def test_malformed_gpu_rows_fail_before_baseline_allowance(monkeypatch, line):
    def fake(command, **kwargs):
        return SimpleNamespace(stdout='"System","4"\n' if command[0] == "tasklist" else line)
    monkeypatch.setattr(environment.subprocess, "run", fake)
    with pytest.raises(RuntimeError, match="cannot interpret GPU process inventory"):
        environment.contamination_check()


@pytest.mark.skipif(os.name != "nt", reason="Windows workload guard")
@pytest.mark.parametrize("name", ["Overwolf.exe", "OverwolfBrowser.exe"])
def test_overwolf_is_blocked_even_without_gpu_listing(monkeypatch, name):
    def fake(command, **kwargs):
        return SimpleNamespace(stdout=f'"{name}","999999"\n' if command[0] == "tasklist" else '')
    monkeypatch.setattr(environment.subprocess, "run", fake)
    with pytest.raises(RuntimeError, match="competing workloads"):
        environment.contamination_check()


@pytest.mark.skipif(sys.platform != "win32", reason="real Windows PowerShell stderr behavior")
@pytest.mark.skipif(shutil.which("rtk") is None, reason="launcher helper runs commands through the rtk CLI")
@pytest.mark.parametrize("native_exit", [0, 9])
def test_logged_native_stderr_is_complete_and_exit_checked(tmp_path, native_exit):
    # Extract the actual launcher helper, then invoke a harmless native stderr emitter.
    # No experiment command, preflight, or model construction runs in this test.
    import re
    script = (runner.REPO_ROOT / "scripts/benchmark-a25.ps1").read_text()
    helper = re.search(r"(?ms)^    function Invoke-LoggedCommand.*?^    }", script).group(0)
    log = tmp_path / "native.log"
    prefix = "$PythonPath = '" + sys.executable.replace("'", "''") + "'; "
    prefix += "$RunLogPath = '" + str(log).replace("'", "''") + "'; "
    prefix += "$NativeExit = " + str(native_exit) + ";\n"
    harness = r'''
    $ErrorActionPreference = 'Stop'
    $TeeParameters = @{ FilePath=$RunLogPath; Append=$true }
    $message = ''
    try {
        Invoke-LoggedCommand @($PythonPath, '-c', "import sys; print('stdout-ok', flush=True); print('stderr-first', file=sys.stderr, flush=True); print('stderr-last', file=sys.stderr, flush=True); sys.exit($NativeExit)")
    } catch { $message = $_.Exception.Message }
    $text = Get-Content -LiteralPath $RunLogPath -Raw
    if ($text -notmatch 'stderr-first' -or $text -notmatch 'stderr-last' -or $text -notmatch 'stdout-ok') { throw 'native output truncated' }
    if ($NativeExit -eq 0 -and $message) { throw ('warning became fatal: ' + $message) }
    if ($NativeExit -ne 0 -and $message -notmatch 'exit 9') { throw ('native exit lost: ' + $message) }
    if ($ErrorActionPreference -ne 'Stop') { throw 'error preference leaked' }
    Write-Output 'PASS native stderr harness'
    '''
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", prefix + helper + harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS native stderr harness" in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell launcher")
@pytest.mark.parametrize("mode", ["success", "plan", "cancel", "failed_native", "exists", "low_disk", "workload_blocked"])
@pytest.mark.parametrize("condition", ["A25", "B400"])
def test_benchmark_launcher_guards_and_log_binding(mode, condition):
    path = runner.REPO_ROOT / f"scripts/benchmark-{condition.lower()}.ps1"
    harness = r'''
    $ErrorActionPreference = 'Stop'
    $global:Calls = @(); $global:Logs = @(); $global:Prompts = 0; $global:LASTEXITCODE = 0
    function rtk {
        $global:Calls += ,@($args); $global:LASTEXITCODE = 0
        if ($Mode -eq 'failed_native') { $global:LASTEXITCODE = 8 }
        if ($Mode -eq 'workload_blocked' -and $args -contains 'ssl_hdc_pilot.scripts.benchmark_environment') { $global:LASTEXITCODE = 1 }
        Write-Output 'mock'
    }
    function Test-Path { param($LiteralPath) return ($Mode -eq 'exists') }
    function New-Item { param($ItemType, [switch]$Force, $Path) }
    function New-Object {
        param($TypeName, $ArgumentList)
        if ($TypeName -ne 'System.IO.DriveInfo') { throw 'unexpected object' }
        [pscustomobject]@{ AvailableFreeSpace = $(if ($Mode -eq 'low_disk') { 0 } else { 20GB }) }
    }
    function Get-Process { [pscustomobject]@{ ProcessName='synthetic'; Id=1; CPU=0 } }
    function Read-Host { param($Prompt) $global:Prompts++; if ($Mode -eq 'cancel') { return 'NO' }; return "RUN BENCH $Condition" }
    function Tee-Object {
        [CmdletBinding()]
        param([Parameter(Mandatory=$true)][string]$FilePath, [switch]$Append,
              [Parameter(ValueFromPipeline=$true)]$InputObject)
        process { if (-not $FilePath) { throw 'no log path' }; $global:Logs += $FilePath; $InputObject }
    }
    $caught = $false
    try { & $ScriptPath -Attempt 2 -PlanOnly:($Mode -eq 'plan') | Out-Null } catch { $caught = $true }
    $execs = @($global:Calls | Where-Object { $_ -contains '--execute' }).Count
    if ($Mode -eq 'success') {
        if ($caught -or $execs -ne 1 -or $global:Prompts -ne 1 -or $global:Logs.Count -eq 0) { throw 'success failed' }
        if (@($global:Logs | Sort-Object -Unique).Count -ne 1) { throw 'log path inconsistent' }
        $run = @($global:Calls | Where-Object { $_ -contains '--execute' })[0]
        if ($run[$run.IndexOf('--output-dir') + 1] -notmatch 'attempt02$') { throw 'wrong attempt path' }
        if ($Condition -eq 'B400' -and $run[$run.IndexOf('--output-dir') + 1] -notmatch '2026-09-14_b_frozen_s0_b400_t010_attempt02$') { throw 'wrong B400 output identity' }
        if ($run -notcontains '--preflight-confirmed') { throw 'missing confirmed preflight flag' }
    } else {
        if ($execs -ne 0) { throw 'benchmark launched despite guard' }
        if (($Mode -eq 'plan') -eq $caught) { throw 'wrong guard result' }
        if ($Mode -eq 'workload_blocked' -and $global:Prompts -ne 0) { throw 'confirmation before workload check' }
    }
    foreach ($call in @($global:Calls | Where-Object { $_ -contains 'ssl_hdc_pilot.scripts.benchmark_study' })) {
        if ($call[$call.IndexOf('--seed') + 1] -ne '0') { throw 'wrong seed' }
        if ($call[$call.IndexOf('--experiment') + 1] -ne $ExpectedExperiment) { throw 'wrong experiment' }
        if ($call[$call.IndexOf('--budget') + 1] -ne $ExpectedBudget) { throw 'wrong budget' }
        if ($Condition -eq 'B400') {
            if ($call -notcontains '--threshold' -or $call[$call.IndexOf('--threshold') + 1] -ne '0.10') { throw 'wrong B threshold' }
        } elseif ($call -contains '--threshold') { throw 'A cannot specify a threshold' }
    }
    Write-Output 'PASS launcher harness'
    '''
    prefix = ("$Mode = '" + mode + "'; $ScriptPath = '" + str(path).replace("'", "''") + "';\n"
              + f"$Condition = '{condition}'; $ExpectedExperiment = '{condition[0]}'; $ExpectedBudget = '{condition[1:]}';\n")
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", prefix + harness],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS launcher harness" in result.stdout


@pytest.mark.parametrize("blocked", [False, True])
def test_read_only_workload_cli_reports_status_without_learner(monkeypatch, capsys, blocked):
    def check():
        if blocked:
            raise RuntimeError("synthetic competing workloads")
        return {"accepted_desktop_baseline": [], "blocked_processes": [], "other_gpu_compute": []}
    monkeypatch.setattr(environment, "contamination_check", check)
    assert environment.preflight_main() == int(blocked)
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == ("BLOCKED" if blocked else "PASS")
