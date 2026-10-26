"""Synthetic regressions for recurring background apps; no real benchmark."""
import json
import os
from types import SimpleNamespace

import pytest

from ssl_hdc_pilot.scripts import benchmark_environment as environment


BACKGROUND = [
    r"C:\Users\thatd\AppData\Local\GitHubDesktop\app-3.6.5\GitHubDesktop.exe",
    r"C:\Program Files\WindowsApps\SpotifyAB.SpotifyMusic_1.299.317.0_x64__zpdnekdrzrea0\SpotifyXboxGamebarWebView.exe",
    r"C:\Program Files\WindowsApps\Microsoft.Edge.GameAssist_1.0.4019.0_x64__8wekyb3d8bbwe\EdgeGameAssist.exe",
]
pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows workload guard")


def inventory(monkeypatch, entries, tasks='"System","4"\n'):
    monkeypatch.setattr(environment.subprocess, "run", lambda command, **kwargs:
        SimpleNamespace(stdout=tasks if command[0] == "tasklist" else
                        "".join(f'{entry["pid"]}, {entry["name"]}\n' for entry in entries)))


def test_reported_background_apps_warn_without_aborting_or_becoming_baseline(monkeypatch):
    for base_pid in (990000, 991000):
        entries = [{"pid": base_pid + i, "name": path} for i, path in enumerate(BACKGROUND)]
        inventory(monkeypatch, entries)
        result = environment.contamination_check()
        assert result["unclassified_gpu_processes"] == entries
        assert result["accepted_desktop_baseline"] == []
        assert result["other_gpu_compute"] == result["blocked_processes"] == []
        assert result["interference_measured"] is False


def test_warning_preflight_returns_success_with_visible_qualification(monkeypatch, capsys):
    inventory(monkeypatch, [{"pid": 990000, "name": BACKGROUND[0]}])
    assert environment.preflight_main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert report["unclassified_gpu_processes"][0]["name"] == BACKGROUND[0]


@pytest.mark.parametrize("name", sorted(environment.BLOCKED_NAMES))
def test_unknown_background_does_not_hide_known_gpu_workload(monkeypatch, name):
    inventory(monkeypatch, [{"pid": 990000, "name": BACKGROUND[0]},
                           {"pid": 990001, "name": "C:\\Apps\\" + name.upper()}])
    with pytest.raises(RuntimeError, match="competing workloads"):
        environment.contamination_check()


def test_parent_python_is_still_blocked_when_gpu_listed(monkeypatch):
    inventory(monkeypatch, [{"pid": os.getppid(), "name": r"C:\Python\python.exe"}])
    with pytest.raises(RuntimeError, match="competing workloads"):
        environment.contamination_check()


def test_failed_optional_ownership_becomes_unclassified_not_verified(monkeypatch):
    entry = {"pid": 990000, "name": environment.SEARCH_WEBVIEW_PATH}
    inventory(monkeypatch, [entry])
    def unavailable():
        raise RuntimeError("cannot interpret Windows Search ownership inventory")
    monkeypatch.setattr(environment, "_search_process_snapshot", unavailable)
    result = environment.contamination_check()
    assert result["accepted_desktop_baseline"] == []
    assert result["unclassified_gpu_processes"] == [entry]
    assert result["classification_warnings"]


def test_qualification_keeps_transient_warning_boundaries():
    observed = [{"pid": 990000, "name": BACKGROUND[0]}]
    checks = [{"phase": phase, "result": {"unclassified_gpu_processes": entries,
               "classification_warnings": []}} for phase, entries in
              [("before/0/hdc/throughput", []), ("after/0/hdc/throughput", observed),
               ("after/1/ncm/latency", [])]]
    result = environment.summarize_workload_checks(checks, smoke=False)
    assert result["qualification"] == "shared_desktop_with_unclassified_activity"
    assert result["warning_boundary_count"] == 1
    assert result["warning_boundaries"][0]["phase"] == checks[1]["phase"]
    assert result["warning_boundaries"][0]["unclassified_gpu_processes"] == observed
    assert result["interference_measured"] is False


@pytest.mark.parametrize("smoke,qualification", [(True, "synthetic_no_machine_claim"),
    (False, "shared_desktop_recorded_baseline")])
def test_no_warning_is_not_an_isolation_claim(smoke, qualification):
    result = environment.summarize_workload_checks([], smoke=smoke)
    assert result["qualification"] == qualification
    assert result["isolated_machine"] is False
