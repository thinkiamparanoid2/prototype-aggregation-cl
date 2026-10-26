"""Synthetic Windows inventory checks; never start Settings or a benchmark."""
import os
from types import SimpleNamespace

import pytest

from ssl_hdc_pilot.scripts import benchmark_environment as environment


SETTINGS = r"C:\Windows\ImmersiveControlPanel\SystemSettings.exe"
pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows workload guard")


def test_settings_can_reappear_with_new_pid_without_aborting(monkeypatch):
    for pid in (None, 999991, 999992, None):
        def inventory(command, **kwargs):
            if command[0] == "tasklist":
                return SimpleNamespace(stdout='"System","4"\n' +
                    (f'"SystemSettings.exe","{pid}"\n' if pid else ''))
            return SimpleNamespace(stdout=f'{pid}, {SETTINGS}\n' if pid else '')
        monkeypatch.setattr(environment.subprocess, "run", inventory)
        result = environment.contamination_check()
        assert result["other_gpu_compute"] == []
        assert result["blocked_processes"] == []
        assert result["accepted_desktop_baseline"] == ([{"pid": pid, "name": SETTINGS}] if pid else [])
        assert result["interference_measured"] is False
        assert "2026-09-14" in result["desktop_baseline_policy"]


@pytest.mark.parametrize("path", [SETTINGS, SETTINGS.upper(), SETTINGS.replace("\\", "/")])
def test_settings_exact_path_normalization_is_accepted(monkeypatch, path):
    monkeypatch.setattr(environment.subprocess, "run", lambda command, **kwargs:
        SimpleNamespace(stdout='"System","4"\n' if command[0] == "tasklist" else f'999991, {path}\n'))
    assert environment.contamination_check()["accepted_desktop_baseline"] == [{"pid": 999991, "name": path}]


@pytest.mark.parametrize("path", [r"C:\Temp\SystemSettings.exe", r"C:\Windows\unknown.exe",
    r"C:\Windows\ImmersiveControlPanel\other.exe", r"C:\Temp\python.exe"])
def test_settings_allowance_does_not_hide_other_gpu_work(monkeypatch, path):
    monkeypatch.setattr(environment.subprocess, "run", lambda command, **kwargs:
        SimpleNamespace(stdout='"System","4"\n' if command[0] == "tasklist" else
                        f'999991, {SETTINGS}\n999992, {path}\n'))
    if path.endswith("python.exe"):
        with pytest.raises(RuntimeError, match="competing workloads"):
            environment.contamination_check()
    else:
        result = environment.contamination_check()
        assert result["unclassified_gpu_processes"] == [{"pid": 999992, "name": path}]
        assert result["accepted_desktop_baseline"] == [{"pid": 999991, "name": SETTINGS}]


@pytest.mark.parametrize("name", ["python.exe", "obs64.exe", "msedge.exe", "overwolf.exe"])
def test_settings_allowance_does_not_hide_blocked_apps(monkeypatch, name):
    monkeypatch.setattr(environment.subprocess, "run", lambda command, **kwargs:
        SimpleNamespace(stdout=f'"{name}","999992"\n' if command[0] == "tasklist" else
                        f'999991, {SETTINGS}\n'))
    with pytest.raises(RuntimeError, match="competing workloads"):
        environment.contamination_check()
