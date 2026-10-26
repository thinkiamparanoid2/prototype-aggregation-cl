"""Read-only workload checks and separate-pass memory sampling on Windows."""
from __future__ import annotations

import csv
import ctypes
from ctypes import wintypes
import io
import json
import ntpath
import os
import subprocess
import threading
from pathlib import Path

import torch


BLOCKED_NAMES = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "obs64.exe",
    "valorant-win64-shipping.exe", "league of legends.exe", "cs2.exe", "fortniteclient-win64-shipping.exe",
    "nvidia overlay.exe", "gamebar.exe", "python.exe", "pythonw.exe",
    "overwolf.exe", "overwolfbrowser.exe"}


# Approved current-machine desktop baseline (2026-09-13, amended 2026-09-14), not a name-only
# allowlist or a claim these processes consume no GPU resources. Path/version
# changes are unclassified warnings, not automatic baseline approval.
DESKTOP_BASELINE_PATHS = frozenset(ntpath.normcase(ntpath.normpath(path)) for path in (
    r"C:\Windows\explorer.exe",
    r"C:\Windows\System32\dwm.exe",
    r"C:\Windows\System32\ShellHost.exe",
    r"C:\Windows\System32\ApplicationFrameHost.exe",
    r"C:\Windows\ImmersiveControlPanel\SystemSettings.exe",
    r"C:\Windows\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\CrossDeviceResume.exe",
    r"C:\Windows\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\SearchHost.exe",
    r"C:\Windows\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TextInputHost.exe",
    r"C:\Windows\SystemApps\Microsoft.Windows.StartMenuExperienceHost_cw5n1h2txyewy\StartMenuExperienceHost.exe",
    r"C:\Windows\SystemApps\ShellExperienceHost_cw5n1h2txyewy\ShellExperienceHost.exe",
    r"G:\newshit\Microsoft VS Code\Code.exe",
    r"G:\WindowsApps\Microsoft.WindowsTerminal_1.24.11911.0_x64__8wekyb3d8bbwe\WindowsTerminal.exe",
    r"C:\Program Files\WindowsApps\Microsoft.WindowsTerminal_1.24.11911.0_x64__8wekyb3d8bbwe\WindowsTerminal.exe",
))


SEARCH_WEBVIEW_PATH = ntpath.normcase(r"C:\Program Files (x86)\Microsoft\EdgeWebView\Application\152.0.4191.66\msedgewebview2.exe")
SEARCH_HOST_PATH = ntpath.normcase(r"C:\Windows\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\SearchHost.exe")


def _search_process_snapshot() -> dict:
    command = r'''$ErrorActionPreference='Stop'; Get-CimInstance Win32_Process -Filter "Name='msedgewebview2.exe' OR Name='SearchHost.exe'" | ForEach-Object { [pscustomobject]@{pid=[int]$_.ProcessId; parent_pid=[int]$_.ParentProcessId; path=[string]$_.ExecutablePath; started_ticks=$_.CreationDate.ToUniversalTime().Ticks} } | ConvertTo-Json -Compress'''
    try:
        output = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                capture_output=True, text=True, check=True, timeout=30).stdout
        rows = json.loads(output)
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list):
            raise ValueError("missing ownership records")
        records = {}
        for row in rows:
            if (not isinstance(row, dict) or type(row.get("pid")) is not int or row["pid"] <= 0
                    or type(row.get("parent_pid")) is not int or row["parent_pid"] < 0
                    or type(row.get("started_ticks")) is not int or row["started_ticks"] <= 0
                    or not isinstance(row.get("path"), str) or not row["path"]
                    or row["pid"] in records):
                raise ValueError("invalid ownership record")
            records[row["pid"]] = row
        return records
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("cannot interpret Windows Search ownership inventory") from error


def _search_owner_chain(pid: int, records: dict) -> list:
    chain, seen = [], set()
    while pid in records and pid not in seen and len(chain) < 16:
        seen.add(pid)
        node = records[pid]
        path = ntpath.normcase(ntpath.normpath(node["path"]))
        if chain and node["started_ticks"] > chain[-1]["started_ticks"]:
            return []  # Parent PID was reused after the child was created.
        if path == SEARCH_HOST_PATH and chain:
            return chain + [node]
        if path != SEARCH_WEBVIEW_PATH:
            return []
        chain.append(node)
        pid = node["parent_pid"]
    return []


def contamination_check() -> dict:
    if os.name != "nt":
        raise RuntimeError("real development benchmark guard requires Windows")
    tasks = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, check=True, timeout=30)
    blocked = []
    task_rows = [row for row in csv.reader(io.StringIO(tasks.stdout)) if row]
    if not task_rows:
        raise RuntimeError("cannot interpret empty Windows process inventory")
    for row in task_rows:
        try:
            if len(row) < 2 or not row[0].strip():
                raise ValueError("missing process name/PID")
            pid = int(row[1].strip())
            if pid < 0:
                raise ValueError("negative PID")
        except ValueError as exc:
            raise RuntimeError("cannot interpret Windows process inventory") from exc
        # Windows venv redirectors may leave a waiting python parent process.
        # GPU-compute inventory below still rejects an actively computing parent.
        if row[0].lower() in BLOCKED_NAMES and pid not in {os.getpid(), os.getppid()}:
            blocked.append({"name": row[0], "pid": pid})
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True, timeout=30)
    other_compute, accepted_desktop, unclassified = [], [], []
    classification_warnings = []
    search_records = None
    for row in csv.reader(io.StringIO(gpu.stdout)):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        try:
            if len(row) != 2 or not row[1].strip():
                raise ValueError("missing GPU process path or unexpected fields")
            pid = int(row[0].strip())
            if pid <= 0:
                raise ValueError("GPU PID must be positive")
        except ValueError as exc:
            raise RuntimeError("cannot interpret GPU process inventory") from exc
        if pid != os.getpid():
            name = row[1].strip()
            entry = {"pid": pid, "name": name}
            normalized = ntpath.normcase(ntpath.normpath(name))
            if ntpath.basename(normalized) in BLOCKED_NAMES:
                other_compute.append(entry)
            elif normalized in DESKTOP_BASELINE_PATHS:
                accepted_desktop.append(entry)
            elif normalized == SEARCH_WEBVIEW_PATH:
                if search_records is None:
                    try:
                        search_records = _search_process_snapshot()
                    except RuntimeError as error:
                        # Ownership is optional classification, not the primary inventory.
                        # An unreadable tasklist/GPU inventory still aborts above.
                        search_records = {}
                        classification_warnings.append(str(error))
                chain = _search_owner_chain(pid, search_records)
                if chain:
                    accepted_desktop.append({**entry, "verified_owner_chain": chain})
                else:
                    unclassified.append(entry)
            else:
                unclassified.append(entry)
    result = {"blocked_processes": blocked, "other_gpu_compute": other_compute,
              "accepted_desktop_baseline": accepted_desktop,
              "unclassified_gpu_processes": unclassified,
              "classification_warnings": classification_warnings,
              "desktop_baseline_policy": "2026-09-14 background-warning amendment; exact-path baseline; known workload exclusions; unclassified GPU entries warn",
              "interference_measured": False,
              "scope": "shared-desktop boundary inventories, not measured interference; unclassified activity is retained as warnings, not certified idle or safe; no isolated-machine claim"}
    if blocked or other_compute:
        raise RuntimeError(f"competing workloads detected; preserve attempt: {result}")
    return result


def summarize_workload_checks(checks: list, *, smoke: bool) -> dict:
    """Qualify the entire attempt, including warnings that disappeared later."""
    warnings = [{"phase": check["phase"],
                 "unclassified_gpu_processes": check["result"].get("unclassified_gpu_processes", []),
                 "classification_warnings": check["result"].get("classification_warnings", [])}
                for check in checks if (check["result"].get("unclassified_gpu_processes")
                                       or check["result"].get("classification_warnings"))]
    qualification = ("synthetic_no_machine_claim" if smoke else
                     "shared_desktop_with_unclassified_activity" if warnings else
                     "shared_desktop_recorded_baseline")
    return {"qualification": qualification, "warning_boundary_count": len(warnings),
            "warning_boundaries": warnings, "isolated_machine": False,
            "interference_measured": False,
            "scope": "all completed blocks retained; integrity PASS is not timing cleanliness; boundary snapshots cannot establish idle activity or exclude between-check interference"}


def preflight_main() -> int:
    """Read-only operator check; no learner, training, timing or attempt creation."""
    try:
        result = contamination_check()
        status = ("PASS_WITH_WARNINGS" if result.get("unclassified_gpu_processes")
                  or result.get("classification_warnings") else "PASS")
        print(json.dumps({"status": status, **result}, indent=2))
        return 0
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error),
                          "scope": "read-only workload preflight; no attempt created"}, indent=2))
        return 1


class _Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]


def working_set_bytes() -> int:
    if os.name != "nt":
        raise RuntimeError("Windows working-set sampling required")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    query = psapi.GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.WorkingSetSize)


def memory_pass(operation, device, *, sample=working_set_bytes, interval=.005):
    """Sampler overhead belongs only to this pass; working-set peak is sampled."""
    values, errors = [sample()], []
    stop = threading.Event()
    def observe():
        try:
            while not stop.wait(interval):
                values.append(sample())
        except Exception as error:
            errors.append(error)
    cuda = torch.device(device).type == "cuda"
    initial_cuda = {}
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        initial_cuda = {"cuda_initial_allocated_bytes": torch.cuda.memory_allocated(device),
                        "cuda_initial_reserved_bytes": torch.cuda.memory_reserved(device)}
    thread = threading.Thread(target=observe, daemon=True)
    thread.start()
    try:
        result = operation()
        if cuda:
            torch.cuda.synchronize(device)
    finally:
        stop.set()
        thread.join(timeout=5)
    if errors:
        raise errors[0]
    if thread.is_alive():
        raise RuntimeError("memory sampler failed to stop")
    values.append(sample())
    return result, {"working_set_initial_bytes": values[0], "sampled_working_set_peak_bytes": max(values),
        **initial_cuda,
        "sample_count": len(values), "sample_interval_seconds": interval,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
        "scope": "separate instrumented stream; process includes shared cache/runtime; sampled RSS may miss transients; CUDA counters exclude non-PyTorch allocations"}


class _FileStandardInfo(ctypes.Structure):
    _fields_ = [("AllocationSize", ctypes.c_longlong), ("EndOfFile", ctypes.c_longlong),
                ("NumberOfLinks", wintypes.DWORD), ("DeletePending", ctypes.c_ubyte),
                ("Directory", ctypes.c_ubyte)]


def allocated_file_bytes(path: Path) -> int:
    if os.name != "nt":
        return path.stat().st_blocks * 512
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    query = kernel.GetFileInformationByHandleEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    query.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = create(str(path.resolve()), 0x80, 7, None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = _FileStandardInfo()
        if not query(handle, 1, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(info.AllocationSize)
    finally:
        kernel.CloseHandle(handle)


def storage_report(paths, hashfile) -> dict:
    paths = list(paths)
    unique_files, contents = {}, {}
    for path in paths:
        stat = path.stat()
        unique_files[(stat.st_dev, stat.st_ino)] = path
        contents[hashfile(path)] = stat.st_size
    return {"logical_file_bytes_before_final_receipt": sum(p.stat().st_size for p in paths),
        "filesystem_allocated_unique_file_bytes_before_final_receipt": sum(allocated_file_bytes(p) for p in unique_files.values()),
        "content_dedup_potential_bytes_before_final_receipt": sum(contents.values()),
        "note": "filesystem allocation is measured with hard-link identities counted once; content dedup is potential only, no deduplication performed; excludes later summary/receipts"}


def machine_info() -> dict:
    if os.name != "nt":
        raise RuntimeError("Windows hardware inventory required")
    import winreg
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in ("physical", "available", "page", "available_page",
                                                  "virtual", "available_virtual", "extended")]
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    function = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
    function.argtypes = [ctypes.POINTER(MemoryStatus)]
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(memory)):
        raise ctypes.WinError(ctypes.get_last_error())
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    return {"cpu_model": cpu, "physical_ram_bytes": memory.physical,
            "available_ram_at_setup_bytes": memory.available, "gpu_driver_memory_inventory": gpu}


if __name__ == "__main__":
    raise SystemExit(preflight_main())
