"""Resumable batch runner for the 270-cell three-dataset replication.

Enumerates dataset x arm x learner x seed cells in the locked order
(PlantDoc -> PlantWild -> PlantVillage; frozen arms before adapted, in
vits16/vitb16/vitl16 order; hdc -> ncm -> ranpac; seeds 0-4) and launches
``python -m ssl_hdc_pilot.replication.run_cell`` as a subprocess for every
cell that is not yet complete. Safe to Ctrl-C and rerun: completed cells are
detected from their saved receipts and skipped; a directory left behind by a
crash is quarantined under .codex-tmp/failed_cells/ and retried, up to three
attempts total (tracked by counting quarantined copies on disk, so the cap
survives a fresh invocation of this script).

Checkpoint-step rule
=====================
Checkpoint steps are ``[0] + [(v, v + 1) for v in introductions_at] +
[stream_length]``, read from each dataset's
``ssl_hdc_pilot/replication/specs/<Dataset>_stream_feasibility.json``. This is
exactly the schedule the Gate 0 equivalence replay used and verified against
the six already-saved original-study B400 cells (six macro-F1/forgetting
matches, all deltas 0.0): see
``ssl_hdc_pilot/results/replication/gate0_claude_2026-09-17/run_gate0.py``,
whose ``steps=[0,195,196,390,391,585,586,780,781,976]`` is exactly this rule
applied to that spec's ``introductions_at=[195,390,585,780]`` and
``stream_length=976``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a declared project dependency
    psutil = None

REPO_ROOT = Path(__file__).resolve().parents[2]
REPLICATION_ROOT = REPO_ROOT / "ssl_hdc_pilot" / "replication"
RESULTS_ROOT = REPO_ROOT / "ssl_hdc_pilot" / "results" / "replication"
CELLS_ROOT = RESULTS_ROOT / "cells"
BATCH_LOG = CELLS_ROOT / "batch_log.jsonl"
FAILED_CELLS_ROOT = REPO_ROOT / ".codex-tmp" / "failed_cells"

DATASETS = ("PlantDoc", "PlantWild", "PlantVillage")
SIZES = ("vits16", "vitb16", "vitl16")
ARMS = tuple(f"frozen_{size}" for size in SIZES) + tuple(f"adapted_{size}" for size in SIZES)
LEARNERS = ("hdc", "ncm", "ranpac")
SEEDS = (0, 1, 2, 3, 4)

MAX_ATTEMPTS = 3
FREE_SPACE_GUARD_BYTES = 8 * (1024 ** 3)
TOTAL_THREADS = 12


class BatchStopped(RuntimeError):
    """Raised to unwind the whole batch on an unrecoverable condition."""


@dataclass(frozen=True)
class Cell:
    dataset: str
    arm: str
    learner: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.dataset}/{self.arm}/{self.learner}_s{self.seed}"

    @property
    def output_dir(self) -> Path:
        return CELLS_ROOT / self.dataset / self.arm / f"{self.learner}_s{self.seed}"

    @property
    def quarantine_prefix(self) -> str:
        return f"{self.dataset}__{self.arm}__{self.learner}_s{self.seed}__"


def enumerate_cells(*, dataset: str | None = None, arm: str | None = None,
                     learner: str | None = None) -> list[Cell]:
    cells: list[Cell] = []
    for d in DATASETS:
        if dataset and d != dataset:
            continue
        for a in ARMS:
            if arm and a != arm:
                continue
            for l in LEARNERS:
                if learner and l != learner:
                    continue
                for s in SEEDS:
                    cells.append(Cell(d, a, l, s))
    return cells


def dataset_paths(dataset: str) -> dict[str, Path]:
    return {
        "train_csv": REPLICATION_ROOT / "specs" / dataset / "stream_train.csv",
        "validation_csv": REPLICATION_ROOT / "data" / dataset / "splits" / "val.csv",
        "stream_spec": REPLICATION_ROOT / "specs" / f"{dataset}_stream_spec.json",
        "feasibility": REPLICATION_ROOT / "specs" / f"{dataset}_stream_feasibility.json",
    }


def embeddings_path(dataset: str, arm: str) -> Path:
    return REPLICATION_ROOT / "embeddings" / dataset / f"{arm}.npz"


def checkpoint_steps_from_schedule(introductions_at: list[int], stream_length: int) -> list[int]:
    """Pure form of the Gate-0-confirmed rule: 0, each introduction and
    introduction+1, then stream end -- see module docstring."""
    steps = {0, stream_length}
    for value in introductions_at:
        steps.add(value)
        steps.add(value + 1)
    return sorted(steps)


_CHECKPOINT_STEPS_CACHE: dict[str, list[int]] = {}


def checkpoint_steps_for_dataset(dataset: str) -> list[int]:
    if dataset not in _CHECKPOINT_STEPS_CACHE:
        feasibility = json.loads(dataset_paths(dataset)["feasibility"].read_text(encoding="utf-8"))
        _CHECKPOINT_STEPS_CACHE[dataset] = checkpoint_steps_from_schedule(
            feasibility["introductions_at"], feasibility["stream_length"])
    return _CHECKPOINT_STEPS_CACHE[dataset]


def cell_status(output_dir: Path) -> str:
    """'done', 'incomplete' (exists but not a verified finished receipt), or 'missing'."""
    if not output_dir.is_dir():
        return "missing"
    receipt_path = output_dir / "integrity_report.json"
    metrics_path = output_dir / "metrics.json"
    if not receipt_path.is_file() or not metrics_path.is_file():
        return "incomplete"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return "incomplete"
    if receipt.get("status") != "PASS" or not receipt.get("artifacts"):
        return "incomplete"
    for relative_path in receipt["artifacts"]:
        if not (output_dir / relative_path).is_file():
            return "incomplete"
    return "done"


def attempt_count(cell: Cell) -> int:
    if not FAILED_CELLS_ROOT.is_dir():
        return 0
    # Leftovers from an interrupted run (power cut, Ctrl+C) are tagged "__interrupted" and do
    # not count: only a cell whose own subprocess failed uses up an attempt.
    return sum(1 for p in FAILED_CELLS_ROOT.iterdir()
               if p.is_dir() and p.name.startswith(cell.quarantine_prefix)
               and not p.name.endswith(INTERRUPTED_SUFFIX))


INTERRUPTED_SUFFIX = "__interrupted"


def quarantine_incomplete(cell: Cell, *, interrupted: bool = False) -> None:
    output_dir = cell.output_dir
    if not output_dir.is_dir():
        return
    FAILED_CELLS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = INTERRUPTED_SUFFIX if interrupted else ""
    destination = FAILED_CELLS_ROOT / f"{cell.quarantine_prefix}{timestamp}{suffix}"
    shutil.move(str(output_dir), str(destination))


def free_space_ok(*, minimum_bytes: int = FREE_SPACE_GUARD_BYTES) -> tuple[bool, str]:
    drive = REPO_ROOT.resolve().anchor or "G:\\"
    usage = shutil.disk_usage(drive)
    return usage.free >= minimum_bytes, drive


def child_env(threads_per_worker: int) -> dict[str, str]:
    env = os.environ.copy()
    threads = str(max(1, threads_per_worker))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS"):
        env[var] = threads
    return env


def run_cell_subprocess(cell: Cell, *, python_exe: str, threads: int,
                         checkpoint_steps: list[int], log_dir: Path) -> dict[str, Any]:
    paths = dataset_paths(cell.dataset)
    output_dir = cell.output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python_exe, "-m", "ssl_hdc_pilot.replication.run_cell",
           "--dataset", cell.dataset, "--arm", cell.arm, "--learner", cell.learner,
           "--seed", str(cell.seed),
           "--embeddings", str(embeddings_path(cell.dataset, cell.arm)),
           "--train-split", str(paths["train_csv"]),
           "--validation-split", str(paths["validation_csv"]),
           "--stream-spec", str(paths["stream_spec"]),
           "--checkpoint-steps", *[str(step) for step in checkpoint_steps],
           "--output-dir", str(output_dir),
           "--execute", "--preflight-confirmed"]
    env = child_env(threads)
    stem = f"{cell.dataset}_{cell.arm}_{cell.learner}_s{cell.seed}"
    stdout_path = log_dir / f"{stem}.stdout.log"
    stderr_path = log_dir / f"{stem}.stderr.log"
    started = time.time()
    peak_rss_bytes = None
    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
        popen_cls = psutil.Popen if psutil is not None else subprocess.Popen
        proc = popen_cls(cmd, cwd=str(REPO_ROOT), env=env, stdout=out_f, stderr=err_f)
        if psutil is not None:
            peak_rss_bytes = 0
        while proc.poll() is None:
            if psutil is not None:
                try:
                    peak_rss_bytes = max(peak_rss_bytes, proc.memory_info().rss)
                except psutil.Error:
                    pass
            time.sleep(0.5)
        returncode = proc.wait()
    elapsed = time.time() - started
    return {"returncode": returncode, "elapsed_seconds": elapsed, "peak_rss_bytes": peak_rss_bytes,
            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}


def process_cell(cell: Cell, *, python_exe: str, threads: int, require_adapted: bool,
                  stop_event: Event, log_dir: Path) -> dict[str, Any]:
    emb = embeddings_path(cell.dataset, cell.arm)
    if not emb.is_file():
        if cell.arm.startswith("adapted") and not require_adapted:
            return {"cell": cell.name, "status": "skipped",
                    "reason": f"adapted embeddings not found yet: {emb}"}
        raise BatchStopped(f"{cell.name}: required embeddings not found: {emb}")

    while True:
        if stop_event.is_set():
            return {"cell": cell.name, "status": "aborted"}
        status = cell_status(cell.output_dir)
        if status == "done":
            return {"cell": cell.name, "status": "already_done"}
        if status == "incomplete":
            quarantine_incomplete(cell, interrupted=True)
        attempts_so_far = attempt_count(cell)
        if attempts_so_far >= MAX_ATTEMPTS:
            raise BatchStopped(
                f"{cell.name}: attempt cap ({MAX_ATTEMPTS}) reached; inspect "
                f"{FAILED_CELLS_ROOT}/{cell.quarantine_prefix}* before rerunning")
        ok, drive = free_space_ok()
        if not ok:
            raise BatchStopped(f"below 8 GiB free space guard on {drive}; stopping before {cell.name}")

        checkpoint_steps = checkpoint_steps_for_dataset(cell.dataset)
        result = run_cell_subprocess(cell, python_exe=python_exe, threads=threads,
                                      checkpoint_steps=checkpoint_steps, log_dir=log_dir)
        if result["returncode"] == 0 and cell_status(cell.output_dir) == "done":
            return {"cell": cell.name, "status": "done", **result}
        quarantine_incomplete(cell)
        # loop back around; attempt_count() will be one higher next pass


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def run_batch(cells: list[Cell], *, workers: int, require_adapted: bool, python_exe: str) -> None:
    CELLS_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = CELLS_ROOT / "logs"
    threads_per_worker = max(1, TOTAL_THREADS // max(1, workers))
    stop_event = Event()
    lock = Lock()
    total = len(cells)
    completed = 0
    durations: list[float] = []
    start_time = time.time()
    errors: list[BaseException] = []

    BATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_handle = BATCH_LOG.open("a", encoding="utf-8")

    def worker_body(cell: Cell) -> None:
        nonlocal completed
        if stop_event.is_set():
            return
        try:
            outcome = process_cell(cell, python_exe=python_exe, threads=threads_per_worker,
                                    require_adapted=require_adapted, stop_event=stop_event, log_dir=log_dir)
        except Exception as exc:  # any unrecoverable condition stops the batch, never hidden
            with lock:
                errors.append(exc)
                stop_event.set()
            return
        with lock:
            completed += 1
            entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **outcome}
            log_handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
            log_handle.flush()
            elapsed = time.time() - start_time
            if outcome.get("elapsed_seconds") is not None:
                durations.append(outcome["elapsed_seconds"])
            if durations:
                mean_duration = sum(durations) / len(durations)
                remaining = max(0, total - completed)
                eta = _format_duration(mean_duration * remaining / max(1, workers))
            else:
                eta = "unknown"
            print(f"[{completed}/{total}] {cell.name}: {outcome['status']} "
                  f"(elapsed {_format_duration(elapsed)}, ETA {eta})", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(worker_body, cell) for cell in cells]
            for future in futures:
                future.result()
    finally:
        log_handle.close()

    if errors:
        raise errors[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--require-adapted", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--learner", choices=LEARNERS)
    parser.add_argument("--plan", action="store_true", help="list cells and their status; run nothing")
    parser.add_argument("--python", default=sys.executable, help="python executable for child cells")
    args = parser.parse_args()

    cells = enumerate_cells(dataset=args.dataset, arm=args.arm, learner=args.learner)
    if args.limit is not None:
        cells = cells[: args.limit]

    if args.plan:
        for index, cell in enumerate(cells, start=1):
            status = cell_status(cell.output_dir)
            emb = embeddings_path(cell.dataset, cell.arm)
            note = ""
            if not emb.is_file():
                note = " [adapted embeddings missing]" if cell.arm.startswith("adapted") else " [MISSING EMBEDDINGS]"
            print(f"{index} {cell.name} status={status}{note}")
        print(f"PLAN cells={len(cells)} require_adapted={args.require_adapted} workers={args.workers}")
        return

    run_batch(cells, workers=args.workers, require_adapted=args.require_adapted, python_exe=args.python)


if __name__ == "__main__":
    main()
