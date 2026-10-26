"""Reconstruct one historical training-stream cell; default CLI action is plan only.

No encoder training, test scoring, budget-policy changes, or whole-grid launch.
Use a fresh output directory. PASS denotes observed history/summary fidelity,
not proof of original input-file identity when old artifacts lack input hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import subprocess
import time
from dataclasses import replace
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

from ssl_hdc_pilot.scripts import stream_runner as runner
from ssl_hdc_pilot.scripts.learner_checkpoint import save_learner_checkpoint
from ssl_hdc_pilot.scripts.build_stream_spec import _build_holdback_manifests


CURRENT_SPEC_SHA = "62b545d31d592735be4404451bddd7f603576d127351f337e1e34c8fd04604cc"
EARLIER_SPEC_SHA = "30447621e0d3701f36a152fb37716f54b6a31cab9f26b118a07c732a6e365812"
DEFAULT_ATOL = 1e-6
PROTOCOL_PATH = runner.REPO_ROOT / "docs/TODO/HDC_CORE_EVALUATION_PROTOCOL_DRAFT_2026-09-09.md"
HISTORY_FIELDS = {
    "step", "image_id", "true_label", "prediction", "confidence", "top_probability",
    "label_requested", "label_request_denied_budget", "class_registered",
    "update_wall_time_seconds", "correct", "rolling_per_class_accuracy",
}
SUMMARY_IGNORED = {
    "embedding_npz", "trusted_split_csv", "stream_spec",
    "update_compute_seconds.mean", "update_compute_seconds.median",
    "update_compute_seconds.p95", "update_compute_seconds.total",
}


def recover_source_spec(raw: bytes, expected_sha256: str, learner: str) -> tuple[bytes, str]:
    """Recover only an exact hash match; never accept an arbitrary spec mismatch."""
    actual = hashlib.sha256(raw).hexdigest()
    if actual == expected_sha256:
        return raw, "exact_source_spec"
    if (actual == CURRENT_SPEC_SHA and expected_sha256 == EARLIER_SPEC_SHA
            and learner in {"hdc", "naive", "replay"}):
        recovered = raw.replace(b'"ewc_decay": 0.9', b'"ewc_decay": 1.0', 1)
        if hashlib.sha256(recovered).hexdigest() == expected_sha256:
            return recovered, "exact_hash_recovery_of_pre_ewc_decay_amendment"
    raise ValueError("source stream-spec hash mismatch: recover/review the historical spec; no bypass")


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read_history(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != HISTORY_FIELDS or len(reader.fieldnames or []) != len(HISTORY_FIELDS):
            raise ValueError("history columns do not match the registered runner")
        rows = list(reader)
    for index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise ValueError("malformed history row")
        row["step"] = int(row["step"])
        if row["step"] != index:
            raise ValueError("history steps must be consecutive and zero-based")
        for field in ("confidence", "top_probability", "update_wall_time_seconds"):
            row[field] = float(row[field])
            if not math.isfinite(row[field]):
                raise ValueError(f"non-finite history field: {field}")
        for field in ("label_requested", "label_request_denied_budget", "class_registered", "correct"):
            if row[field] not in ("True", "False"):
                raise ValueError(f"invalid history boolean: {field}")
            row[field] = row[field] == "True"
        row["prediction"] = row["prediction"] or None
        row["rolling_per_class_accuracy"] = json.loads(row["rolling_per_class_accuracy"])
    return rows


def compare_values(expected: Any, actual: Any, *, atol: float = DEFAULT_ATOL,
                   ignored: set[str] | None = None) -> dict[str, Any]:
    """Exact structure/discrete values; absolute tolerance only for finite floats."""
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    ignored = ignored or set()
    differences: list[dict[str, Any]] = []
    mismatch_count = 0

    def record(path: str, left: Any, right: Any) -> None:
        nonlocal mismatch_count
        mismatch_count += 1
        if len(differences) < 100:
            differences.append({"path": path, "expected": left, "actual": right})

    def walk(left: Any, right: Any, path: str) -> None:
        if path in ignored:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                child = f"{path}.{key}" if path else str(key)
                if child in ignored:
                    continue
                if key not in left or key not in right:
                    record(child, {"present": key in left}, {"present": key in right})
                else:
                    walk(left[key], right[key], child)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                record(path + ".length", len(left), len(right))
            for index, (a, b) in enumerate(zip(left, right)):
                walk(a, b, f"{path}[{index}]")
        elif type(left) is float and type(right) in (float, int):
            if not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > atol:
                record(path, str(left) if not math.isfinite(left) else left,
                       str(right) if not math.isfinite(right) else right)
        elif type(left) is not type(right) or left != right:
            record(path, left, right)

    walk(expected, actual, "")
    return {"mismatch_count": mismatch_count, "first_mismatches": differences}


def _prepare_source(*, source_summary: Path, embeddings_path: Path, split_csv: Path,
                    stream_spec_path: Path, smoke_mode: bool) -> tuple[dict, list, Any, bytes, dict]:
    canonical_train = runner.DEFAULT_APPROVED_SPLIT_DIR / "train.csv"
    if not smoke_mode and split_csv.resolve() != canonical_train.resolve():
        raise RuntimeError("official reconstruction requires canonical approved Option-A train; rejected before opening inputs")
    source = _read_json(source_summary)
    learner = source["learner"]
    if learner not in runner.LEARNER_NAMES:
        raise ValueError("unknown historical learner")
    if type(source.get("smoke")) is not bool or source["smoke"] != smoke_mode:
        raise ValueError("historical smoke mode mismatch")
    tag = source["run_tag"]
    if not isinstance(tag, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", tag) is None:
        raise ValueError("invalid historical run tag")
    if source_summary.name != f"{tag}_summary.json":
        raise ValueError("source summary filename does not match run tag")
    source_history = source_summary.with_name(f"{tag}_history.csv")
    history = _read_history(source_history)
    spec = runner.load_stream_spec(stream_spec_path, smoke_mode=smoke_mode)
    raw = stream_spec_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec.sha256:
        raise ValueError("stream specification changed during inspection")
    if not smoke_mode and spec.sha256 != CURRENT_SPEC_SHA:
        raise ValueError("official reconstruction requires the locked current spec bytes")
    recovered, recovery = recover_source_spec(raw, source["stream_spec_sha256"], learner)
    historical_config = json.loads(recovered)["baseline"]
    spec = replace(spec, sha256=source["stream_spec_sha256"], baseline_config=replace(
        spec.baseline_config, ewc_decay=historical_config["ewc_decay"]))
    seed, threshold = source["stream_seed"], source["confidence_threshold"]
    if type(seed) is not int or seed not in spec.stream_order_seeds:
        raise ValueError("historical seed is not registered")
    if type(threshold) not in (float, int) or threshold not in spec.confidence_thresholds:
        raise ValueError("historical threshold is not registered")
    expected_tag = (f"{spec.tag}__{runner._safe_stem(embeddings_path.stem)}__stream_{seed}__"
                    f"{learner}__margin_{runner._threshold_tag(threshold)}")
    if tag != expected_tag:
        raise ValueError("historical run tag disagrees with inputs/learner/seed/threshold")
    rows = (runner._load_split_rows(split_csv, smoke_mode=True) if smoke_mode
            else runner.load_approved_train(runner.DEFAULT_APPROVED_SPLIT_DIR))
    if not smoke_mode:
        # The locked bytes fix rarity/order. Do not rederive rarity from val/test.
        if len(rows) != 3396:
            raise ValueError("official reconstruction requires 3396 approved training rows")
        manifests = _build_holdback_manifests(rows, spec.initially_known_classes)
        if any(frozenset(manifests[str(s)]) != spec.held_back_image_ids_by_seed[s]
               for s in spec.stream_order_seeds):
            raise ValueError("training-only holdback derivation disagrees with locked spec")
        allowed = {"arm_f_vitb16_trusted_all.npz", "arm_d_vitb16_ep3_trusted_all.npz"}
        original_name = PureWindowsPath(source["embedding_npz"]).name
        if original_name not in allowed or embeddings_path.resolve() != (runner.REPO_ROOT / "ssl_hdc_pilot/results" / original_name).resolve():
            raise ValueError("official reconstruction requires the historical ViT-B embedding arm in this checkout")
    if type(source["stream_steps"]) is not int or len(history) != source["stream_steps"] or not history:
        raise ValueError("historical stream length mismatch")
    files = {"source_summary": source_summary, "source_history": source_history,
             "embeddings": embeddings_path, "split": split_csv, "current_stream_spec": stream_spec_path}
    manifest = {
        "run_tag": tag, "learner": learner, "stream_seed": seed, "confidence_threshold": threshold,
        "smoke": smoke_mode, "stream_steps": len(history),
        "source_stream_spec_sha256": source["stream_spec_sha256"],
        "current_stream_spec_sha256": hashlib.sha256(raw).hexdigest(), "spec_recovery": recovery,
        "inputs": {key: {"path": str(path.resolve()), "sha256": runner._file_sha256(path)}
                   for key, path in files.items()},
        "historical_input_hashes_available": False,
        "provenance_note": "Old summaries lack embedding/split content hashes; current hashes pin reconstruction inputs, not retrospective original identity.",
    }
    return source, history, spec, recovered, manifest


def plan_reconstruction(**kwargs: Any) -> dict[str, Any]:
    """Read-only plan: validate metadata and hash inputs; no learner execution."""
    return _prepare_source(**kwargs)[-1]


def _git_provenance() -> dict[str, Any]:
    def git(*args: str) -> bytes:
        return subprocess.run(["rtk", "proxy", "git", "--no-optional-locks", *args],
                              cwd=runner.REPO_ROOT, check=True, capture_output=True,
                              timeout=10).stdout
    try:
        return {"head": git("rev-parse", "HEAD").decode().strip(),
                "dirty": bool(git("status", "--porcelain")),
                "tracked_patch_sha256": hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest(),
                "untracked_code_coverage": "covered by per-file code_hashes, not git diff"}
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "note": "Git unavailable; use working-file code_hashes"}


def reconstruct_run(*, source_summary: Path, embeddings_path: Path, split_csv: Path,
                    stream_spec_path: Path, output_dir: Path, smoke_mode: bool = False,
                    preflight_confirmed: bool = False, clock: Callable[[], float] = time.perf_counter,
                    atol: float = DEFAULT_ATOL) -> dict[str, Any]:
    if not smoke_mode and not preflight_confirmed:
        raise RuntimeError("official reconstruction requires user launch and contamination preflight confirmation")
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite/reuse reconstruction directory")
    source, old_history, spec, recovered, manifest = _prepare_source(
        source_summary=Path(source_summary), embeddings_path=Path(embeddings_path),
        split_csv=Path(split_csv), stream_spec_path=Path(stream_spec_path), smoke_mode=smoke_mode)
    embeddings, input_dim = runner._load_embedding_map(Path(embeddings_path))
    if input_dim != source["embedding_dim"]:
        raise ValueError("historical embedding dimension mismatch")
    bootstrap, stream = runner._prepare_stream(
        split_rows=runner._load_split_rows(Path(split_csv), smoke_mode=smoke_mode),
        embedding_map=embeddings, spec=spec, stream_seed=source["stream_seed"])
    if [(s.image_id, s.label) for s in stream] != [(r["image_id"], r["true_label"]) for r in old_history]:
        raise ValueError("reconstructed stream IDs/order/labels disagree with historical history")
    code_hashes = runner._checkpoint_code_hashes()
    code_hashes["ssl_hdc_pilot/scripts/reconstruct_run.py"] = runner._file_sha256(Path(__file__))
    schedule = {"bootstrap": [[s.image_id, s.label] for s in bootstrap],
                "stream": [[s.image_id, s.label] for s in stream]}
    role = "synthetic_smoke" if smoke_mode else "trusted_train"
    manifest.update(atol=atol, rtol=0, code_hashes=code_hashes,
                    git=_git_provenance(), data_role=role,
                    protocol={"path": str(PROTOCOL_PATH), "sha256": runner._file_sha256(PROTOCOL_PATH),
                              "status": "draft_not_frozen"},
                    event_schedule_sha256=hashlib.sha256(json.dumps(
                        schedule, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    registered_configuration=json.loads(recovered),
                    environment={"python": platform.python_version(), "platform": platform.platform(),
                                 "torch": str(runner.torch.__version__), "numpy": runner.np.__version__,
                                 "torch_cuda_build": runner.torch.version.cuda, "requested_device": spec.device},
                    excluded_summary_fields=sorted(SUMMARY_IGNORED),
                    excluded_history_fields=["update_wall_time_seconds"],
                    historical_learner_configuration_absent="learner_configuration" not in source)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "reconstruction_manifest.json", manifest)
    with (output_dir / "source_stream_spec.json").open("xb") as handle:
        handle.write(recovered)
    tag = source["run_tag"]
    state_path = output_dir / f"{tag}_final_state.npz"

    def preserve(learner: Any, progress: dict[str, Any]) -> None:
        save_learner_checkpoint(state_path, learner, context={
            "run_tag": tag, "reconstruction_manifest": "reconstruction_manifest.json",
            "fidelity_requires_report": True, "progress": progress,
            "source_stream_spec_sha256": spec.sha256, "code_hashes": code_hashes,
            "bootstrap_image_ids": [sample.image_id for sample in bootstrap],
            "stream_image_ids": [sample.image_id for sample in stream],
        })

    predictions_path = output_dir / "final_training_predictions.json"

    def preserve_predictions(rows: list[dict[str, Any]]) -> None:
        _write_json(predictions_path, {"data_role": role, "rows": rows})

    history, scientific = runner._run_one(
        learner_name=source["learner"], threshold=source["confidence_threshold"], bootstrap=bootstrap,
        stream=stream, spec=spec, input_dim=input_dim, clock=clock, on_final_state=preserve,
        on_final_predictions=preserve_predictions)
    summary = {
        "run_tag": tag, "learner": source["learner"],
        "learner_configuration": runner._learner_configuration(source["learner"], spec),
        "confidence_definition": "top1 minus top2 cosine-similarity score margin",
        "confidence_threshold": source["confidence_threshold"], "embedding_dim": input_dim,
        "embedding_npz": str(embeddings_path), "trusted_split_csv": str(split_csv),
        "stream_spec": str(output_dir / "source_stream_spec.json"), "stream_spec_sha256": spec.sha256,
        "smoke": smoke_mode, "stream_seed": source["stream_seed"], "stream_steps": len(stream), **scientific,
    }
    history_path = output_dir / f"{tag}_history.csv"
    summary_path = output_dir / f"{tag}_summary.json"
    runner._write_history(history_path, history)
    _write_json(summary_path, summary)
    expected = dict(source)
    # Earlier heads had no configuration field; their current field is empty.
    if "learner_configuration" not in expected and summary["learner_configuration"] == {}:
        expected["learner_configuration"] = {}
    new_history = _read_history(history_path)
    for rows in (old_history, new_history):
        for row in rows:
            row.pop("update_wall_time_seconds")
    history_check = compare_values(old_history, new_history, atol=atol)
    summary_check = compare_values(expected, summary, atol=atol, ignored=SUMMARY_IGNORED)
    changed = [key for key, record in manifest["inputs"].items()
               if runner._file_sha256(Path(record["path"])) != record["sha256"]]
    code_changed = [path for path, digest in code_hashes.items()
                    if runner._file_sha256(runner.REPO_ROOT / path) != digest]
    if runner._file_sha256(PROTOCOL_PATH) != manifest["protocol"]["sha256"]:
        changed.append("protocol")
    report = {
        "status": "PASS" if not (history_check["mismatch_count"] or summary_check["mismatch_count"] or changed or code_changed) else "FAIL",
        "run_tag": tag, "stream_steps": len(stream), "history": history_check, "summary": summary_check,
        "inputs_changed_during_run": changed, "atol": atol, "rtol": 0,
        "code_changed_during_run": code_changed,
        "artifacts": {path.name: runner._file_sha256(path) for path in (
            state_path, history_path, summary_path, predictions_path,
            output_dir / "reconstruction_manifest.json", output_dir / "source_stream_spec.json")},
        "scope": "training-side observed-output fidelity only; no held-out scoring or benchmark",
    }
    _write_json(output_dir / "fidelity_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--trusted-split", type=Path, required=True)
    parser.add_argument("--stream-spec", type=Path, default=runner.DEFAULT_STREAM_SPEC)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true", help="otherwise inspect/hash inputs only")
    parser.add_argument("--preflight-confirmed", action="store_true")
    args = parser.parse_args()
    common = dict(source_summary=args.source_summary, embeddings_path=args.embeddings,
                  split_csv=args.trusted_split, stream_spec_path=args.stream_spec, smoke_mode=args.smoke)
    if not args.execute:
        print(json.dumps(plan_reconstruction(**common), indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        parser.error("--execute requires a fresh --output-dir")
    report = reconstruct_run(**common, output_dir=args.output_dir, preflight_confirmed=args.preflight_confirmed)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
