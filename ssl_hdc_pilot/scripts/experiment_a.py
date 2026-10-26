"""One development-only Experiment-A cell; default action plans, never trains.

No validation/test scoring, benchmark, intermediate snapshots, or whole-grid launch.
All experiments are user-launched. Synthetic tests may use --smoke.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

from ssl_hdc_pilot.scripts import stream_runner as runner
from ssl_hdc_pilot.scripts.build_stream_spec import _build_holdback_manifests
from ssl_hdc_pilot.scripts.learner_checkpoint import save_learner_checkpoint
from ssl_hdc_pilot.scripts.reconstruct_run import CURRENT_SPEC_SHA, PROTOCOL_PATH, _git_provenance, _write_json


ADDITIONAL_BUDGETS = (0, 25, 50, 100, 200, 400)
ABLATION_ARMS = frozenset({
    "arm_f_trusted_all.npz", "arm_f_vitb16_trusted_all.npz", "arm_f_vitl16_trusted_all.npz",
    "arm_d_vitb16_ep3_trusted_all.npz", "arm_d_vitl16_ep4_trusted_all.npz",
})
SCHEDULE_PREFIX = "hdc-shared-v1|20260909"


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def build_shared_label_schedule(image_ids: Sequence[str], *, stream_seed: int,
                                budget: int) -> dict[str, Any]:
    """Select by ID hash only; return selected events in original stream order."""
    ids = list(image_ids)
    if (not ids or any(type(x) is not str or not x or x != x.strip() for x in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("stream IDs must be unique nonempty literal strings without edge whitespace")
    if type(stream_seed) is not int or stream_seed < 0:
        raise ValueError("stream seed must be a nonnegative integer")
    if type(budget) is not int or not 0 <= budget <= len(ids):
        raise ValueError("budget must be an integer between zero and stream length")
    ranked = sorted(ids, key=lambda x: (
        hashlib.sha256(f"{SCHEDULE_PREFIX}|{stream_seed}|{x}".encode("utf-8")).digest(), x))
    selected = frozenset(ranked[:budget])
    schedule = {
        "schema_version": 1, "algorithm": SCHEDULE_PREFIX, "stream_seed": stream_seed,
        "additional_label_budget": budget, "stream_image_ids": ids,
        "ranked_ids_sha256": _json_hash(ranked),
        "selected_events": [{"step": step, "image_id": x} for step, x in enumerate(ids) if x in selected],
        "id_semantics": "literal UTF-8 strings; no numeric conversion or Unicode normalization",
    }
    return {**schedule, "schedule_sha256": _json_hash(schedule)}


ABLATION_HYPERVECTOR_DIMS = frozenset({1000, 2000, 10000})


def _prepare(*, embeddings_path: Path, split_csv: Path, stream_spec_path: Path,
             learner_name: str, stream_seed: int, budget: int, smoke_mode: bool,
             ablation_mode: bool = False, hypervector_dim: int | None = None) -> tuple:
    allowed_learners = runner.LEARNER_NAMES + (runner.ABLATION_LEARNER_NAMES if ablation_mode else ())
    if learner_name not in allowed_learners:
        raise ValueError("unknown learner")
    if hypervector_dim is not None:
        if not ablation_mode:
            raise ValueError("hypervector_dim override requires ablation_mode")
        if type(hypervector_dim) is bool or hypervector_dim not in ABLATION_HYPERVECTOR_DIMS:
            raise ValueError("hypervector_dim override must be one of {1000, 2000, 10000}")
    if type(stream_seed) is not int or type(budget) is not int or budget < 0:
        raise ValueError("seed and budget must be integers; budget must be nonnegative")
    if not smoke_mode:
        if split_csv.resolve() != (runner.DEFAULT_APPROVED_SPLIT_DIR / "train.csv").resolve():
            raise RuntimeError("Experiment A requires canonical approved train; rejected before opening inputs")
        allowed = {"arm_f_vitb16_trusted_all.npz", "arm_d_vitb16_ep3_trusted_all.npz"}
        permitted = ABLATION_ARMS if ablation_mode else allowed
        if (embeddings_path.name not in permitted or embeddings_path.resolve() !=
                (runner.REPO_ROOT / "ssl_hdc_pilot/results" / embeddings_path.name).resolve()):
            raise ValueError("Experiment A requires an existing canonical ViT-B embedding arm" if not ablation_mode else "ablation requires a registered trusted embedding arm")
        if budget not in ADDITIONAL_BUDGETS:
            raise ValueError("budget is outside the approved Experiment-A grid")
    # Pin all inputs before parsing/loading any of them.
    inputs = {name: {"path": str(path.resolve()), "sha256": runner._file_sha256(path)}
              for name, path in {"embeddings": embeddings_path, "train": split_csv,
                                 "base_stream_spec": stream_spec_path}.items()}
    spec = runner.load_stream_spec(stream_spec_path, smoke_mode=smoke_mode)
    raw = stream_spec_path.read_bytes()
    if (hashlib.sha256(raw).hexdigest() != spec.sha256
            or spec.sha256 != inputs["base_stream_spec"]["sha256"]):
        raise ValueError("base stream spec changed during inspection")
    if not smoke_mode and spec.sha256 != CURRENT_SPEC_SHA:
        raise ValueError("Experiment A requires the locked current base stream spec")
    if stream_seed not in spec.stream_order_seeds:
        raise ValueError("stream seed is not registered")
    rows = (runner._load_split_rows(split_csv, smoke_mode=True) if smoke_mode
            else runner.load_approved_train(runner.DEFAULT_APPROVED_SPLIT_DIR))
    if not smoke_mode:
        if len(rows) != 3396:
            raise ValueError("expected 3396 approved training rows")
        holdbacks = _build_holdback_manifests(rows, spec.initially_known_classes)
        if any(frozenset(holdbacks[str(seed)]) != spec.held_back_image_ids_by_seed[seed]
               for seed in spec.stream_order_seeds):
            raise ValueError("training-only holdback derivation disagrees with locked spec")
    embeddings, input_dim = runner._load_embedding_map(embeddings_path)
    bootstrap, stream = runner._prepare_stream(split_rows=rows, embedding_map=embeddings,
                                               spec=spec, stream_seed=stream_seed)
    expected_dimensions = {384, 768, 1024} if ablation_mode else {768}
    if not smoke_mode and (len(bootstrap), len(stream)) != (2420, 976) or (not smoke_mode and input_dim not in expected_dimensions):
        raise ValueError("official bootstrap/stream/dimension differs from the approved design")
    schedule = build_shared_label_schedule([s.image_id for s in stream], stream_seed=stream_seed, budget=budget)
    if any(runner._file_sha256(Path(record["path"])) != record["sha256"] for record in inputs.values()):
        raise ValueError("inputs changed during preparation; no learner was executed")
    return spec, raw, bootstrap, stream, input_dim, schedule, inputs


def plan_experiment_a(**kwargs: Any) -> dict[str, Any]:
    """Validate inputs and compute the label-blind schedule without learner execution."""
    spec, _, bootstrap, stream, _, schedule, inputs = _prepare(**kwargs)
    return {"experiment": "A", "stage": "development_only", "learner": kwargs["learner_name"],
            "stream_seed": kwargs["stream_seed"], "budget": kwargs["budget"],
            "bootstrap_labels_supplied": len(bootstrap), "stream_steps": len(stream),
            "base_stream_spec_sha256": spec.sha256, "schedule_sha256": schedule["schedule_sha256"],
            "selected_events": len(schedule["selected_events"]), "inputs": inputs,
            "scope": "training-side plan only; no prediction or validation/test access"}


def run_experiment_a_cell(*, embeddings_path: Path, split_csv: Path, stream_spec_path: Path,
                          learner_name: str, stream_seed: int, budget: int, output_dir: Path,
                          smoke_mode: bool = False, preflight_confirmed: bool = False,
                          clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    if not smoke_mode and not preflight_confirmed:
        raise RuntimeError("Experiment A requires manual user launch and contamination preflight")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite/reuse Experiment-A directory")
    spec, raw, bootstrap, stream, input_dim, schedule, inputs = _prepare(
        embeddings_path=Path(embeddings_path), split_csv=Path(split_csv), stream_spec_path=Path(stream_spec_path),
        learner_name=learner_name, stream_seed=stream_seed, budget=budget, smoke_mode=smoke_mode)
    effective = replace(spec, max_total_requests=budget, bootstrap_counts_toward_budget=False,
                        count_low_confidence=False, count_novel_registration=False)
    selected = frozenset(row["image_id"] for row in schedule["selected_events"])
    code_hashes = runner._checkpoint_code_hashes()
    for relative in ("ssl_hdc_pilot/scripts/experiment_a.py", "ssl_hdc_pilot/scripts/reconstruct_run.py"):
        code_hashes[relative] = runner._file_sha256(runner.REPO_ROOT / relative)
    role = "synthetic_smoke" if smoke_mode else "trusted_train"
    policy = {"experiment": "A", "additional_label_budget": budget,
              "bootstrap_counts_toward_budget": False, "uncertainty_acquisition": False,
              "privileged_novel_registration": False, "acquisition_threshold": None,
              "schedule_sha256": schedule["schedule_sha256"]}
    manifest = {"schema_version": 1, "stage": "development_only", "data_role": role,
                "learner": learner_name, "stream_seed": stream_seed, "policy": policy,
                "policy_sha256": _json_hash(policy), "inputs": inputs,
                "base_stream_spec_sha256": spec.sha256, "registered_base_configuration": json.loads(raw),
                "code_hashes": code_hashes, "git": _git_provenance(),
                "protocol": {"path": str(PROTOCOL_PATH), "sha256": runner._file_sha256(PROTOCOL_PATH),
                             "status": "draft_not_frozen"},
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "torch": str(runner.torch.__version__), "numpy": runner.np.__version__,
                                "torch_cuda_build": runner.torch.version.cuda, "requested_device": spec.device},
                "bootstrap_image_ids": [s.image_id for s in bootstrap],
                "evaluation_scope": "prequential and final bootstrap-plus-stream training pool only"}
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "shared_label_schedule.json", schedule)
    with (output_dir / "base_stream_spec.json").open("xb") as handle:
        handle.write(raw)

    def preserve(learner: Any, progress: dict[str, Any]) -> None:
        save_learner_checkpoint(output_dir / "final_state.npz", learner, context={
            "experiment": "A", "stage": "development_only", "data_role": role,
            "learner": learner_name, "stream_seed": stream_seed, "progress": progress,
            "policy": policy, "inputs": inputs, "code_hashes": code_hashes,
            "bootstrap_image_ids": [s.image_id for s in bootstrap],
            "stream_image_ids": [s.image_id for s in stream]})

    def preserve_predictions(rows: list[dict[str, Any]]) -> None:
        _write_json(output_dir / "final_training_predictions.json", {"data_role": role, "rows": rows})

    history, metrics = runner._run_one(
        learner_name=learner_name, threshold=None, bootstrap=bootstrap, stream=stream,
        spec=effective, input_dim=input_dim, clock=clock, on_final_state=preserve,
        on_final_predictions=preserve_predictions, shared_label_ids=selected)
    runner._write_history(output_dir / "history.csv", history)
    supplied = metrics["label_budget"]["stream_label_requests"]
    summary = {"experiment": "A", "stage": "development_only", "data_role": role,
               "learner": learner_name, "stream_seed": stream_seed, "policy": policy,
               "embedding_npz": str(embeddings_path), "embedding_dim": input_dim,
               "stream_steps": len(stream), "supervision": {
                   "bootstrap_labels_supplied": len(bootstrap), "additional_labels_consumed": supplied,
                   "continual_learner_labels_supplied": len(bootstrap) + supplied,
                   "conditional_on_preselected_representations": True,
                   "not_total_project_label_requirement": True}, **metrics}
    _write_json(output_dir / "summary.json", summary)
    observed = [{"step": r["step"], "image_id": r["image_id"]} for r in history if r["label_requested"]]
    failures = []
    if observed != schedule["selected_events"]:
        failures.append("observed_label_events_disagree_with_schedule")
    if supplied != budget or metrics["label_budget"]["total_charged"] != budget:
        failures.append("additional_label_accounting_mismatch")
    if any(r["label_request_denied_budget"] or (r["class_registered"] and not r["label_requested"]) for r in history):
        failures.append("unexpected_denial_or_unpurchased_registration")
    changed = [name for name, record in inputs.items()
               if runner._file_sha256(Path(record["path"])) != record["sha256"]]
    changed_code = [name for name, digest in code_hashes.items()
                    if runner._file_sha256(runner.REPO_ROOT / name) != digest]
    if runner._file_sha256(PROTOCOL_PATH) != manifest["protocol"]["sha256"]:
        changed.append("protocol")
    artifacts = {p.name: runner._file_sha256(p) for p in output_dir.iterdir() if p.is_file()}
    report = {"status": "PASS" if not (failures or changed or changed_code) else "FAIL",
              "experiment": "A", "stage": "development_only", "learner": learner_name,
              "stream_seed": stream_seed, "budget": budget, "labels_consumed": supplied,
              "stream_steps": len(stream), "schedule_sha256": schedule["schedule_sha256"],
              "failures": failures, "inputs_changed_during_run": changed,
              "code_changed_during_run": changed_code, "artifacts": artifacts,
              "scope": "schedule/accounting/provenance integrity, not predictive success, held-out evidence or benchmark"}
    _write_json(output_dir / "integrity_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--trusted-split", type=Path, required=True)
    parser.add_argument("--stream-spec", type=Path, default=runner.DEFAULT_STREAM_SPEC)
    parser.add_argument("--learner", choices=runner.LEARNER_NAMES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-confirmed", action="store_true")
    args = parser.parse_args()
    common = dict(embeddings_path=args.embeddings, split_csv=args.trusted_split,
                  stream_spec_path=args.stream_spec, learner_name=args.learner,
                  stream_seed=args.seed, budget=args.budget, smoke_mode=args.smoke)
    if not args.execute:
        print(json.dumps(plan_experiment_a(**common), indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        parser.error("--execute requires a fresh --output-dir")
    report = run_experiment_a_cell(**common, output_dir=args.output_dir,
                                   preflight_confirmed=args.preflight_confirmed)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
