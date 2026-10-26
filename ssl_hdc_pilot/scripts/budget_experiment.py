"""One user-launched development A/B cell with approved longitudinal snapshots.

Default action is a training-input plan only. No validation/test scoring or
benchmark. Existing Experiment-A and historical artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Callable

from ssl_hdc_pilot.scripts import stream_runner as runner
from ssl_hdc_pilot.scripts import checkpoint_storage
from ssl_hdc_pilot.scripts.experiment_a import _json_hash, _prepare as prepare_training
from ssl_hdc_pilot.scripts.learner_checkpoint import save_learner_checkpoint
from ssl_hdc_pilot.scripts.reconstruct_run import PROTOCOL_PATH, _git_provenance, _write_json


RETENTION_STEPS = (0, 195, 196, 390, 391, 585, 586, 780, 781, 976)
UPTAKE_COUNTS = (1, 2, 5, 10, 20, 50, 100)


def snapshot_policy(spec: runner.StreamSpec, stream_steps: int) -> dict[str, Any]:
    steps = sorted({0, stream_steps, *(i.at_step for i in spec.class_introductions),
                    *(i.at_step + 1 for i in spec.class_introductions)})
    if any(step < 0 or step > stream_steps for step in steps):
        raise ValueError("introduction checkpoint outside stream")
    if not spec.smoke and tuple(steps) != RETENTION_STEPS:
        raise ValueError("real retention grid differs from approved development settings")
    return {"retention_processed_events": steps, "uptake_purchased_counts": list(UPTAKE_COUNTS),
            "relative_final_recall_fraction": .8, "absolute_recall_target": .8,
            "event_semantics": "bootstrap t=0; zero-based history event i is observed at t=i+1",
            "capture": "one physical state per processed-event count; reasons combined"}


def _prepare(*, experiment: str, threshold: float | None, embeddings_path: Path,
             split_csv: Path, stream_spec_path: Path, learner_name: str,
             stream_seed: int, budget: int, smoke_mode: bool, ablation_mode: bool = False,
             hypervector_dim: int | None = None) -> tuple:
    if experiment not in {"A", "B"}:
        raise ValueError("experiment must be A or B")
    if experiment == "A" and threshold is not None:
        raise ValueError("Experiment A has no acquisition threshold")
    if experiment == "B" and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                               or not math.isfinite(threshold)):
        raise ValueError("Experiment B requires a finite acquisition threshold")
    if smoke_mode and split_csv.resolve().is_relative_to(runner.DEFAULT_APPROVED_SPLIT_DIR.resolve()):
        raise ValueError("synthetic mode cannot consume canonical partitions")
    prepared = prepare_training(embeddings_path=embeddings_path, split_csv=split_csv,
        stream_spec_path=stream_spec_path, learner_name=learner_name, stream_seed=stream_seed,
        budget=budget, smoke_mode=smoke_mode, ablation_mode=ablation_mode,
        hypervector_dim=hypervector_dim)
    spec, raw, bootstrap, stream, dimension, schedule, inputs = prepared
    if experiment == "B" and threshold not in spec.confidence_thresholds:
        raise ValueError("Experiment-B threshold is not registered")
    snapshots = snapshot_policy(spec, len(stream))
    policy = {"experiment": experiment, "additional_label_budget": budget,
        "bootstrap_counts_toward_budget": False, "acquisition_threshold": threshold,
        "uncertainty_acquisition": experiment == "B", "privileged_novel_registration": experiment == "B",
        "schedule_sha256": schedule["schedule_sha256"] if experiment == "A" else None,
        "snapshot_policy": snapshots}
    return spec, raw, bootstrap, stream, dimension, schedule, inputs, policy


def plan_budget_cell(**kwargs: Any) -> dict[str, Any]:
    spec, _, bootstrap, stream, _, _, inputs, policy = _prepare(**kwargs)
    return {"stage": "development_only", "experiment": kwargs["experiment"],
        "learner": kwargs["learner_name"], "stream_seed": kwargs["stream_seed"],
        "bootstrap_labels_supplied": len(bootstrap), "stream_steps": len(stream),
        "base_stream_spec_sha256": spec.sha256, "policy": policy, "inputs": inputs, "ablation_mode": kwargs.get("ablation_mode", False),
        "scope": "training input plan only; no learner, snapshots, validation/test scoring or benchmark"}


def _snapshot_reasons(processed: int, final_step: int, event: dict[str, Any] | None,
                      counts: dict[str, int], policy: dict[str, Any]) -> list[dict[str, Any]]:
    reasons = []
    if processed in policy["retention_processed_events"]:
        reasons.append({"kind": "retention"})
    if event and event["purchased"] and event["true_label"] in counts:
        label = event["true_label"]
        if counts[label] in policy["uptake_purchased_counts"]:
            reasons.append({"kind": "uptake", "class_label": label, "purchased_labels": counts[label]})
    if processed == final_step:
        reasons.append({"kind": "final"})
    return reasons


def run_budget_cell(*, experiment: str, threshold: float | None, embeddings_path: Path,
                    split_csv: Path, stream_spec_path: Path, learner_name: str, stream_seed: int,
                    budget: int, output_dir: Path, smoke_mode: bool = False,
                    ablation_mode: bool = False,
                    hypervector_dim: int | None = None,
                    preflight_confirmed: bool = False,
                    clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    if not smoke_mode and not preflight_confirmed:
        raise RuntimeError("budget experiment requires manual user launch and contamination preflight")
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite/reuse budget experiment directory")
    spec, raw, bootstrap, stream, dimension, schedule, inputs, policy = _prepare(
        experiment=experiment, threshold=threshold, embeddings_path=embeddings_path, split_csv=split_csv,
        stream_spec_path=stream_spec_path, learner_name=learner_name, stream_seed=stream_seed,
        budget=budget, smoke_mode=smoke_mode, ablation_mode=ablation_mode,
        hypervector_dim=hypervector_dim)
    effective = replace(spec, max_total_requests=budget, bootstrap_counts_toward_budget=False,
                        count_low_confidence=experiment == "B", count_novel_registration=experiment == "B")
    code = runner._checkpoint_code_hashes()
    for name in ("experiment_a.py", "reconstruct_run.py", "budget_experiment.py"):
        relative = "ssl_hdc_pilot/scripts/" + name
        code[relative] = runner._file_sha256(runner.REPO_ROOT / relative)
    role = "synthetic_smoke" if smoke_mode else "trusted_train"
    arm_keys = {"arm_f_trusted_all.npz": "s_frozen", "arm_f_vitb16_trusted_all.npz": "b_frozen",
        "arm_f_vitl16_trusted_all.npz": "l_frozen", "arm_d_vitb16_ep3_trusted_all.npz": "b_adapted",
        "arm_d_vitl16_ep4_trusted_all.npz": "l_adapted"}
    arm_key = arm_keys.get(embeddings_path.name) if ablation_mode else None
    common = {"schema_version": 1, "artifact_kind": "budget_longitudinal_development",
        "stage": "development_only", "data_role": role, "experiment": experiment,
        "learner": learner_name, "stream_seed": stream_seed, "policy": policy,
        "policy_sha256": _json_hash(policy), "inputs": inputs, "code_hashes": code,
        "bootstrap_image_ids": [x.image_id for x in bootstrap],
        "stream_image_ids": [x.image_id for x in stream]}
    manifest = {**common, "base_stream_spec_sha256": spec.sha256,
        "ablation_mode": ablation_mode, "ablation_arm": arm_key,
        **({"ablation_hypervector_dim": hypervector_dim} if ablation_mode else {}),
        "checkpoint_storage": checkpoint_storage.STORAGE_DESCRIPTION,
        "registered_base_configuration": json.loads(raw), "git": _git_provenance(),
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": runner._file_sha256(PROTOCOL_PATH),
                     "status": "development_settings_approved_full_protocol_not_frozen"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": str(runner.torch.__version__), "numpy": runner.np.__version__,
                        "requested_device": spec.device},
        "scope": "training-only snapshot capture; no held-out scoring or benchmark"}
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "manifest.json", manifest)
    with (output_dir / "base_stream_spec.json").open("xb") as handle:
        handle.write(raw)
    if experiment == "A":
        _write_json(output_dir / "shared_label_schedule.json", schedule)
    snapshots, events = [], []

    def observe(learner: Any, progress: dict[str, Any]) -> None:
        processed, event = progress["processed_stream_events"], progress["last_event"]
        if event is not None:
            events.append(event)
        reasons = _snapshot_reasons(processed, len(stream), event, progress["novel_labels_consumed"],
                                    policy["snapshot_policy"])
        if not reasons:
            return
        filename = "final_state.npz" if processed == len(stream) else f"snapshots/step_{processed:06d}.npz"
        context = {**common, "progress": progress, "snapshot_reasons": reasons}
        save_learner_checkpoint(output_dir / filename, learner, context=context,
            shared_projection_root=output_dir)
        snapshots.append({"processed_stream_events": processed, "path": filename,
                          "reasons": reasons, "labels_consumed": progress["labels_charged"],
                          "novel_labels_consumed": progress["novel_labels_consumed"],
                          "sha256": runner._file_sha256(output_dir / filename)})

    def predictions(rows: list[dict[str, Any]]) -> None:
        _write_json(output_dir / "final_training_predictions.json", {"data_role": role, "rows": rows})

    history, metrics = runner._run_one(learner_name=learner_name, threshold=threshold,
        bootstrap=bootstrap, stream=stream, spec=effective, input_dim=dimension, clock=clock,
        shared_label_ids=frozenset(x["image_id"] for x in schedule["selected_events"]) if experiment == "A" else None,
        on_observation=observe, on_final_predictions=predictions, ablation_mode=ablation_mode,
        hypervector_dim=hypervector_dim)
    runner._write_history(output_dir / "history.csv", history)
    consumed = sum(e["purchased"] for e in events)
    accounting = {"allowance": budget, "potential_requests": sum(e["potential_request"] for e in events),
        "labels_consumed": consumed, "unused_allowance": budget - consumed,
        "denied_requests": sum(e["denied"] for e in events),
        "potential_requests_by_class": dict(Counter(e["true_label"] for e in events if e["potential_request"])),
        "purchases_by_class": dict(Counter(e["true_label"] for e in events if e["purchased"])),
        "registration_events": [e for e in events if e["class_registered"]],
        "request_semantics": "potential requests include denied events; each purchase is charged once"}
    _write_json(output_dir / "label_events.json", {"data_role": role, "rows": events, "accounting": accounting})
    _write_json(output_dir / "snapshot_index.json", {"snapshot_policy": policy["snapshot_policy"], "snapshots": snapshots})
    _write_json(output_dir / "summary.json", {**metrics, "experiment": experiment, "stage": "development_only",
        "data_role": role, "learner": learner_name, "stream_seed": stream_seed, "policy": policy,
        "ablation_mode": ablation_mode, "ablation_arm": arm_key,
        "stream_steps": len(stream), "accounting": accounting, "supervision": {
            "bootstrap_labels_supplied": len(bootstrap), "additional_labels_consumed": consumed,
            "continual_learner_labels_supplied": len(bootstrap) + consumed,
            "conditional_on_preselected_representations": True, "not_total_project_label_requirement": True}})
    failures = []
    if len(events) != len(stream) or len(history) != len(stream):
        failures.append("event_count_mismatch")
    if (not 0 <= consumed <= budget or metrics["label_budget"]["stream_label_requests"] != consumed
            or metrics["label_budget"]["total_charged"] != consumed):
        failures.append("budget_accounting_mismatch")
    spent = 0
    confirmed = set()
    expected_counts = {label: 0 for label in spec.novel_classes}
    expected_snapshots = [(0, _snapshot_reasons(0, len(stream), None, expected_counts, policy["snapshot_policy"]),
                           0, dict(expected_counts))]
    for event, row in zip(events, history):
        requested = any(event["request_reasons"].values())
        expected_purchase = requested and spent < budget
        if (requested != event["potential_request"] or expected_purchase != event["purchased"]
                or event["denied"] != (requested and not expected_purchase)
                or row["label_requested"] != event["purchased"]
                or row["label_request_denied_budget"] != event["denied"]
                or (event["class_registered"] and not event["purchased"])):
            failures.append("event_accounting_mismatch")
        if experiment == "B":
            expected_novel = event["true_label"] in spec.novel_classes and event["true_label"] not in confirmed
            if event["request_reasons"]["novel_registration"] != expected_novel:
                failures.append("novel_request_policy_mismatch")
        if event["purchased"]:
            spent += 1
            if event["true_label"] in expected_counts:
                confirmed.add(event["true_label"])
                expected_counts[event["true_label"]] += 1
        reasons = _snapshot_reasons(event["step"] + 1, len(stream), event, expected_counts, policy["snapshot_policy"])
        if reasons:
            expected_snapshots.append((event["step"] + 1, reasons, spent, dict(expected_counts)))
    actual = [(s["processed_stream_events"], s["reasons"], s["labels_consumed"], s["novel_labels_consumed"])
              for s in snapshots]
    if actual != expected_snapshots:
        failures.append("snapshot_schedule_mismatch")
    if experiment == "A" and ([{"step": e["step"], "image_id": e["image_id"]} for e in events if e["purchased"]]
                               != schedule["selected_events"] or consumed != budget):
        failures.append("shared_label_schedule_mismatch")
    changed_inputs = [name for name, record in inputs.items()
                      if runner._file_sha256(Path(record["path"])) != record["sha256"]]
    if runner._file_sha256(PROTOCOL_PATH) != manifest["protocol"]["sha256"]:
        changed_inputs.append("protocol")
    changed_code = [name for name, digest in code.items() if runner._file_sha256(runner.REPO_ROOT / name) != digest]
    artifacts = {p.relative_to(output_dir).as_posix(): runner._file_sha256(p)
                 for p in output_dir.rglob("*") if p.is_file()}
    if any(artifacts.get(s["path"]) != s["sha256"] for s in snapshots):
        failures.append("snapshot_changed_after_capture")
    report = {"status": "PASS" if not (failures or changed_inputs or changed_code) else "FAIL",
        "artifact_kind": common["artifact_kind"], "experiment": experiment, "stage": "development_only",
        "learner": learner_name, "stream_seed": stream_seed, "budget": budget, "threshold": threshold,
        "ablation_mode": ablation_mode, "ablation_arm": arm_key,
        **({"ablation_hypervector_dim": hypervector_dim} if ablation_mode else {}),
        "stream_steps": len(stream), "labels_consumed": consumed, "unused_allowance": budget - consumed,
        "snapshots_saved": len(snapshots), "retention_checkpoints": policy["snapshot_policy"]["retention_processed_events"],
        "failures": sorted(set(failures)), "inputs_changed_during_run": changed_inputs,
        "code_changed_during_run": changed_code, "artifacts": artifacts,
        "scope": "training acquisition/snapshot integrity only; not held-out, forgetting or benchmark evidence"}
    _write_json(output_dir / "integrity_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("A", "B"), required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--trusted-split", type=Path, required=True)
    parser.add_argument("--stream-spec", type=Path, default=runner.DEFAULT_STREAM_SPEC)
    parser.add_argument("--learner", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ablation-mode", action="store_true")
    parser.add_argument("--hypervector-dim", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-confirmed", action="store_true")
    args = parser.parse_args()
    allowed_learners = runner.LEARNER_NAMES + (runner.ABLATION_LEARNER_NAMES if args.ablation_mode else ())
    if args.learner not in allowed_learners:
        parser.error("--learner must be canonical unless --ablation-mode is explicit")
    if args.hypervector_dim is not None and not args.ablation_mode:
        parser.error("--hypervector-dim is only accepted together with --ablation-mode")
    kwargs = dict(experiment=args.experiment, threshold=args.threshold, embeddings_path=args.embeddings,
                  split_csv=args.trusted_split, stream_spec_path=args.stream_spec, learner_name=args.learner,
                  stream_seed=args.seed, budget=args.budget, smoke_mode=args.smoke, ablation_mode=args.ablation_mode,
                  hypervector_dim=args.hypervector_dim)
    if args.execute:
        if args.output_dir is None:
            parser.error("--execute requires a fresh --output-dir")
        result = run_budget_cell(**kwargs, output_dir=args.output_dir, preflight_confirmed=args.preflight_confirmed)
    else:
        result = plan_budget_cell(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
