"""Manual validation-only scoring of approved A/B longitudinal snapshots.

Default is a read-only input plan. No training, test access, state selection,
calibration fitting or benchmark. Completed sources are never modified.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from ssl_hdc_pilot.scripts import budget_experiment as capture
from ssl_hdc_pilot.scripts import checkpoint_storage
from ssl_hdc_pilot.scripts import evaluate_validation as validation
from ssl_hdc_pilot.scripts import learner_checkpoint as checkpoints
from ssl_hdc_pilot.scripts import longitudinal_metrics as metrics
from ssl_hdc_pilot.scripts import stream_runner as runner
from ssl_hdc_pilot.scripts.build_stream_spec import _load_approved_partition
from ssl_hdc_pilot.scripts.experiment_a import _json_hash
from ssl_hdc_pilot.scripts.reconstruct_run import PROTOCOL_PATH, _git_provenance, _write_json


BASE_FILES = {"manifest.json", "base_stream_spec.json", "history.csv", "summary.json",
              "final_training_predictions.json", "label_events.json", "snapshot_index.json", "final_state.npz"}


@dataclass
class Prepared:
    manifest: dict[str, Any]
    report: dict[str, Any]
    spec: runner.StreamSpec
    rows: list
    classes: list[str]
    embeddings: dict
    dimension: int
    pins: dict
    snapshots: list[dict]
    progress: dict[int, dict]
    purchases: dict[str, list[int]]


def _audit_events(manifest, report, index, events, bootstrap, stream, spec, schedule):
    """Rebuild capture schedule from authenticated stream identities and purchases."""
    if len(events) != len(stream):
        raise ValueError("source event count differs from stream")
    policy = manifest["policy"]
    grid = policy["snapshot_policy"]
    if index.get("snapshot_policy") != grid:
        raise ValueError("snapshot policy differs from manifest")
    scheduled = {e["image_id"] for e in schedule["selected_events"]}
    spent, requests, denied = 0, 0, 0
    counts = {label: 0 for label in spec.novel_classes}
    purchases = {label: [] for label in spec.novel_classes}
    progress, expected = {}, []

    def remember(t, event):
        reasons = capture._snapshot_reasons(t, len(stream), event, counts, grid)
        if reasons:
            path = "final_state.npz" if t == len(stream) else f"snapshots/step_{t:06d}.npz"
            expected.append({"processed_stream_events": t, "path": path, "reasons": reasons,
                             "labels_consumed": spent, "novel_labels_consumed": dict(counts),
                             "sha256": report["artifacts"].get(path)})
            progress[t] = {"processed_stream_events": t, "labels_charged": spent,
                "bootstrap_labels_charged": 0, "novel_labels_consumed": dict(counts),
                "confirmed_novel_classes": sorted(c for c, n in counts.items() if n),
                "stream_complete": t == len(stream), "last_event": event}

    remember(0, None)
    for step, (event, sample) in enumerate(zip(events, stream)):
        if (event.get("step") != step or event.get("image_id") != sample.image_id
                or event.get("true_label") != sample.label):
            raise ValueError("event identity differs from regenerated training stream")
        reasons = event["request_reasons"]
        if (set(reasons) != {"shared_schedule", "low_confidence", "novel_registration"}
                or any(type(v) is not bool for v in reasons.values())
                or any(type(event.get(k)) is not bool for k in
                       ("potential_request", "purchased", "denied", "class_registered"))):
            raise ValueError("invalid event decision flags")
        novel = sample.label in counts and counts[sample.label] == 0
        if policy["experiment"] == "A":
            if reasons != {"shared_schedule": sample.image_id in scheduled,
                           "low_confidence": False, "novel_registration": False}:
                raise ValueError("A acquisition differs from shared schedule")
        elif reasons["shared_schedule"] or reasons["novel_registration"] != novel:
            raise ValueError("B acquisition differs from approved novel policy")
        requested = any(reasons.values())
        purchased = requested and spent < policy["additional_label_budget"]
        if (event["potential_request"] != requested or event["purchased"] != purchased
                or event["denied"] != (requested and not purchased)
                or event["class_registered"] != (purchased and novel)):
            raise ValueError("source purchase/exhaustion/registration accounting mismatch")
        requests += requested
        denied += event["denied"]
        if purchased:
            spent += 1
            if sample.label in counts:
                counts[sample.label] += 1
                purchases[sample.label].append(step + 1)
        remember(step + 1, event)
    if (index.get("snapshots") != expected or report.get("snapshots_saved") != len(expected)
            or report.get("retention_checkpoints") != grid["retention_processed_events"]):
        raise ValueError("snapshot index does not match purchase/retention schedule")
    if (report.get("labels_consumed") != spent
            or report.get("unused_allowance") != policy["additional_label_budget"] - spent
            or report.get("stream_steps") != len(stream)):
        raise ValueError("source receipt accounting mismatch")
    return progress, purchases, {"labels_consumed": spent, "potential_requests": requests,
                               "denied_requests": denied, "unused_allowance": report["unused_allowance"]}


def _prepare(source_dir: Path, embeddings_path: Path, validation_csv: Path, smoke_mode: bool) -> Prepared:
    canonical = runner.DEFAULT_APPROVED_SPLIT_DIR.resolve()
    if not smoke_mode and validation_csv.resolve() != canonical / "val.csv":
        raise RuntimeError("validation-only: canonical val.csv required before input access")
    if smoke_mode and validation_csv.resolve().is_relative_to(canonical):
        raise RuntimeError("synthetic smoke cannot open canonical partitions")
    pins = {}

    def pin(name, path):
        pins[name] = {"path": str(path.resolve()), "sha256": runner._file_sha256(path)}

    def read(name):
        return json.loads((source_dir / name).read_text(encoding="utf-8"))

    pin("source_report", source_dir / "integrity_report.json")
    report = read("integrity_report.json")
    if (report.get("status") != "PASS" or report.get("stage") != "development_only"
            or report.get("artifact_kind") != "budget_longitudinal_development"
            or report.get("experiment") not in {"A", "B"}
            or any(report.get(k) != [] for k in
                   ("failures", "inputs_changed_during_run", "code_changed_during_run"))):
        raise ValueError("complete PASS A/B snapshot receipt required")
    names = set(report["artifacts"])
    required = BASE_FILES | ({"shared_label_schedule.json"} if report["experiment"] == "A" else set())
    if not required <= names or any(not re.fullmatch(
            rf"snapshots/step_[0-9]{{6}}\.npz|{checkpoint_storage.BLOB_PATTERN}", n) for n in names - required):
        raise ValueError("invalid or missing source artifact paths")
    for name, digest in report["artifacts"].items():
        path = (source_dir / name).resolve()
        if not path.is_relative_to(source_dir.resolve()):
            raise ValueError("artifact path escapes source directory")
        pin("source/" + name, path)
        if pins["source/" + name]["sha256"] != digest:
            raise ValueError(f"source artifact hash mismatch: {name}")
    manifest = read("manifest.json")
    role = "synthetic_smoke" if smoke_mode else "trusted_train"
    if (manifest.get("schema_version") != 1 or manifest.get("data_role") != role
            or any(manifest.get(k) != report.get(k) for k in
                   ("artifact_kind", "stage", "experiment", "learner", "stream_seed"))):
        raise ValueError("source manifest identity or role mismatch")
    required_code = set(runner._checkpoint_code_hashes()) | {
        "ssl_hdc_pilot/scripts/" + n for n in ("experiment_a.py", "reconstruct_run.py", "budget_experiment.py")}
    if set(manifest["code_hashes"]) != required_code:
        raise ValueError("source lacks exact required code provenance")
    for name, digest in manifest["code_hashes"].items():
        pin("code/" + name, runner.REPO_ROOT / name)
        if pins["code/" + name]["sha256"] != digest:
            raise ValueError(f"source code changed; compatibility review required: {name}")
    for name in ("evaluate_validation.py", "longitudinal_metrics.py", "evaluate_snapshots.py"):
        pin("code/" + name, Path(__file__).parent / name)
    pin("protocol", PROTOCOL_PATH)
    train_path = Path(manifest["inputs"]["train"]["path"])
    spec_path = Path(manifest["inputs"]["base_stream_spec"]["path"])
    prepared = capture._prepare(experiment=report["experiment"], threshold=report["threshold"],
        embeddings_path=embeddings_path, split_csv=train_path, stream_spec_path=spec_path,
        learner_name=report["learner"], stream_seed=report["stream_seed"], budget=report["budget"],
        smoke_mode=smoke_mode, ablation_mode=bool(manifest.get("ablation_mode", False)))
    spec, raw, bootstrap, stream, dimension, schedule, inputs, policy = prepared
    pins.update(inputs)
    if (manifest["inputs"] != inputs or manifest["policy"] != policy
            or manifest["policy_sha256"] != _json_hash(policy)
            or manifest["base_stream_spec_sha256"] != spec.sha256
            or manifest["registered_base_configuration"] != json.loads(raw)
            or (source_dir / "base_stream_spec.json").read_bytes() != raw
            or manifest["bootstrap_image_ids"] != [s.image_id for s in bootstrap]
            or manifest["stream_image_ids"] != [s.image_id for s in stream]
            or manifest["environment"]["requested_device"] != spec.device):
        raise ValueError("source differs from regenerated approved training inputs/policy")
    if report["experiment"] == "A" and read("shared_label_schedule.json") != schedule:
        raise ValueError("source shared schedule mismatch")
    index, ledger = read("snapshot_index.json"), read("label_events.json")
    progress, purchases, accounting = _audit_events(
        manifest, report, index, ledger["rows"], bootstrap, stream, spec, schedule)
    if ledger.get("data_role") != role or any(ledger["accounting"].get(k) != v for k, v in accounting.items()):
        raise ValueError("source ledger accounting mismatch")
    snapshot_names = {s["path"] for s in index["snapshots"]}
    dependencies = set()
    for name in snapshot_names:
        dependencies.update(checkpoint_storage.projection_dependencies(source_dir / name,
            shared_projection_root=source_dir))
    if names != required | snapshot_names | dependencies:
        raise ValueError("unindexed source snapshot")
    if dependencies and manifest.get("checkpoint_storage") != checkpoint_storage.STORAGE_DESCRIPTION:
        raise ValueError("source lacks shared projection storage declaration")
    pin("validation", validation_csv)
    rows = (runner._load_split_rows(validation_csv, smoke_mode=True) if smoke_mode
            else _load_approved_partition(canonical, "val"))
    classes = sorted(set(spec.initially_known_classes) | set(spec.novel_classes))
    ids = [i for i, _ in rows]
    if (len(ids) != len(set(ids)) or set(label for _, label in rows) != set(classes)
            or set(ids) & set(manifest["bootstrap_image_ids"] + manifest["stream_image_ids"])
            or (not smoke_mode and (len(rows) != 1132 or len(classes) != 19))):
        raise ValueError("invalid fixed validation population or training overlap")
    embeddings, loaded_dimension = runner._load_embedding_map(embeddings_path)
    if loaded_dimension != dimension or any(i not in embeddings for i in ids):
        raise ValueError("missing or inconsistent validation embeddings")
    # The cached file spans partitions, but the scorer receives validation only.
    embeddings = {image_id: embeddings[image_id] for image_id in ids}
    validation._unchanged(pins)
    return Prepared(manifest, report, spec, rows, classes, embeddings, dimension,
                    pins, index["snapshots"], progress, purchases)


def plan_snapshot_validation(*, source_dir: Path, embeddings_path: Path, validation_csv: Path,
                             smoke_mode: bool = False) -> dict:
    p = _prepare(source_dir, embeddings_path, validation_csv, smoke_mode)
    return {"stage": "development_only", "samples_per_snapshot": len(p.rows),
        "classes": p.classes, "snapshots": len(p.snapshots), "experiment": p.report["experiment"],
        "learner": p.report["learner"], "budget": p.report["budget"],
        "processed_event_counts": [s["processed_stream_events"] for s in p.snapshots],
        "scope": "read-only validation input plan; no learner loaded or scored"}


def run_snapshot_validation(*, source_dir: Path, embeddings_path: Path, validation_csv: Path,
                            output_dir: Path, smoke_mode: bool = False,
                            preflight_confirmed: bool = False) -> dict:
    if not smoke_mode and not preflight_confirmed:
        raise RuntimeError("snapshot validation requires manual user launch and preflight")
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite/reuse snapshot validation output")
    if output_dir.resolve().is_relative_to(source_dir.resolve()):
        raise ValueError("evaluation output cannot be inside source")
    p = _prepare(source_dir, embeddings_path, validation_csv, smoke_mode)
    role = "synthetic_smoke" if smoke_mode else "trusted_val"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "predictions").mkdir()
    ledgers, results = {}, []
    common = {key: p.manifest[key] for key in ("schema_version", "artifact_kind", "stage", "data_role",
        "experiment", "learner", "stream_seed", "policy", "policy_sha256", "inputs", "code_hashes",
        "bootstrap_image_ids", "stream_image_ids")}
    for snapshot in p.snapshots:
        step = snapshot["processed_stream_events"]
        checkpoint = checkpoints.load_learner_checkpoint(source_dir / snapshot["path"],
            device=p.spec.device, restore_rng=False, shared_projection_root=source_dir)
        expected = {**common, "progress": p.progress[step], "snapshot_reasons": snapshot["reasons"]}
        if checkpoint.context != expected or checkpoint.learner.input_dim != p.dimension:
            raise ValueError("checkpoint context/dimension differs from authenticated capture")
        known = set(p.spec.initially_known_classes) | set(p.progress[step]["confirmed_novel_classes"])
        if set(checkpoint.learner.known_classes) != known:
            raise ValueError("checkpoint class registry differs from paid registrations")
        result = validation.evaluate_fixed_population(checkpoint.learner, p.rows, p.embeddings, classes=p.classes)
        rows = result.pop("rows")
        ledgers[step] = rows
        filename = f"predictions/step_{step:06d}.json"
        _write_json(output_dir / filename, {"data_role": role, "processed_events": step, "rows": rows})
        results.append({"processed_events": step, "source_snapshot": snapshot, "predictions": filename, **result})
        print(f"Validated snapshot {len(results)}/{len(p.snapshots)} at t={step}", flush=True)
        del checkpoint
    grid = p.manifest["policy"]["snapshot_policy"]
    forgetting = metrics.summarize_forgetting(ledgers, classes=p.classes,
        old_classes=p.spec.initially_known_classes, checkpoint_steps=grid["retention_processed_events"])
    uptake = {label: metrics.summarize_uptake(ledgers, classes=p.classes, class_label=label,
        purchased_label_steps=steps, checkpoint_counts=grid["uptake_purchased_counts"],
        final_step=p.report["stream_steps"], relative_fraction=grid["relative_final_recall_fraction"],
        absolute_target=grid["absolute_recall_target"]) for label, steps in p.purchases.items()}
    validation._unchanged(p.pins)
    _write_json(output_dir / "metrics.json", {"data_role": role, "snapshots": results,
        "forgetting": forgetting, "uptake": uptake, "scope": "one validation development cell; no comparator claim"})
    _write_json(output_dir / "manifest.json", {"stage": "development_only", "data_role": role,
        "source_manifest_sha256": p.pins["source/manifest.json"]["sha256"], "inputs": p.pins,
        "experiment": p.report["experiment"], "learner": p.report["learner"], "budget": p.report["budget"],
        "stream_seed": p.report["stream_seed"], "policy": p.manifest["policy"], "git": _git_provenance(),
        "device": p.spec.device, "classes": p.classes, "population_sha256": _json_hash(p.rows),
        "selection": "all approved snapshots; no fitting or model selection",
        "probability_semantics": "existing softmax over registered classes; 15-bin ECE"})
    validation._unchanged(p.pins)
    report = {"status": "PASS", "stage": "development_only", "data_role": role,
        "snapshots_scored": len(results), "samples_per_snapshot": len(p.rows), "class_count": len(p.classes),
        "state_and_rng_unchanged": True,
        "artifacts": {x.relative_to(output_dir).as_posix(): runner._file_sha256(x)
                      for x in output_dir.rglob("*") if x.is_file()},
        "scope": "fixed validation longitudinal development; not test, comparative evidence or benchmark"}
    _write_json(output_dir / "integrity_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--validation-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-confirmed", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    kwargs = dict(source_dir=args.source_dir, embeddings_path=args.embeddings,
                  validation_csv=args.validation_split, smoke_mode=args.smoke)
    if args.execute:
        if args.output_dir is None:
            parser.error("--execute requires a fresh --output-dir")
        result = run_snapshot_validation(**kwargs, output_dir=args.output_dir,
                                         preflight_confirmed=args.preflight_confirmed)
    else:
        result = plan_snapshot_validation(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
