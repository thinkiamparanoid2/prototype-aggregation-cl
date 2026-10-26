"""One manual six-learner A/B benchmark development condition, never a grid.

Default plans training inputs only. Real CUDA execution requires user preflight.
No validation/test scoring; each measurement starts from preserved bootstrap/RNG.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import os
import platform
import random
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from ssl_hdc_pilot.scripts import benchmark_core as core
from ssl_hdc_pilot.scripts import benchmark_environment as environment
from ssl_hdc_pilot.scripts import budget_experiment as capture
from ssl_hdc_pilot.scripts import checkpoint_storage
from ssl_hdc_pilot.scripts import learner_checkpoint as checkpoints
from ssl_hdc_pilot.scripts import stream_runner as runner
from ssl_hdc_pilot.scripts.evaluate_validation import _fingerprint, _unchanged
from ssl_hdc_pilot.scripts.experiment_a import _json_hash
from ssl_hdc_pilot.scripts.reconstruct_run import PROTOCOL_PATH, _git_provenance, _write_json, compare_values


POLICY_PATH = runner.REPO_ROOT / "ssl_hdc_pilot/BENCHMARK_DEVELOPMENT_PLAN.md"


def _prepare(**kwargs):
    prepared = capture._prepare(**kwargs, learner_name="ncm")
    spec, raw, bootstrap, stream, dimension, schedule, inputs, policy = prepared
    if not kwargs["smoke_mode"] and spec.device != "cuda":
        raise ValueError("approved benchmark placement requires CUDA")
    if any(s.embedding.device.type != "cpu" for s in [*bootstrap, *stream]):
        raise ValueError("benchmark requires host-resident embeddings")
    return prepared


def plan_benchmark(**kwargs):
    spec, _, bootstrap, stream, _, _, inputs, policy = _prepare(**kwargs)
    return {"stage": "development_only", "acquisition_policy": policy, "inputs": inputs,
        "experiment": kwargs["experiment"], "budget": kwargs["budget"], "stream_seed": kwargs["stream_seed"],
        "learners": list(core.ORDER), "bootstrap_labels": len(bootstrap), "stream_events": len(stream),
        "device": spec.device, "warmups_per_cell": 1, "paired_blocks": core.REPEATS,
        "block_orders": [list(core.block_order(i)) for i in range(core.REPEATS)],
        "passes": "untimed references, disposable warmup, five throughput and latency passes per learner, separate memory pass",
        "scope": "input plan only; no model construction, training, scoring or timing"}


def _capture_rng():
    arrays = {}
    return checkpoints._capture_random_state(arrays), arrays


def _restore_rng(saved):
    record, arrays = saved
    rng = checkpoints._read_random_state(record, dict(arrays))
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["cpu"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])


def _check_result(result, predictions, reference):
    comparison = compare_values(reference, {"trace": result["trace"], "final_predictions": predictions}, atol=1e-6)
    if comparison["mismatch_count"]:
        raise ValueError(f"benchmark fidelity mismatch: {comparison}")


def run_benchmark(*, output_dir: Path, preflight_confirmed: bool = False,
                  clock=time.perf_counter, **kwargs):
    smoke = kwargs["smoke_mode"]
    if not smoke and not preflight_confirmed:
        raise RuntimeError("benchmark requires manual user launch and preflight")
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite/reuse benchmark attempt")
    spec, raw, bootstrap, stream, dimension, schedule, inputs, policy = _prepare(**kwargs)
    if not smoke:
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real benchmark development")
        torch.cuda.init()  # Pin global CUDA RNG before building any reference.
    effective = replace(spec, max_total_requests=kwargs["budget"], bootstrap_counts_toward_budget=False,
                        count_low_confidence=kwargs["experiment"] == "B",
                        count_novel_registration=kwargs["experiment"] == "B")
    selected = frozenset(x["image_id"] for x in schedule["selected_events"]) if kwargs["experiment"] == "A" else None
    pins = dict(inputs)
    code = runner._checkpoint_code_hashes()
    for name in ("experiment_a.py", "reconstruct_run.py", "budget_experiment.py", "evaluate_validation.py",
                 "benchmark_core.py", "benchmark_environment.py", "benchmark_study.py"):
        relative = "ssl_hdc_pilot/scripts/" + name
        code[relative] = runner._file_sha256(runner.REPO_ROOT / relative)
    for name, digest in code.items():
        pins["code/" + name] = {"path": str(runner.REPO_ROOT / name), "sha256": digest}
    for name, path in (("protocol", PROTOCOL_PATH), ("benchmark_policy", POLICY_PATH)):
        pins[name] = {"path": str(path), "sha256": runner._file_sha256(path)}
    output_dir.mkdir(parents=True, exist_ok=False)
    checks = []

    def guard(label):
        check = {"phase": label, "result": {"scope": "synthetic tests; no machine cleanliness claim"}
                 if smoke else environment.contamination_check()}
        checks.append(check)

    def load(path):
        return checkpoints.load_learner_checkpoint(path, device=spec.device, restore_rng=True,
            shared_projection_root=output_dir).learner

    def measured(learner, mode):
        return core.stream_pass(learner, stream, budget=kwargs["budget"], novel_classes=spec.novel_classes,
                                selected_ids=selected, threshold=kwargs["threshold"], mode=mode, clock=clock)

    try:
        guard("before_setup")
        _unchanged(pins)
        _write_json(output_dir / "manifest.json", {"schema_version": 1, "stage": "development_only",
            "experiment": kwargs["experiment"], "budget": kwargs["budget"], "stream_seed": kwargs["stream_seed"],
            "checkpoint_storage": checkpoint_storage.STORAGE_DESCRIPTION,
            "data_role": "synthetic_smoke" if smoke else "trusted_train", "inputs": pins,
            "acquisition_policy": policy, "policy_sha256": _json_hash(policy), "git": _git_provenance(),
            "environment": {"platform": platform.platform(), "cpu": platform.processor(),
                "python": platform.python_version(), "torch": str(torch.__version__),
                "numpy": np.__version__, "cuda_build": torch.version.cuda, "device": spec.device,
                "gpu": torch.cuda.get_device_name() if not smoke else None,
                "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                "thread_environment": {k:os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                "embedding_dtype": str(bootstrap[0].embedding.dtype), "learner_tensor_dtype": "float32",
                "hardware": environment.machine_info() if not smoke else {"scope": "synthetic; no hardware evidence"}},
            "block_orders": [core.block_order(i) for i in range(core.REPEATS)], "warmups_per_cell": 1,
            "embedding_residency": "host; original learner transfers included",
            "trace_overhead": "minimal decision ledger included consistently; output/serialization excluded",
            "scope": "one training-side development condition; no held-out evaluation or final scientific release"})
        with (output_dir / "base_stream_spec.json").open("xb") as handle:
            handle.write(raw)
        if selected is not None:
            _write_json(output_dir / "shared_label_schedule.json", schedule)
        references, bootstrap_fingerprints = {}, {}
        for name in core.ORDER:
            guard("reference/" + name)
            directory = output_dir / name
            directory.mkdir()
            before_build = _capture_rng()
            initial = runner._build_learner(name, spec=effective, input_dim=dimension)
            checkpoints.save_learner_checkpoint(directory / "initial.npz", initial,
                context={"phase": "before_bootstrap"}, shared_projection_root=output_dir)
            del initial
            _restore_rng(before_build)
            final_rows = []
            def observe(model, progress):
                if progress["processed_stream_events"] == 0:
                    checkpoints.save_learner_checkpoint(directory / "bootstrap.npz", model,
                        context=progress, shared_projection_root=output_dir)
                    bootstrap_fingerprints[name] = _fingerprint(model)
                if progress["stream_complete"]:
                    checkpoints.save_learner_checkpoint(directory / "reference_final.npz", model,
                        context=progress, shared_projection_root=output_dir)
            history, summary = runner._run_one(learner_name=name, threshold=kwargs["threshold"],
                bootstrap=bootstrap, stream=stream, spec=effective, input_dim=dimension, clock=clock,
                shared_label_ids=selected, on_observation=observe, on_final_predictions=final_rows.extend)
            keys = ("step", "image_id", "true_label", "prediction", "confidence", "top_probability",
                    "label_requested", "label_request_denied_budget", "class_registered")
            references[name] = {"trace": [{k:r[k] for k in keys} for r in history], "final_predictions": final_rows}
            _write_json(directory / "reference.json", {**references[name], "training_summary": summary})
            model = load(directory / "bootstrap.npz")
            warmup = measured(model, "untimed")
            predictions = core.final_predictions(model, [*bootstrap, *stream])
            _check_result(warmup, predictions, references[name])
            _write_json(directory / "warmup.json", {"fidelity": "PASS", "discarded": True,
                        "trace_sha256": _json_hash(warmup["trace"]), "predictions_sha256": _json_hash(predictions)})
            del model
            gc.collect()
            guard("after_warmup/" + name)
        timings = []
        for block in range(core.REPEATS):
            for name in core.block_order(block):
                directory = output_dir / name
                for mode in ("throughput", "latency"):
                    guard(f"before/{block}/{name}/{mode}")
                    model = load(directory / "bootstrap.npz")
                    if _fingerprint(model) != bootstrap_fingerprints[name]:
                        raise ValueError("bootstrap state/RNG differs before repetition")
                    result = measured(model, mode)
                    predictions = core.final_predictions(model, [*bootstrap, *stream])
                    _check_result(result, predictions, references[name])
                    result.update(block=block, learner=name, fidelity="PASS", final_predictions=predictions)
                    _write_json(directory / f"block_{block}_{mode}.json", result)
                    timings.append({k:v for k,v in result.items() if k not in {"trace", "final_predictions", "update_seconds"}})
                    del model
                    gc.collect()
                    guard(f"after/{block}/{name}/{mode}")
                # Bootstrap latency is separate from stream timing and excludes construction/loading.
                guard(f"before/{block}/{name}/bootstrap_latency")
                model = load(directory / "initial.npz")
                samples = []
                for sample in bootstrap:
                    core.synchronize(model.device)
                    started = clock()
                    model.update(sample.embedding, sample.label)
                    core.synchronize(model.device)
                    samples.append(clock() - started)
                if _fingerprint(model) != bootstrap_fingerprints[name]:
                    raise ValueError("bootstrap latency pass changed state/RNG fidelity")
                _write_json(directory / f"block_{block}_bootstrap_latency.json",
                            {"block": block, "learner": name, "fidelity": "PASS",
                             "update_seconds": samples, **core.latency_summary(samples)})
                del model
                gc.collect()
                guard(f"after/{block}/{name}/bootstrap_latency")
                print(f"Completed paired block {block + 1}/5: {name}", flush=True)
        memory = {}
        for name in core.ORDER:
            guard("before_memory/" + name)
            directory = output_dir / name
            gc.collect()
            if not smoke:
                torch.cuda.empty_cache()
            model = load(directory / "bootstrap.npz")
            if not smoke:
                # Drop freed restoration temporaries before measuring this head.
                torch.cuda.empty_cache()
            result, peaks = environment.memory_pass(lambda: measured(model, "untimed"), spec.device,
                **({"sample": lambda: 0} if smoke else {}))
            predictions = core.final_predictions(model, [*bootstrap, *stream])
            _check_result(result, predictions, references[name])
            del model
            gc.collect()
            # A distinct inventory/serialization pass cannot pollute peak sampling.
            model = load(directory / "bootstrap.npz")
            samples = []
            sample_dir = directory / "state_inventory"
            sample_dir.mkdir()
            def inventory(learner, t):
                if t in policy["snapshot_policy"]["retention_processed_events"]:
                    full_path = sample_dir / f"step_{t:06d}_resume.npz"
                    prediction_path = sample_dir / f"step_{t:06d}_prediction.npz"
                    payload = core.payload_report(learner)
                    checkpoints.save_learner_checkpoint(full_path, learner,
                        context={"processed_stream_events": t}, shared_projection_root=output_dir)
                    prediction_bytes = core.save_prediction_payload(prediction_path, learner,
                        shared_projection_root=output_dir)
                    dependencies = checkpoint_storage.projection_dependencies(full_path,
                        shared_projection_root=output_dir)
                    shared_bytes = sum((output_dir / p).stat().st_size for p in dependencies)
                    samples.append({"processed_stream_events": t, "payload": payload,
                        "full_resume_npz_bytes": full_path.stat().st_size, "prediction_only_npz_bytes": prediction_bytes,
                        "shared_projection_dependencies": sorted(dependencies), "shared_projection_bytes": shared_bytes,
                        "full_resume_dependency_closure_bytes": full_path.stat().st_size + shared_bytes,
                        "prediction_only_dependency_closure_bytes": prediction_bytes + shared_bytes})
            inventory_result = core.stream_pass(model, stream, budget=kwargs["budget"], novel_classes=spec.novel_classes,
                selected_ids=selected, threshold=kwargs["threshold"], mode="untimed", on_observation=inventory)
            _check_result(inventory_result, core.final_predictions(model, [*bootstrap, *stream]), references[name])
            memory[name] = {"state_samples": samples,
                "sampled_max_learner_tensor_bytes": max(s["payload"]["total_exported_tensor_bytes"] for s in samples),
                "peaks": peaks, "fidelity": "PASS",
                "scope": "payload/storage at retention grid; peaks on distinct stream pass; unused CUDA cache cleared before peak pass; process/native overhead not allocated to a head"}
            _write_json(directory / "memory.json", memory[name])
            del model
            gc.collect()
            guard("after_memory/" + name)
        summaries, paired = {}, []
        for name in core.ORDER:
            values = [r["stream_events_per_second"] for r in timings if r["learner"] == name and r["mode"] == "throughput"]
            summaries[name] = {"median_events_per_second": statistics.median(values),
                               "min_events_per_second": min(values), "max_events_per_second": max(values)}
        for block in range(core.REPEATS):
            rates = {r["learner"]: r["stream_events_per_second"] for r in timings
                     if r["block"] == block and r["mode"] == "throughput"}
            paired.append({"block": block, "hdc_over_comparator": {n:rates["hdc"]/rates[n] for n in core.ORDER if n != "hdc"}})
        files = [p for p in output_dir.rglob("*") if p.is_file()]
        storage = environment.storage_report(files, runner._file_sha256)
        qualification = environment.summarize_workload_checks(checks, smoke=smoke)
        _write_json(output_dir / "summary.json", {"per_learner": summaries, "paired_ratios": paired,
            "timings": timings, "memory": memory, "storage": storage,
            "environment_qualification": qualification,
            "scope": "one development condition; timing repeats are technical, not independent streams; no predictive/retention superiority claim"})
        _unchanged(pins)
        _write_json(output_dir / "contamination_checks.json", {"checks": checks})
        receipt = {"status": "PASS", "stage": "development_only", "data_role": "synthetic_smoke" if smoke else "trusted_train",
            "paired_blocks": core.REPEATS, "learners": list(core.ORDER),
            "all_passes_fidelity": True, "inputs_unchanged": True,
            "environment_qualification": qualification,
            "artifacts": {p.relative_to(output_dir).as_posix(): runner._file_sha256(p) for p in output_dir.rglob("*") if p.is_file()},
            "scope": "development benchmark integrity, not a final thesis claim"}
        _write_json(output_dir / "integrity_report.json", receipt)
        return receipt
    except (Exception, KeyboardInterrupt) as error:
        _write_json(output_dir / "failure.json", {"status": "FAIL", "error": str(error), "type": type(error).__name__,
                    "completed_contamination_checks": checks, "instruction": "preserve attempt; no automatic resume or retry"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("A", "B"), required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--trusted-split", type=Path, required=True)
    parser.add_argument("--stream-spec", type=Path, default=runner.DEFAULT_STREAM_SPEC)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-confirmed", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    kwargs = dict(experiment=args.experiment, threshold=args.threshold, budget=args.budget,
        stream_seed=args.seed, embeddings_path=args.embeddings, split_csv=args.trusted_split,
        stream_spec_path=args.stream_spec, smoke_mode=args.smoke)
    if args.execute:
        if args.output_dir is None:
            parser.error("--execute requires a fresh output directory")
        result = run_benchmark(**kwargs, output_dir=args.output_dir, preflight_confirmed=args.preflight_confirmed)
    else:
        result = plan_benchmark(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
