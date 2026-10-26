"""Timing boundary and payload helpers; no input loading or run authorization."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ssl_hdc_pilot.scripts import learner_checkpoint as checkpoints
from ssl_hdc_pilot.scripts import checkpoint_storage


ORDER = ("hdc", "ncm", "ranpac", "naive", "ewc", "replay")
REPEATS = 5


def block_order(block: int) -> tuple[str, ...]:
    if type(block) is not int or not 0 <= block < REPEATS:
        raise ValueError("block must be in 0..4")
    return ORDER[block:] + ORDER[:block]


def synchronize(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def latency_summary(seconds: list[float]) -> dict:
    if any(not math.isfinite(x) or x <= 0 for x in seconds):
        raise ValueError("update durations must be finite and positive")
    return {"count": len(seconds), "total_seconds": sum(seconds),
            "median_seconds": float(np.median(seconds)) if seconds else None,
            "p95_seconds": float(np.percentile(seconds, 95)) if seconds else None,
            "updates_per_second": len(seconds) / sum(seconds) if seconds else None}


def stream_pass(learner: Any, stream, *, budget: int, novel_classes,
                selected_ids: frozenset[str] | None, threshold: float | None,
                mode: str, clock: Callable = time.perf_counter,
                sync: Callable = synchronize, on_observation: Callable | None = None) -> dict:
    """No scoring/serialization/rolling metrics inside the measured region.

    Host input vectors reach original predict/update methods, so their existing
    transfer behavior is included. Minimal decision trace bookkeeping is included
    consistently; JSON formatting, hashing and all file I/O happen afterward.
    """
    if mode not in {"throughput", "latency", "untimed"}:
        raise ValueError("invalid pass mode")
    if on_observation is not None and mode != "untimed":
        raise ValueError("state inventory hooks are forbidden in timing passes")
    if (type(budget) is not int or budget < 0 or not stream
            or any(s.embedding.device.type != "cpu" for s in stream)):
        raise ValueError("nonempty host-resident stream and nonnegative budget required")
    if selected_ids is None and (threshold is None or not math.isfinite(threshold)):
        raise ValueError("B requires finite threshold")
    if selected_ids is not None and (threshold is not None or len(selected_ids) != budget
                                     or not selected_ids <= {s.image_id for s in stream}):
        raise ValueError("A requires exact scheduled budget and no threshold")
    confirmed, trace, latencies = set(), [], []
    spent = 0
    if on_observation is not None:
        on_observation(learner, 0)
    if mode == "throughput":
        sync(learner.device)
        started = clock()
    for step, sample in enumerate(stream):
        prediction = learner.predict(sample.embedding)
        requested = (sample.image_id in selected_ids if selected_ids is not None else
                     prediction.confidence < threshold or
                     (sample.label in novel_classes and sample.label not in confirmed))
        purchased = bool(requested and spent < budget)
        registered = False
        if purchased:
            if mode == "latency":
                sync(learner.device)
                update_start = clock()
            result = learner.update(sample.embedding, sample.label)
            if mode == "latency":
                sync(learner.device)
                latencies.append(clock() - update_start)
            registered = bool(result.class_registered)
            spent += 1
            if sample.label in novel_classes:
                confirmed.add(sample.label)
        trace.append({"step": step, "image_id": sample.image_id, "true_label": sample.label,
            "prediction": prediction.label, "confidence": round(float(prediction.confidence), 9),
            "top_probability": round(float(prediction.top_probability), 9),
            "label_requested": purchased, "label_request_denied_budget": bool(requested and not purchased),
            "class_registered": registered})
        if on_observation is not None:
            on_observation(learner, step + 1)
    elapsed = None
    if mode == "throughput":
        sync(learner.device)
        elapsed = clock() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("invalid stream duration")
    return {"mode": mode, "stream_steps": len(stream), "labels_consumed": spent,
            "elapsed_stream_seconds": elapsed,
            "stream_events_per_second": len(stream) / elapsed if elapsed else None,
            "update_latency": latency_summary(latencies) if mode == "latency" else None,
            "update_seconds": latencies, "trace": trace}


def final_predictions(learner, population) -> list[dict]:
    rows = []
    for sample in population:
        p = learner.predict(sample.embedding)
        rows.append({"image_id": sample.image_id, "true_label": sample.label,
                     "prediction": p.label, "correct": p.label == sample.label,
                     "top_probability": round(float(p.top_probability), 9)})
    return rows


def _payload_parts(learner):
    arrays = {}
    state = checkpoints._export_learner(learner, arrays)
    prediction_keys = {"learner", "input_dim", "classes", "source_device", "prototype_dim",
                       "counts", "hypervector_dim", "active_classes"}
    prediction_metadata = {k:v for k,v in state.items() if k in prediction_keys}
    extra_metadata = {k:v for k,v in state.items() if k not in prediction_keys and k != "buffer"}
    if "config" in state:
        prediction_metadata["head_type"] = state["config"]["head_type"]
        prediction_metadata["hidden_dim"] = state["config"]["hidden_dim"]
        extra_metadata["config"] = {k:v for k,v in state["config"].items() if k not in {"head_type", "hidden_dim"}}
    rng_arrays = {}
    rng = checkpoints._capture_random_state(rng_arrays)
    return state, arrays, prediction_metadata, extra_metadata, rng, rng_arrays


def payload_report(learner) -> dict:
    """Logical state payload, NOT Python object overhead or process/device peak."""
    state, arrays, prediction_metadata, extra_metadata, rng, rng_arrays = _payload_parts(learner)
    prediction, extra, replay = {}, {}, {}
    for key, value in arrays.items():
        if key == "replay_embeddings":
            replay[key] = int(value.nbytes)
        elif key == "projection" or key.startswith(("prototype_", "model.")):
            prediction[key] = int(value.nbytes)
        else:
            extra[key] = int(value.nbytes)
    buffer = state.get("buffer")
    return {"prediction_tensor_bytes": sum(prediction.values()),
        "extra_resumable_tensor_bytes": sum(extra.values()),
        "replay_embedding_bytes": sum(replay.values()),
        "total_exported_tensor_bytes": sum(int(x.nbytes) for x in arrays.values()),
        "tensor_inventory": {"prediction": prediction, "extra_resumable": extra, "replay": replay},
        "state_metadata_json_bytes": len(json.dumps(state, sort_keys=True).encode("utf-8")),
        "prediction_metadata_json_bytes": len(json.dumps(prediction_metadata, sort_keys=True).encode("utf-8")),
        "extra_resumable_metadata_json_bytes": len(json.dumps(extra_metadata, sort_keys=True).encode("utf-8")),
        "global_rng_tensor_bytes": sum(int(x.nbytes) for x in rng_arrays.values()),
        "global_rng_metadata_json_bytes": len(json.dumps(rng, sort_keys=True).encode("utf-8")),
        "replay_metadata_json_bytes": len(json.dumps(buffer, sort_keys=True).encode("utf-8")) if buffer else 0,
        "replay": {"capacity": buffer["capacity"], "occupancy": len(buffer["labels"]),
                   "seen": buffer["seen"],
                   "label_utf8_bytes": sum(len(x.encode("utf-8")) for x in buffer["labels"])} if buffer else None,
        "scope": "disjoint learner tensor categories; global RNG separately allocated; JSON encoding sizes are not Python/native runtime overhead"}


def save_prediction_payload(path: Path, learner, *, shared_projection_root: Path | None = None) -> int:
    """Measured prediction-only export, not accepted by the resumable-state loader."""
    _, arrays, metadata, _, _, _ = _payload_parts(learner)
    selected = {k:v for k,v in arrays.items() if k == "projection" or k.startswith(("prototype_", "model."))}
    selected["metadata_json"] = np.frombuffer(json.dumps({"schema_version": 1,
        "artifact_kind": "prediction_payload_export", "state": metadata}, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    checkpoint_storage.save_arrays(path, selected, shared_projection_root=shared_projection_root)
    return path.stat().st_size
