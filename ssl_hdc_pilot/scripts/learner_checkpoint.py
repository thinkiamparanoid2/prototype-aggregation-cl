"""Versioned, non-pickle checkpoints for the existing continual learners.

This persistence adapter deliberately leaves scientific update code unchanged.
Artifacts contain learner-resumable state, not a resumable stream controller or
an encoder. Caller-supplied context records the stream position and provenance.
Never load checkpoints from untrusted sources; hashes detect corruption, not
authenticity. Saves refuse to overwrite an existing path.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ssl_hdc_pilot.scripts import checkpoint_storage

from ssl_hdc_pilot.scripts.continual_hdc import HDCPrototypeLearner
from ssl_hdc_pilot.scripts.continual_baselines import (
    EWCLearner, HeadTrainingConfig, NaiveIncrementalLearner, ReplayLearner,
)
from ssl_hdc_pilot.scripts.continual_prototypes import NCMPrototypeLearner, RanPACPrototypeLearner
from ssl_hdc_pilot.scripts.aggregation_prototypes import SignNCMPrototypeLearner, MedianNCMPrototypeLearner


SCHEMA_VERSION = 1
NEURAL_TYPES = {"naive": NaiveIncrementalLearner, "ewc": EWCLearner, "replay": ReplayLearner}
EWC_FIELDS = ("_ewc_means", "_ewc_penalty_fisher", "_ewc_running_fisher")


@dataclass(frozen=True)
class LearnerCheckpoint:
    learner: Any
    context: dict[str, Any]
    metadata: dict[str, Any]


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _labels(value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(label, str) or not label for label in value)
        or len(value) != len(set(value))
    ):
        raise ValueError("checkpoint classes must be unique non-empty strings")
    return value


def _array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _capture_random_state(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    arrays["rng_numpy_keys"] = numpy_state[1].copy()
    arrays["rng_torch_cpu"] = torch.get_rng_state().numpy().copy()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    for index, value in enumerate(cuda_states):
        arrays[f"rng_cuda_{index}"] = value.cpu().numpy().copy()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])],
        "cuda_count": len(cuda_states),
    }


def _read_random_state(record: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    def take(name: str, dtype: Any) -> np.ndarray:
        value = arrays.pop(name)
        if value.dtype != dtype or value.ndim != 1:
            raise ValueError(f"invalid RNG array: {name}")
        return value

    version, internal, gaussian = record["python"]
    python_state = (version, tuple(internal), gaussian)
    random.Random().setstate(python_state)  # Validate without changing global RNG.
    algorithm, position, has_gauss, cached_gauss = record["numpy"]
    numpy_state = (algorithm, take("rng_numpy_keys", np.uint32), position, has_gauss, cached_gauss)
    np.random.RandomState(0).set_state(numpy_state)
    cpu_state = torch.from_numpy(take("rng_torch_cpu", np.uint8))
    torch.Generator(device="cpu").set_state(cpu_state)
    count = record["cuda_count"]
    if type(count) is not int or count < 0:
        raise ValueError("invalid CUDA RNG count")
    cuda_states = [torch.from_numpy(take(f"rng_cuda_{index}", np.uint8)) for index in range(count)]
    return {"python": python_state, "numpy": numpy_state, "cpu": cpu_state, "cuda": cuda_states}


def _export_learner(learner: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if type(learner) not in (HDCPrototypeLearner, NCMPrototypeLearner, RanPACPrototypeLearner, SignNCMPrototypeLearner, MedianNCMPrototypeLearner, *NEURAL_TYPES.values()):
        raise TypeError("unsupported learner type for checkpoint")
    def put(name: str, tensor: torch.Tensor) -> None:
        arrays[name] = tensor.detach().cpu().numpy().copy()

    state: dict[str, Any] = {
        "input_dim": learner.input_dim,
        "classes": list(learner.known_classes),
        "source_device": str(learner.device),
    }
    if type(learner) in NEURAL_TYPES.values():
        state.update(
            learner=next(name for name, cls in NEURAL_TYPES.items() if type(learner) is cls),
            config=asdict(learner.config), learner_seed=learner._seed,
            module_modes={name: module.training for name, module in learner.model.named_modules()},
            requires_grad={name: parameter.requires_grad for name, parameter in learner.model.named_parameters()},
            gradient_names=[],
        )
        for name, value in learner.model.state_dict().items():
            put(f"model.{name}", value)
        for name, parameter in learner.model.named_parameters():
            if parameter.grad is not None:
                state["gradient_names"].append(name)
                put(f"gradient.{name}", parameter.grad)
        if type(learner) is EWCLearner:
            for field in EWC_FIELDS:
                values = getattr(learner, field)
                state[field] = list(values)
                for name, value in values.items():
                    put(f"{field}.{name}", value)
        if type(learner) is ReplayLearner:
            state["buffer"] = {
                "capacity": learner.buffer.capacity, "seen": learner.buffer._seen,
                "labels": list(learner.buffer.labels), "random_state": learner.buffer._random.getstate(),
            }
            items = learner.buffer._items
            put("replay_embeddings", torch.stack([item[0] for item in items])
                if items else torch.empty((0, learner.input_dim), dtype=torch.float32))
        return state
    if type(learner) in (NCMPrototypeLearner, RanPACPrototypeLearner):
        state.update(learner="ncm", prototype_dim=learner.prototype_dim,
                     counts=[learner._counts[label] for label in learner.known_classes])
        if type(learner) is RanPACPrototypeLearner:
            state["learner"] = "ranpac"
            put("projection", learner._projection)
        for index, label in enumerate(learner.known_classes):
            put(f"prototype_{index}", learner._prototypes[label])
        return state
    if type(learner) in (SignNCMPrototypeLearner, MedianNCMPrototypeLearner):
        state.update(learner=("sign_ncm" if type(learner) is SignNCMPrototypeLearner else "median_ncm"),
                     prototype_dim=learner.input_dim,
                     counts=[len(learner._vectors[label]) for label in learner.known_classes])
        for index, label in enumerate(learner.known_classes):
            vectors = learner.get_vectors(label)
            put(f"vectors_{index}", torch.stack(vectors) if vectors else torch.empty((0, learner.input_dim), dtype=torch.float32))
            if learner._prototypes[label] is not None:
                put(f"prototype_{index}", learner._prototypes[label])
        return state
    if type(learner) is not HDCPrototypeLearner:
        raise TypeError("unsupported learner type for checkpoint")
    state.update(learner="hdc", hypervector_dim=learner.hypervector_dim)
    put("projection", learner._projection)
    state["active_classes"] = []
    for index, label in enumerate(learner.known_classes):
        put(f"accumulator_{index}", learner._accumulators[label])
        if learner._prototypes[label] is not None:
            state["active_classes"].append(label)
            put(f"prototype_{index}", learner._prototypes[label])
    return state


def save_learner_checkpoint(
    path: Path,
    learner: Any,
    *,
    context: Mapping[str, Any] | None = None,
    shared_projection_root: Path | None = None,
) -> Path:
    """Save a detached learner snapshot without changing the learner or its RNG."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("learner checkpoints must use .npz")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite learner checkpoint: {path}")
    arrays: dict[str, np.ndarray] = {}
    state = _export_learner(learner, arrays)
    rng = _capture_random_state(arrays)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "state_kind": "learner_resumable",
        "state": state,
        "rng": rng,
        "context": dict(context or {}),
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__},
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype),
                  "sha256": _array_digest(value)}
            for key, value in arrays.items()
        },
    }
    encoded = json.dumps(metadata, allow_nan=False, sort_keys=True).encode("utf-8")
    arrays["metadata_json"] = np.frombuffer(encoded, dtype=np.uint8)
    checkpoint_storage.save_arrays(path, arrays, shared_projection_root=shared_projection_root)
    return path


def load_learner_checkpoint(
    path: Path, *, device: torch.device | str = "cpu", restore_rng: bool = False,
    shared_projection_root: Path | None = None,
) -> LearnerCheckpoint:
    """Restore an independent learner, optionally restoring saved global RNGs.

    The default leaves caller RNGs unchanged; the replay-private RNG is always
    restored. Exact continued-update fidelity is validated on CPU only. CUDA
    restoration requires the original number of visible devices.
    """
    archive = checkpoint_storage.load_arrays(Path(path), shared_projection_root=shared_projection_root)
    raw = archive["metadata_json"]
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError("invalid checkpoint metadata encoding")
    metadata = json.loads(raw.tobytes().decode("utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported learner checkpoint schema_version")
    if metadata.get("state_kind") != "learner_resumable":
        raise ValueError("unsupported learner checkpoint state_kind")
    if not isinstance(metadata.get("context"), dict):
        raise ValueError("checkpoint context must be an object")
    manifest = metadata["arrays"]
    if set(archive) != {"metadata_json", *manifest}:
        raise ValueError("checkpoint array inventory mismatch")
    arrays = {}
    for key, record in manifest.items():
        value = archive[key]
        if (list(value.shape) != record["shape"]
                or str(value.dtype) != record["dtype"]
                or _array_digest(value) != record["sha256"]):
            raise ValueError(f"checkpoint array integrity mismatch: {key}")
        arrays[key] = value

    rng = _read_random_state(metadata["rng"], arrays)
    if restore_rng and rng["cuda"] and len(rng["cuda"]) != torch.cuda.device_count():
        raise ValueError("CUDA RNG restore requires the original visible device count")
    checkpoint = _restore_learner(metadata, arrays, device)
    if restore_rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["cpu"])
        if rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
    return checkpoint


def _restore_learner(metadata: dict[str, Any], arrays: dict[str, np.ndarray], device: torch.device | str) -> LearnerCheckpoint:
    state = metadata["state"]
    classes = _labels(state["classes"])
    input_dim = _positive_int(state["input_dim"], "input_dim")
    target = torch.device(device)
    used: set[str] = set()

    def tensor(name: str, shape: tuple[int, ...]) -> torch.Tensor:
        value = arrays[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"invalid checkpoint tensor: {name}")
        used.add(name)
        return torch.from_numpy(value).to(target)

    if state["learner"] in NEURAL_TYPES:
        config = HeadTrainingConfig(**state["config"])
        config.validate()
        if type(state["learner_seed"]) is not int:
            raise ValueError("invalid learner seed")
        kwargs = dict(input_dim=input_dim, initial_classes=classes,
                      learner_seed=state["learner_seed"], config=config, device=target,
                      _checkpoint_restore=True)
        if state["learner"] == "replay":
            kwargs.update(replay_capacity=_positive_int(state["buffer"]["capacity"], "replay capacity"),
                          replay_seed=0)
        # Construction initializes weights we will replace; preserve caller RNGs.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            learner = NEURAL_TYPES[state["learner"]](**kwargs)
        learner.model.load_state_dict({
            name: tensor(f"model.{name}", tuple(value.shape))
            for name, value in learner.model.state_dict().items()
        }, strict=True)
        modules = dict(learner.model.named_modules())
        modes = state["module_modes"]
        if set(modes) != set(modules) or any(type(flag) is not bool for flag in modes.values()):
            raise ValueError("invalid module modes")
        for name, module in modules.items():
            module.training = modes[name]
        parameters = dict(learner.model.named_parameters())
        flags = state["requires_grad"]
        gradients = _labels(state["gradient_names"])
        if (set(flags) != set(parameters) or any(type(flag) is not bool for flag in flags.values())
                or not set(gradients).issubset(parameters)):
            raise ValueError("invalid parameter/gradient metadata")
        for name, parameter in parameters.items():
            parameter.requires_grad_(flags[name])
            if name in gradients:
                parameter.grad = tensor(f"gradient.{name}", tuple(parameter.shape))
        if state["learner"] == "ewc":
            for field in EWC_FIELDS:
                names = _labels(state[field])
                if names and set(names) != set(parameters):
                    raise ValueError("incomplete EWC state")
                setattr(learner, field, {
                    name: tensor(f"{field}.{name}", tuple(parameters[name].shape)) for name in names
                })
            if set(learner._ewc_means) != set(learner._ewc_penalty_fisher):
                raise ValueError("EWC anchors and penalty Fisher must align")
        if state["learner"] == "replay":
            buffer = state["buffer"]
            labels = buffer["labels"]
            seen = buffer["seen"]
            if (not isinstance(labels, list) or any(label not in classes for label in labels)
                    or type(seen) is not int or seen < 0
                    or len(labels) != min(seen, learner.buffer.capacity)):
                raise ValueError("invalid replay occupancy, labels, or seen count")
            vectors = tensor("replay_embeddings", (len(labels), input_dim)).cpu()
            learner.buffer._items = [(vector.clone(), label) for vector, label in zip(vectors, labels)]
            learner.buffer._seen = seen
            version, internal, gaussian = buffer["random_state"]
            learner.buffer._random.setstate((version, tuple(internal), gaussian))
        if used != set(arrays):
            raise ValueError("unexpected learner checkpoint arrays")
        return LearnerCheckpoint(learner, metadata["context"], metadata)
    if state["learner"] in ("ncm", "ranpac"):
        dim = _positive_int(state["prototype_dim"], "prototype_dim")
        if state["learner"] == "ncm":
            if dim != input_dim:
                raise ValueError("NCM prototype dimension must equal input dimension")
            learner = NCMPrototypeLearner(input_dim=input_dim, device=target)
        else:
            learner = RanPACPrototypeLearner(input_dim=input_dim, projection_dim=dim,
                                            projection_seed=0, device=target)
            learner._projection = tensor("projection", (input_dim, dim))
        counts = state["counts"]
        if len(counts) != len(classes) or any(type(n) is not int or n < 0 for n in counts):
            raise ValueError("invalid prototype counts")
        for index, (label, count) in enumerate(zip(classes, counts)):
            learner.register_class(label)
            learner._counts[label] = count
            learner._prototypes[label] = tensor(f"prototype_{index}", (dim,))
        if used != set(arrays):
            raise ValueError("unexpected learner checkpoint arrays")
        return LearnerCheckpoint(learner, metadata["context"], metadata)
    if state["learner"] in ("sign_ncm", "median_ncm"):
        dim = _positive_int(state["prototype_dim"], "prototype_dim")
        if dim != input_dim:
            raise ValueError("native prototype dimension must equal input dimension")
        learner = (SignNCMPrototypeLearner if state["learner"] == "sign_ncm" else MedianNCMPrototypeLearner)(input_dim=input_dim, device=target)
        counts = state["counts"]
        if len(counts) != len(classes) or any(type(n) is not int or n < 0 for n in counts):
            raise ValueError("invalid native prototype counts")
        for index, (label, count) in enumerate(zip(classes, counts)):
            learner.register_class(label)
            vectors = tensor(f"vectors_{index}", (count, input_dim))
            learner._vectors[label] = [row.clone() for row in vectors]
            if count:
                learner._prototypes[label] = tensor(f"prototype_{index}", (input_dim,))
            elif f"prototype_{index}" in arrays:
                raise ValueError("empty native class must not have a prototype")
        if used != set(arrays):
            raise ValueError("unexpected learner checkpoint arrays")
        return LearnerCheckpoint(learner, metadata["context"], metadata)
    if state["learner"] != "hdc":
        raise ValueError("unsupported checkpoint learner")
    dim = _positive_int(state["hypervector_dim"], "hypervector_dim")
    learner = HDCPrototypeLearner(input_dim=input_dim, hypervector_dim=dim,
                                  projection_seed=0, device=target)
    learner._projection = tensor("projection", (input_dim, dim))
    active = set(_labels(state["active_classes"]))
    if not active.issubset(classes):
        raise ValueError("active classes are absent from checkpoint registry")
    for index, label in enumerate(classes):
        learner.register_class(label)
        learner._accumulators[label] = tensor(f"accumulator_{index}", (dim,))
        if label in active:
            prototype = tensor(f"prototype_{index}", (dim,))
            if not torch.all((prototype == -1) | (prototype == 1)):
                raise ValueError("HDC checkpoint prototype must be bipolar")
            learner._prototypes[label] = prototype
    if used != set(arrays):
        raise ValueError("unexpected learner checkpoint arrays")
    return LearnerCheckpoint(learner, metadata["context"], metadata)
