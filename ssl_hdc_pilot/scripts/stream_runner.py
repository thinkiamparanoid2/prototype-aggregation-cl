"""Run the preregistered continual-learning simulator on frozen NPZ embeddings."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ssl_hdc_pilot.scripts.build_trusted_splits import (
    CANONICAL_SOURCE_CSV,
    OFFICIAL_SEED,
    PROVENANCE_VERSION,
)
from ssl_hdc_pilot.scripts.build_stream_spec import (
    APPROVED_POLICY,
    DEFAULT_SPLIT_DIR as DEFAULT_APPROVED_SPLIT_DIR,
    derive_section6a_plan,
    load_approved_train,
)
from ssl_hdc_pilot.scripts.continual_baselines import (
    EWCLearner,
    HeadTrainingConfig,
    NaiveIncrementalLearner,
    ReplayLearner,
)
from ssl_hdc_pilot.scripts.continual_hdc import HDCPrototypeLearner, HYPERVECTOR_DIM
from ssl_hdc_pilot.scripts.continual_prototypes import (
    NCMPrototypeLearner,
    RanPACPrototypeLearner,
)
from ssl_hdc_pilot.scripts.aggregation_prototypes import SignNCMPrototypeLearner, MedianNCMPrototypeLearner
from ssl_hdc_pilot.scripts.continual_metrics import build_summary, rolling_per_class_accuracy
from ssl_hdc_pilot.scripts.learner_checkpoint import save_learner_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STREAM_SPEC = REPO_ROOT / "ssl_hdc_pilot" / "stream_spec.json"
LEARNER_NAMES = ("hdc", "ncm", "ranpac", "naive", "ewc", "replay")
ABLATION_LEARNER_NAMES = ("sign_ncm", "median_ncm")
LABEL_ARRAY_NAMES = {"label", "labels", "target", "targets", "crop_disease_stage"}


@dataclass(frozen=True)
class ClassIntroduction:
    class_label: str
    at_step: int


@dataclass(frozen=True)
class StreamSpec:
    path: Path
    sha256: str
    tag: str
    smoke: bool
    device: str
    seeds: Mapping[str, int]
    stream_order_seeds: tuple[int, ...]
    initially_known_classes: tuple[str, ...]
    class_introductions: tuple[ClassIntroduction, ...]
    bootstrap_confirmed_per_class: int | None
    held_back_image_ids_by_seed: Mapping[int, frozenset[str]] | None
    bootstrap_counts_toward_budget: bool
    confidence_thresholds: tuple[float, ...]
    max_total_requests: int
    count_low_confidence: bool
    count_novel_registration: bool
    expected_stream_length: int | None
    rolling_window: int
    hypervector_dim: int
    baseline_config: HeadTrainingConfig
    replay_buffer_size: int
    stabilization_window: int
    stabilization_min_accuracy: float
    stabilization_consecutive_windows: int
    ece_bins: int

    @property
    def novel_classes(self) -> tuple[str, ...]:
        return tuple(item.class_label for item in self.class_introductions)


@dataclass(frozen=True)
class StreamSample:
    image_id: str
    label: str
    embedding: torch.Tensor


@dataclass(frozen=True)
class StudyArtifact:
    run_tag: str
    history_csv: Path
    summary_json: Path
    final_state_npz: Path | None = None


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} keys mismatch; missing={missing}, extra={extra}")


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _class_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    labels = tuple(str(item) for item in value)
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError(f"{name} must contain unique non-empty strings")
    return labels


def _parse_stream_spec(path: Path, payload: Any) -> StreamSpec:
    root = _mapping(payload, "stream spec")
    _exact_keys(
        root,
        {
            "schema_version",
            "tag",
            "smoke",
            "execution",
            "seeds",
            "protocol",
            "hdc",
            "baseline",
            "metrics",
        },
        "stream spec",
    )
    if root["schema_version"] != 1:
        raise ValueError("stream spec schema_version must be 1")
    tag = root["tag"]
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError("stream spec tag must be filesystem-safe")
    if not isinstance(root["smoke"], bool):
        raise ValueError("stream spec smoke must be boolean")

    execution = _mapping(root["execution"], "execution")
    _exact_keys(execution, {"device"}, "execution")
    device = execution["device"]
    if device not in {"cpu", "cuda"}:
        raise ValueError("execution.device must be cpu or cuda")

    seeds = _mapping(root["seeds"], "seeds")
    _exact_keys(
        seeds,
        {"stream_order", "projection", "learner", "replay"},
        "seeds",
    )
    raw_stream_seeds = seeds["stream_order"]
    if not isinstance(raw_stream_seeds, list) or not raw_stream_seeds:
        raise ValueError("seeds.stream_order must be a non-empty list")
    stream_order_seeds = tuple(
        _integer(value, f"seeds.stream_order[{index}]")
        for index, value in enumerate(raw_stream_seeds)
    )
    if len(stream_order_seeds) != len(set(stream_order_seeds)):
        raise ValueError("seeds.stream_order values must be unique")
    parsed_seeds = {
        name: _integer(seeds[name], f"seeds.{name}")
        for name in ("projection", "learner", "replay")
    }

    protocol = _mapping(root["protocol"], "protocol")
    protocol_keys = {
            "initially_known_classes",
            "class_introductions",
            "bootstrap",
            "confidence_thresholds",
            "label_budget",
            "rolling_window",
    }
    if "stream_length" in protocol:
        protocol_keys.add("stream_length")
    _exact_keys(protocol, protocol_keys, "protocol")
    initially_known = _class_list(
        protocol["initially_known_classes"], "protocol.initially_known_classes"
    )
    raw_introductions = protocol["class_introductions"]
    if not isinstance(raw_introductions, list) or not raw_introductions:
        raise ValueError("protocol.class_introductions must be a non-empty list")
    introductions: list[ClassIntroduction] = []
    for index, raw in enumerate(raw_introductions):
        item = _mapping(raw, f"class_introductions[{index}]")
        _exact_keys(item, {"class_label", "at_step"}, f"class_introductions[{index}]")
        label = item["class_label"]
        if not isinstance(label, str) or not label:
            raise ValueError("introduced class labels must be non-empty strings")
        introductions.append(
            ClassIntroduction(
                class_label=label,
                at_step=_integer(item["at_step"], f"class_introductions[{index}].at_step"),
            )
        )
    introduced_labels = [item.class_label for item in introductions]
    if len(introduced_labels) != len(set(introduced_labels)):
        raise ValueError("introduced class labels must be unique")
    if set(introduced_labels) & set(initially_known):
        raise ValueError("introduced classes must not be initially known")

    bootstrap = _mapping(protocol["bootstrap"], "protocol.bootstrap")
    held_back_image_ids_by_seed: dict[int, frozenset[str]] | None = None
    if "confirmed_per_class" in bootstrap:
        _exact_keys(
            bootstrap,
            {"confirmed_per_class", "counts_toward_budget"},
            "protocol.bootstrap",
        )
        confirmed_per_class: int | None = _integer(
            bootstrap["confirmed_per_class"],
            "protocol.bootstrap.confirmed_per_class",
            minimum=1,
        )
    else:
        _exact_keys(
            bootstrap,
            {
                "known_class_holdback_fraction",
                "held_back_image_ids_by_seed",
                "counts_toward_budget",
            },
            "protocol.bootstrap",
        )
        fraction = _number(
            bootstrap["known_class_holdback_fraction"],
            "protocol.bootstrap.known_class_holdback_fraction",
        )
        if fraction != 0.25:
            raise ValueError("official known-class holdback fraction must be exactly 0.25")
        raw_manifests = _mapping(
            bootstrap["held_back_image_ids_by_seed"],
            "protocol.bootstrap.held_back_image_ids_by_seed",
        )
        if set(raw_manifests) != {str(seed) for seed in stream_order_seeds}:
            raise ValueError("holdback manifests must exactly cover stream-order seeds")
        held_back_image_ids_by_seed = {}
        for seed in stream_order_seeds:
            raw_ids = raw_manifests[str(seed)]
            if not isinstance(raw_ids, list) or any(
                not isinstance(image_id, str) or not image_id for image_id in raw_ids
            ):
                raise ValueError("holdback manifests must contain image-id lists")
            if len(raw_ids) != len(set(raw_ids)):
                raise ValueError("holdback manifest image ids must be unique")
            held_back_image_ids_by_seed[seed] = frozenset(raw_ids)
        confirmed_per_class = None
    if not isinstance(bootstrap["counts_toward_budget"], bool):
        raise ValueError("protocol.bootstrap.counts_toward_budget must be boolean")

    raw_thresholds = protocol["confidence_thresholds"]
    if not isinstance(raw_thresholds, list) or not raw_thresholds:
        raise ValueError("protocol.confidence_thresholds must be a non-empty list")
    thresholds = tuple(
        _number(value, f"protocol.confidence_thresholds[{index}]")
        for index, value in enumerate(raw_thresholds)
    )
    if any(value < 0 or value > 2 for value in thresholds):
        raise ValueError("similarity-margin thresholds must be within [0, 2]")
    if len(thresholds) != len(set(thresholds)):
        raise ValueError("confidence thresholds must be unique")

    budget = _mapping(protocol["label_budget"], "protocol.label_budget")
    if set(budget) == {"max_total_requests"}:
        count_low_confidence = True
        count_novel_registration = False
    else:
        _exact_keys(
            budget,
            {"max_total_requests", "count_low_confidence", "count_novel_registration"},
            "protocol.label_budget",
        )
        if (
            budget["count_low_confidence"] is not True
            or budget["count_novel_registration"] is not True
        ):
            raise ValueError("Section 6a requires both confirmed-event accounting rules")
        count_low_confidence = True
        count_novel_registration = True
    max_total_requests = _integer(
        budget["max_total_requests"],
        "protocol.label_budget.max_total_requests",
    )
    expected_stream_length = (
        _integer(protocol["stream_length"], "protocol.stream_length", minimum=1)
        if "stream_length" in protocol
        else None
    )

    hdc = _mapping(root["hdc"], "hdc")
    _exact_keys(hdc, {"hypervector_dim"}, "hdc")
    hypervector_dim = _integer(hdc["hypervector_dim"], "hdc.hypervector_dim", minimum=1)
    if hypervector_dim != HYPERVECTOR_DIM:
        raise ValueError("official HDC design is fixed to 10,000 dimensions")

    baseline = _mapping(root["baseline"], "baseline")
    _exact_keys(
        baseline,
        {
            "head_type",
            "hidden_dim",
            "learning_rate",
            "weight_decay",
            "update_steps",
            "ewc_lambda",
            "ewc_decay",
            "replay_buffer_size",
            "replay_batch_size",
        },
        "baseline",
    )
    hidden_dim = baseline["hidden_dim"]
    if hidden_dim is not None:
        hidden_dim = _integer(hidden_dim, "baseline.hidden_dim", minimum=1)
    baseline_config = HeadTrainingConfig(
        head_type=str(baseline["head_type"]),
        hidden_dim=hidden_dim,
        learning_rate=_number(baseline["learning_rate"], "baseline.learning_rate"),
        weight_decay=_number(baseline["weight_decay"], "baseline.weight_decay"),
        update_steps=_integer(baseline["update_steps"], "baseline.update_steps", minimum=1),
        ewc_lambda=_number(baseline["ewc_lambda"], "baseline.ewc_lambda"),
        ewc_decay=_number(baseline["ewc_decay"], "baseline.ewc_decay"),
        replay_batch_size=_integer(
            baseline["replay_batch_size"], "baseline.replay_batch_size", minimum=1
        ),
    )
    baseline_config.validate()
    if baseline_config.ewc_lambda <= 0:
        raise ValueError("baseline.ewc_lambda must be positive for a real EWC comparator")
    replay_buffer_size = _integer(
        baseline["replay_buffer_size"], "baseline.replay_buffer_size", minimum=1
    )

    metrics = _mapping(root["metrics"], "metrics")
    _exact_keys(
        metrics,
        {
            "stabilization_window",
            "stabilization_min_accuracy",
            "stabilization_consecutive_windows",
            "ece_bins",
        },
        "metrics",
    )
    stabilization_min_accuracy = _number(
        metrics["stabilization_min_accuracy"], "metrics.stabilization_min_accuracy"
    )
    if not 0 <= stabilization_min_accuracy <= 1:
        raise ValueError("stabilization_min_accuracy must be within [0, 1]")

    raw_bytes = path.read_bytes()
    return StreamSpec(
        path=path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        tag=tag,
        smoke=root["smoke"],
        device=device,
        seeds=parsed_seeds,
        stream_order_seeds=stream_order_seeds,
        initially_known_classes=initially_known,
        class_introductions=tuple(introductions),
        bootstrap_confirmed_per_class=confirmed_per_class,
        held_back_image_ids_by_seed=held_back_image_ids_by_seed,
        bootstrap_counts_toward_budget=bootstrap["counts_toward_budget"],
        confidence_thresholds=thresholds,
        max_total_requests=max_total_requests,
        count_low_confidence=count_low_confidence,
        count_novel_registration=count_novel_registration,
        expected_stream_length=expected_stream_length,
        rolling_window=_integer(protocol["rolling_window"], "protocol.rolling_window", minimum=1),
        hypervector_dim=hypervector_dim,
        baseline_config=baseline_config,
        replay_buffer_size=replay_buffer_size,
        stabilization_window=_integer(
            metrics["stabilization_window"], "metrics.stabilization_window", minimum=1
        ),
        stabilization_min_accuracy=stabilization_min_accuracy,
        stabilization_consecutive_windows=_integer(
            metrics["stabilization_consecutive_windows"],
            "metrics.stabilization_consecutive_windows",
            minimum=1,
        ),
        ece_bins=_integer(metrics["ece_bins"], "metrics.ece_bins", minimum=2),
    )


def load_stream_spec(path: Path, *, smoke_mode: bool) -> StreamSpec:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"stream spec is required and was not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid stream spec JSON: {path}") from error
    spec = _parse_stream_spec(path, payload)
    if smoke_mode:
        if not spec.smoke:
            raise RuntimeError("smoke path requires a spec with smoke=true")
    else:
        if spec.smoke:
            raise RuntimeError("official path refuses smoke=true stream specs")
        if path.resolve() != DEFAULT_STREAM_SPEC.resolve():
            raise RuntimeError(f"official path requires canonical spec: {DEFAULT_STREAM_SPEC}")
        if spec.held_back_image_ids_by_seed is None or spec.expected_stream_length is None:
            raise RuntimeError("official path requires a generated Section 6a holdback manifest")
        if spec.stream_order_seeds != (0, 1, 2):
            raise RuntimeError("official Section 6a stream seeds must be exactly 0, 1, and 2")
        if spec.confidence_thresholds != (0.05, 0.10, 0.20):
            raise RuntimeError("official Section 6a thresholds must be exactly 0.05, 0.10, 0.20")
        if len(spec.initially_known_classes) != 15 or len(spec.novel_classes) != 4:
            raise RuntimeError("official Section 6a class composition must be 15 known plus 4 novel")
        expected_steps = tuple(
            int(spec.expected_stream_length * fraction)
            for fraction in (0.20, 0.40, 0.60, 0.80)
        )
        if tuple(item.at_step for item in spec.class_introductions) != expected_steps:
            raise RuntimeError("official Section 6a novel entry steps do not match 20/40/60/80%")
        if (
            not spec.count_low_confidence
            or not spec.count_novel_registration
            or spec.bootstrap_counts_toward_budget
        ):
            raise RuntimeError("official Section 6a confirmed-label accounting is invalid")
    return spec


def _load_embedding_map(path: Path) -> tuple[dict[str, torch.Tensor], int]:
    if not path.is_file():
        raise FileNotFoundError(f"embedding NPZ not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        forbidden = keys & LABEL_ARRAY_NAMES
        if forbidden:
            raise RuntimeError(
                f"embedding NPZ must be label-free; forbidden arrays: {sorted(forbidden)}"
            )
        if not {"image_ids", "embeddings"}.issubset(keys):
            raise ValueError("embedding NPZ requires image_ids and embeddings arrays")
        image_ids = np.asarray(payload["image_ids"]).astype(str)
        embeddings = np.asarray(payload["embeddings"])
    if image_ids.ndim != 1 or embeddings.ndim != 2 or len(image_ids) != len(embeddings):
        raise ValueError("embedding NPZ arrays have incompatible shapes")
    if embeddings.dtype != np.float32:
        raise ValueError("embeddings must be float32")
    if embeddings.shape[1] < 1 or not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite and non-empty")
    ids = image_ids.tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("embedding image_ids must be unique")
    return {
        image_id: torch.from_numpy(embeddings[index].copy())
        for index, image_id in enumerate(ids)
    }, int(embeddings.shape[1])


def _load_split_rows(path: Path, *, smoke_mode: bool) -> list[tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"trusted split CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("trusted split CSV is empty")
    if not smoke_mode:
        for row in rows:
            if (row.get("supervisor_approved") or "").lower() != "true":
                raise RuntimeError("official stream requires supervisor-approved split provenance")
            if row.get("split_partition") != "train":
                raise RuntimeError("official stream requires the approved Option-A train partition")
            if row.get("split_policy") != APPROVED_POLICY:
                raise RuntimeError("official stream requires the approved Option-A split policy")
            if row.get("split_seed") != str(OFFICIAL_SEED):
                raise RuntimeError("official stream requires the approved Option-A split seed")
            if row.get("provenance_version") != PROVENANCE_VERSION:
                raise RuntimeError("official stream split provenance version mismatch")
            if row.get("source_csv") != CANONICAL_SOURCE_CSV:
                raise RuntimeError("official stream split must derive from canonical Data/test.csv")
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        image_id = (row.get("image") or row.get("image_id") or "").strip()
        label = (row.get("label") or row.get("crop_disease_stage") or "").strip()
        if not image_id or not label:
            raise ValueError("trusted split rows require image/image_id and label")
        if image_id in seen:
            raise ValueError(f"duplicate trusted split image id: {image_id}")
        seen.add(image_id)
        records.append((image_id, label))
    return records


def _validate_official_section6a(
    *, spec: StreamSpec, split_csv: Path, split_rows: Sequence[tuple[str, str]]
) -> None:
    """Require the official spec to equal the locked generator's derivation."""
    canonical_train = DEFAULT_APPROVED_SPLIT_DIR / "train.csv"
    if Path(split_csv).resolve() != canonical_train.resolve():
        raise RuntimeError(
            f"official stream requires canonical approved Option-A split: {canonical_train}"
        )
    approved_rows = load_approved_train(DEFAULT_APPROVED_SPLIT_DIR)
    if list(split_rows) != approved_rows:
        raise RuntimeError("official stream rows do not match the canonical approved train split")
    plan = derive_section6a_plan(DEFAULT_APPROVED_SPLIT_DIR, approved_rows)
    if spec.initially_known_classes != plan.known_classes:
        raise RuntimeError(
            "official stream spec known/novel classes do not match Section 6a rarity rules"
        )
    if spec.novel_classes != plan.entry_order:
        raise RuntimeError(
            "official stream spec novel entry order does not match Section 6a rules"
        )
    if tuple(item.at_step for item in spec.class_introductions) != plan.introduction_steps:
        raise RuntimeError(
            "official stream spec introduction steps do not match Section 6a rules"
        )
    if spec.expected_stream_length != plan.stream_length:
        raise RuntimeError(
            "official stream spec stream length does not match Section 6a derivation"
        )
    assert spec.held_back_image_ids_by_seed is not None
    for seed in spec.stream_order_seeds:
        if spec.held_back_image_ids_by_seed[seed] != frozenset(plan.manifests[str(seed)]):
            raise RuntimeError(
                "official stream spec holdback manifest does not match the deterministic "
                "25% per-known-class Section 6a derivation"
            )


def _prepare_stream(
    *,
    split_rows: Sequence[tuple[str, str]],
    embedding_map: Mapping[str, torch.Tensor],
    spec: StreamSpec,
    stream_seed: int,
) -> tuple[list[StreamSample], list[StreamSample]]:
    allowed_classes = set(spec.initially_known_classes) | set(spec.novel_classes)
    observed_classes = {label for _, label in split_rows}
    if observed_classes != allowed_classes:
        raise RuntimeError(
            "stream spec classes must exactly match trusted split labels; "
            f"spec_only={sorted(allowed_classes - observed_classes)}, "
            f"split_only={sorted(observed_classes - allowed_classes)}"
        )
    missing_embeddings = sorted(image_id for image_id, _ in split_rows if image_id not in embedding_map)
    if missing_embeddings:
        raise RuntimeError(
            f"trusted split has {len(missing_embeddings)} ids missing from embedding NPZ"
        )

    class_order = [*spec.initially_known_classes, *spec.novel_classes]
    groups: dict[str, list[StreamSample]] = {label: [] for label in class_order}
    for image_id, label in split_rows:
        groups[label].append(StreamSample(image_id, label, embedding_map[image_id]))
    randomizer = random.Random(stream_seed)
    for label in class_order:
        groups[label].sort(key=lambda item: item.image_id)
        randomizer.shuffle(groups[label])

    bootstrap: list[StreamSample] = []
    if spec.held_back_image_ids_by_seed is not None:
        held_back_ids = spec.held_back_image_ids_by_seed[stream_seed]
        split_labels_by_id = {image_id: label for image_id, label in split_rows}
        invalid_ids = sorted(
            image_id
            for image_id in held_back_ids
            if split_labels_by_id.get(image_id) not in spec.initially_known_classes
        )
        if invalid_ids:
            raise RuntimeError("holdback manifest contains missing or non-known image ids")
        for label in spec.initially_known_classes:
            known_group = groups[label]
            bootstrap.extend(item for item in known_group if item.image_id not in held_back_ids)
            groups[label] = [item for item in known_group if item.image_id in held_back_ids]
        if sum(len(groups[label]) for label in spec.initially_known_classes) != len(held_back_ids):
            raise RuntimeError("holdback manifest does not map one-to-one to known-class rows")
        randomizer.shuffle(bootstrap)
    else:
        assert spec.bootstrap_confirmed_per_class is not None
        for index in range(spec.bootstrap_confirmed_per_class):
            for label in spec.initially_known_classes:
                if not groups[label]:
                    raise RuntimeError(
                        f"class {label} lacks bootstrap confirmed example {index + 1}"
                    )
                bootstrap.append(groups[label].pop())

    remaining_count = sum(len(group) for group in groups.values())
    if (
        spec.expected_stream_length is not None
        and remaining_count != spec.expected_stream_length
    ):
        raise RuntimeError(
            "generated holdback manifest does not match preregistered stream length"
        )
    introduction_at: dict[int, list[str]] = {}
    for item in spec.class_introductions:
        introduction_at.setdefault(item.at_step, []).append(item.class_label)
    if any(step >= remaining_count for step in introduction_at):
        raise RuntimeError("class introduction step falls outside the post-bootstrap stream")

    active = list(spec.initially_known_classes)
    stream: list[StreamSample] = []
    for step in range(remaining_count):
        introduced_now = introduction_at.get(step, [])
        for introduced_label in introduced_now:
            active.append(introduced_label)
            if not groups[introduced_label]:
                raise RuntimeError(
                    f"introduced class has no stream examples: {introduced_label}"
                )
        candidates = [label for label in introduced_now if groups[label]]
        if not candidates:
            candidates = [label for label in active if groups[label]]
        if not candidates:
            raise RuntimeError(
                "stream schedule is infeasible: active classes exhausted before next introduction"
            )
        label = randomizer.choice(candidates)
        stream.append(groups[label].pop())
    if any(groups.values()):
        raise RuntimeError("internal stream construction failed to consume every trusted sample")
    return bootstrap, stream


def _build_learner(name: str, *, spec: StreamSpec, input_dim: int,
                   ablation_mode: bool = False, hypervector_dim: int | None = None) -> Any:
    device = torch.device(spec.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("stream spec requires CUDA but CUDA is unavailable")
    common = {
        "input_dim": input_dim,
        "initial_classes": spec.initially_known_classes,
        "learner_seed": spec.seeds["learner"],
        "config": spec.baseline_config,
        "device": device,
    }
    if name == "hdc":
        return HDCPrototypeLearner(
            input_dim=input_dim,
            projection_seed=spec.seeds["projection"],
            hypervector_dim=hypervector_dim if hypervector_dim is not None else spec.hypervector_dim,
            device=device,
        )
    if name == "ncm":
        return NCMPrototypeLearner(
            input_dim=input_dim,
            device=device,
        )
    if name == "ranpac":
        return RanPACPrototypeLearner(
            input_dim=input_dim,
            projection_seed=spec.seeds["projection"],
            projection_dim=spec.hypervector_dim,
            device=device,
        )
    if name == "naive":
        return NaiveIncrementalLearner(**common)
    if name == "ewc":
        return EWCLearner(**common)
    if name == "replay":
        return ReplayLearner(
            **common,
            replay_capacity=spec.replay_buffer_size,
            replay_seed=spec.seeds["replay"],
        )
    if ablation_mode and name == "sign_ncm":
        return SignNCMPrototypeLearner(input_dim=input_dim, device=device)
    if ablation_mode and name == "median_ncm":
        return MedianNCMPrototypeLearner(input_dim=input_dim, device=device)
    raise ValueError(f"unknown continual learner: {name}")


def _threshold_tag(threshold: float) -> str:
    return format(threshold, ".12g").replace("-", "neg").replace(".", "p")


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.") or "embeddings"


def _select_learner_names(
    learner_names: Sequence[str] | None,
    *, ablation_mode: bool = False,
) -> tuple[str, ...]:
    if learner_names is None:
        return LEARNER_NAMES
    selected = tuple(str(name) for name in learner_names)
    if not selected:
        raise ValueError("learner selection must not be empty")
    if len(selected) != len(set(selected)):
        raise ValueError("learner selection must not contain duplicates")
    allowed = LEARNER_NAMES + (ABLATION_LEARNER_NAMES if ablation_mode else ())
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"unknown continual learners: {unknown}")
    return selected


def _parse_learner_selection(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    try:
        return _select_learner_names(names)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _learner_configuration(name: str, spec: StreamSpec) -> dict[str, Any]:
    if name == "ncm":
        return {
            "embedding_variant": "l2_normalized",
            "prototype_update": "online_running_mean",
            "backpropagation": False,
        }
    if name == "ranpac":
        return {
            "projection_distribution": "gaussian_standard_normal",
            "projection_dim": spec.hypervector_dim,
            "projection_seed": spec.seeds["projection"],
            "activation": "relu",
            "prototype_update": "online_running_mean",
            "decorrelation": False,
            "ridge": False,
            "backpropagation": False,
        }
    if name == "sign_ncm":
        return {"embedding_variant": "native", "prototype_update": "elementwise_sign_majority", "backpropagation": False}
    if name == "median_ncm":
        return {"embedding_variant": "native", "prototype_update": "exact_running_median", "backpropagation": False}

    return {}


def _run_one(
    *,
    learner_name: str,
    threshold: float | None,
    bootstrap: Sequence[StreamSample],
    stream: Sequence[StreamSample],
    spec: StreamSpec,
    input_dim: int,
    clock: Callable[[], float],
    on_final_state: Callable[[Any, dict[str, Any]], None] | None = None,
    on_final_predictions: Callable[[list[dict[str, Any]]], None] | None = None,
    shared_label_ids: frozenset[str] | None = None,
    on_observation: Callable[[Any, dict[str, Any]], None] | None = None,
    ablation_mode: bool = False,
    hypervector_dim: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if shared_label_ids is not None:
        stream_ids = [sample.image_id for sample in stream]
        if (len(stream_ids) != len(set(stream_ids))
                or not shared_label_ids.issubset(stream_ids)
                or len(shared_label_ids) != spec.max_total_requests
                or spec.bootstrap_counts_toward_budget
                or spec.count_low_confidence or spec.count_novel_registration
                or threshold is not None):
            raise ValueError("shared-label execution requires exact stream IDs/budget, no acquisition/oracle, and uncharged bootstrap")
    elif threshold is None:
        raise ValueError("historical acquisition requires a confidence threshold")
    allowed = LEARNER_NAMES + (ABLATION_LEARNER_NAMES if ablation_mode else ())
    if learner_name not in allowed:
        raise ValueError(f"unknown continual learner: {learner_name}")
    learner = _build_learner(learner_name, spec=spec, input_dim=input_dim,
                             ablation_mode=ablation_mode, hypervector_dim=hypervector_dim)
    bootstrap_charge = (
        len(bootstrap) if spec.bootstrap_counts_toward_budget else 0
    )
    if bootstrap_charge > spec.max_total_requests:
        raise RuntimeError("bootstrap labels exceed the preregistered total label budget")
    charged = bootstrap_charge
    bootstrap_update_seconds: list[float] = []
    for sample in bootstrap:
        started = clock()
        learner.update(sample.embedding, sample.label)
        bootstrap_update_seconds.append(round(clock() - started, 9))

    history: list[dict[str, Any]] = []
    confirmed_novel_classes: set[str] = set()
    novel_purchase_counts = {label: 0 for label in spec.novel_classes}

    def observe(processed: int, event: dict[str, Any] | None) -> None:
        if on_observation is not None:
            on_observation(learner, {
                "processed_stream_events": processed, "labels_charged": charged,
                "bootstrap_labels_charged": bootstrap_charge,
                "novel_labels_consumed": dict(novel_purchase_counts),
                "confirmed_novel_classes": sorted(confirmed_novel_classes),
                "stream_complete": processed == len(stream), "last_event": event,
            })

    observe(0, None)
    for step, sample in enumerate(stream):
        prediction = learner.predict(sample.embedding)
        correct = prediction.label == sample.label
        if shared_label_ids is not None:
            # Experiment A: neither confidence nor hidden class identity selects labels.
            confirmed_event = sample.image_id in shared_label_ids
            request_reasons = {"shared_schedule": confirmed_event, "low_confidence": False,
                               "novel_registration": False}
        else:
            low_confidence = prediction.confidence < threshold
            novel_registration = (
                spec.count_novel_registration
                and sample.label in spec.novel_classes
                and sample.label not in confirmed_novel_classes
            )
            confirmed_event = (
                spec.count_low_confidence and low_confidence
            ) or novel_registration
            request_reasons = {"shared_schedule": False,
                               "low_confidence": bool(spec.count_low_confidence and low_confidence),
                               "novel_registration": bool(novel_registration)}
        label_requested = confirmed_event and charged < spec.max_total_requests
        denied = confirmed_event and not label_requested
        update_seconds = 0.0
        class_registered = False
        if label_requested:
            started = clock()
            result = learner.update(sample.embedding, sample.label)
            update_seconds = round(clock() - started, 9)
            class_registered = bool(result.class_registered)
            charged += 1
            if sample.label in spec.novel_classes:
                confirmed_novel_classes.add(sample.label)
                novel_purchase_counts[sample.label] += 1
        row: dict[str, Any] = {
            "step": step,
            "image_id": sample.image_id,
            "true_label": sample.label,
            "prediction": prediction.label,
            "confidence": round(float(prediction.confidence), 9),
            "top_probability": round(float(prediction.top_probability), 9),
            "label_requested": label_requested,
            "label_request_denied_budget": denied,
            "class_registered": class_registered,
            "update_wall_time_seconds": update_seconds,
            "correct": correct,
        }
        history.append(row)
        row["rolling_per_class_accuracy"] = rolling_per_class_accuracy(
            history,
            window=spec.rolling_window,
        )
        # Post-event state: zero-based event i is processed-event count i+1.
        # Passive collectors must not mutate the learner/RNG. No extra clock calls.
        observe(step + 1, {"step": step, "image_id": sample.image_id, "true_label": sample.label,
                           "potential_request": bool(confirmed_event), "request_reasons": request_reasons,
                           "purchased": bool(label_requested), "denied": bool(denied),
                           "class_registered": class_registered})

    if on_final_state is not None:
        on_final_state(learner, {
            "processed_stream_events": len(stream),
            "labels_charged": charged,
            "bootstrap_labels_charged": bootstrap_charge,
            "confirmed_novel_classes": sorted(confirmed_novel_classes),
            "stream_complete": True,
        })

    final_evaluation: list[dict[str, Any]] = []
    for sample in [*bootstrap, *stream]:
        prediction = learner.predict(sample.embedding)
        final_evaluation.append(
            {
                "true_label": sample.label,
                "prediction": prediction.label,
                "top_probability": round(float(prediction.top_probability), 9),
                "correct": prediction.label == sample.label,
            }
        )

    summary = build_summary(
        history,
        final_evaluation=final_evaluation,
        initially_known_classes=spec.initially_known_classes,
        novel_classes=spec.novel_classes,
        rolling_window=spec.rolling_window,
        stabilization_window=spec.stabilization_window,
        stabilization_min_accuracy=spec.stabilization_min_accuracy,
        stabilization_consecutive_windows=spec.stabilization_consecutive_windows,
        ece_bins=spec.ece_bins,
        bootstrap_update_seconds=bootstrap_update_seconds,
        bootstrap_labels_charged=bootstrap_charge,
        label_budget_max=spec.max_total_requests,
    )
    summary["bootstrap"] = {
        "confirmed_examples": len(bootstrap),
        "confirmed_per_initial_class": spec.bootstrap_confirmed_per_class,
        "known_class_holdback_manifest": spec.held_back_image_ids_by_seed is not None,
        "counts_toward_budget": spec.bootstrap_counts_toward_budget,
    }
    if on_final_predictions is not None:
        on_final_predictions([
            {"image_id": sample.image_id, **row}
            for sample, row in zip([*bootstrap, *stream], final_evaluation)
        ])
    return history, summary


def _write_history(path: Path, history: Sequence[dict[str, Any]]) -> None:
    fields = [
        "step",
        "image_id",
        "true_label",
        "prediction",
        "confidence",
        "top_probability",
        "label_requested",
        "label_request_denied_budget",
        "class_registered",
        "update_wall_time_seconds",
        "correct",
        "rolling_per_class_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            output = dict(row)
            output["rolling_per_class_accuracy"] = json.dumps(
                output["rolling_per_class_accuracy"], sort_keys=True, separators=(",", ":")
            )
            writer.writerow(output)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_code_hashes() -> dict[str, str]:
    # Hash the actual working code, including uncommitted changes, rather than
    # misidentifying a dirty checkout by HEAD alone.
    paths = [
        "ssl_hdc_pilot/scripts/stream_runner.py",
        "ssl_hdc_pilot/scripts/learner_checkpoint.py",
        "ssl_hdc_pilot/scripts/checkpoint_storage.py",
        "ssl_hdc_pilot/scripts/continual_hdc.py",
        "ssl_hdc_pilot/scripts/continual_prototypes.py",
        "ssl_hdc_pilot/scripts/aggregation_prototypes.py",
        "ssl_hdc_pilot/scripts/continual_baselines.py",
        "ssl_hdc_pilot/scripts/continual_metrics.py",
        "ssl_hdc_pilot/scripts/build_stream_spec.py",
        "ssl_hdc_pilot/scripts/build_trusted_splits.py",
        "common/thesis_eval.py",
    ]
    return {path: _file_sha256(REPO_ROOT / path) for path in paths}


def run_study(
    *,
    embeddings_path: Path,
    split_csv: Path,
    stream_spec_path: Path,
    output_dir: Path,
    smoke_mode: bool,
    ewc_only: bool = False,
    learner_names: Sequence[str] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    save_final_state: bool = False,
) -> list[StudyArtifact]:
    if ewc_only and learner_names is not None:
        raise ValueError("--ewc-only cannot be combined with an explicit learner selection")
    selected_learners = (
        ("ewc",) if ewc_only else _select_learner_names(learner_names)
    )
    spec = load_stream_spec(stream_spec_path, smoke_mode=smoke_mode)
    embedding_map, input_dim = _load_embedding_map(Path(embeddings_path))
    split_rows = _load_split_rows(Path(split_csv), smoke_mode=smoke_mode)
    if not smoke_mode:
        _validate_official_section6a(
            spec=spec,
            split_csv=Path(split_csv),
            split_rows=split_rows,
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_tag = _safe_stem(Path(embeddings_path).stem)
    checkpoint_inputs = ({
        "embeddings_sha256": _file_sha256(Path(embeddings_path)),
        "split_sha256": _file_sha256(Path(split_csv)),
    } if save_final_state else {})
    code_hashes = _checkpoint_code_hashes() if save_final_state else {}
    artifacts: list[StudyArtifact] = []
    for stream_seed in spec.stream_order_seeds:
        bootstrap, stream = _prepare_stream(
            split_rows=split_rows,
            embedding_map=embedding_map,
            spec=spec,
            stream_seed=stream_seed,
        )
        for learner_name in selected_learners:
            for threshold in spec.confidence_thresholds:
                run_tag = (
                    f"{spec.tag}__{embedding_tag}__stream_{stream_seed}__"
                    f"{learner_name}__margin_{_threshold_tag(threshold)}"
                )
                history_path = output_dir / f"{run_tag}_history.csv"
                summary_path = output_dir / f"{run_tag}_summary.json"
                state_path = output_dir / f"{run_tag}_final_state.npz"
                if history_path.exists() or summary_path.exists() or state_path.exists():
                    raise FileExistsError(
                        f"refusing to overwrite continual run tag: {run_tag}"
                    )

                def preserve_final_state(learner: Any, progress: dict[str, Any]) -> None:
                    save_learner_checkpoint(state_path, learner, context={
                        "run_tag": run_tag,
                        "learner": learner_name,
                        "confidence_threshold": threshold,
                        "stream_seed": stream_seed,
                        "stream_spec_sha256": spec.sha256,
                        "stream_spec_path": str(stream_spec_path),
                        "smoke": spec.smoke,
                        "data_role": "synthetic_smoke" if spec.smoke else "trusted_train",
                        "input_hashes": checkpoint_inputs,
                        "code_hashes": code_hashes,
                        "bootstrap_image_ids": [sample.image_id for sample in bootstrap],
                        "stream_image_ids": [sample.image_id for sample in stream],
                        "progress": progress,
                    })

                history, summary = _run_one(
                    learner_name=learner_name,
                    threshold=threshold,
                    bootstrap=bootstrap,
                    stream=stream,
                    spec=spec,
                    input_dim=input_dim,
                    clock=clock,
                    on_final_state=preserve_final_state if save_final_state else None,
                )
                summary = {
                    "run_tag": run_tag,
                    "learner": learner_name,
                    "learner_configuration": _learner_configuration(
                        learner_name, spec
                    ),
                    "confidence_definition": (
                        "top1 minus top2 cosine-similarity score margin"
                    ),
                    "confidence_threshold": threshold,
                    "embedding_dim": input_dim,
                    "embedding_npz": str(Path(embeddings_path)),
                    "trusted_split_csv": str(Path(split_csv)),
                    "stream_spec": str(Path(stream_spec_path)),
                    "stream_spec_sha256": spec.sha256,
                    "smoke": spec.smoke,
                    "stream_seed": stream_seed,
                    "stream_steps": len(stream),
                    **summary,
                }
                _write_history(history_path, history)
                summary_path.write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                artifacts.append(StudyArtifact(run_tag, history_path, summary_path,
                                               state_path if save_final_state else None))
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--trusted-split", type=Path, required=True)
    parser.add_argument("--stream-spec", type=Path, default=DEFAULT_STREAM_SPEC)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ewc-only",
        action="store_true",
        help="run only EWC correction cells; the default runs the complete grid",
    )
    parser.add_argument(
        "--learners",
        type=_parse_learner_selection,
        help=(
            "comma-separated learner subset; choices: "
            + ",".join(LEARNER_NAMES)
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--save-final-state", action="store_true",
        help="save a non-overwriting learner-resumable NPZ after the stream; does not enable stream resume or held-out evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_study(
        embeddings_path=args.embeddings,
        split_csv=args.trusted_split,
        stream_spec_path=args.stream_spec,
        output_dir=args.output_dir,
        smoke_mode=args.smoke,
        ewc_only=args.ewc_only,
        learner_names=args.learners,
        save_final_state=args.save_final_state,
    )
    print(
        json.dumps(
            [
                {
                    "run_tag": artifact.run_tag,
                    "history_csv": str(artifact.history_csv),
                    "summary_json": str(artifact.summary_json),
                    **({"final_state_npz": str(artifact.final_state_npz)}
                       if artifact.final_state_npz is not None else {}),
                }
                for artifact in artifacts
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
