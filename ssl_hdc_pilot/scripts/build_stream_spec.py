"""Generate the locked Section 6a continual stream specification."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ssl_hdc_pilot.scripts.build_trusted_splits import (
    CANONICAL_SOURCE_CSV,
    OFFICIAL_SEED,
    PROVENANCE_VERSION,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = REPO_ROOT / "ssl_hdc_pilot" / "trusted_splits"
DEFAULT_OUTPUT = REPO_ROOT / "ssl_hdc_pilot" / "stream_spec.json"
DEFAULT_SUMMARY_LOG = REPO_ROOT / "ssl_hdc_pilot" / "stream_spec_summary.log"
APPROVED_POLICY = "option-a-19class"
EXPECTED_CLASS_COUNT = 19
NOVEL_CLASS_COUNT = 4
STREAM_SEEDS = (0, 1, 2)
CONFIDENCE_THRESHOLDS = (0.05, 0.10, 0.20)
KNOWN_HOLDBACK_FRACTION = 0.25
ENTRY_FRACTIONS = (0.20, 0.40, 0.60, 0.80)

# These are runner implementation settings, not Section 6a protocol choices.
# Section 6a values above are deliberately not exposed as command-line flags.
RUNNER_SETTINGS: Mapping[str, Any] = {
    "execution": {"device": "cuda"},
    "hdc": {"hypervector_dim": 10_000},
    "baseline": {
        "head_type": "linear",
        "hidden_dim": None,
        "learning_rate": 0.1,
        "weight_decay": 0.0,
        "update_steps": 1,
        "ewc_lambda": 10.0,
        "ewc_decay": 0.9,
        "replay_buffer_size": 512,
        "replay_batch_size": 32,
    },
    "metrics": {
        "stabilization_window": 20,
        "stabilization_min_accuracy": 0.8,
        "stabilization_consecutive_windows": 3,
        "ece_bins": 15,
    },
}


@dataclass(frozen=True)
class Section6aPlan:
    trusted_counts: Mapping[str, int]
    known_classes: tuple[str, ...]
    novel_classes: tuple[str, ...]
    entry_order: tuple[str, ...]
    manifests: Mapping[str, tuple[str, ...]]
    stream_length: int
    introduction_steps: tuple[int, ...]


def _load_approved_partition(
    split_dir: Path, partition: str
) -> list[tuple[str, str]]:
    path = split_dir / f"{partition}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"approved Option-A {partition} split is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "image",
            "label",
            "split_partition",
            "split_policy",
            "split_seed",
            "supervisor_approved",
            "provenance_version",
            "source_csv",
        }
        if not required <= set(reader.fieldnames or []):
            raise RuntimeError("split lacks supervisor-approved provenance fields")
        rows = list(reader)
    if not rows:
        raise RuntimeError("approved Option-A training split is empty")
    expected = {
        "split_partition": partition,
        "split_policy": APPROVED_POLICY,
        "split_seed": str(OFFICIAL_SEED),
        "supervisor_approved": "true",
        "provenance_version": PROVENANCE_VERSION,
        "source_csv": CANONICAL_SOURCE_CSV,
    }
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if any((row.get(field) or "").strip() != value for field, value in expected.items()):
            raise RuntimeError("split lacks supervisor-approved provenance values")
        image_id = (row.get("image") or "").strip()
        label = (row.get("label") or "").strip()
        if not image_id or not label or image_id in seen:
            raise RuntimeError(f"invalid or duplicate approved split row: {image_id!r}")
        seen.add(image_id)
        records.append((image_id, label))
    if len({label for _, label in records}) != EXPECTED_CLASS_COUNT:
        raise RuntimeError(f"approved Option-A {partition} split must contain 19 classes")
    return records


def load_approved_train(split_dir: Path) -> list[tuple[str, str]]:
    return _load_approved_partition(Path(split_dir), "train")


def _load_retained_counts(
    split_dir: Path, train_rows: Sequence[tuple[str, str]]
) -> dict[str, int]:
    """Count retained classes from the three approved split CSVs themselves."""
    partition_rows = {
        "train": list(train_rows),
        "val": _load_approved_partition(Path(split_dir), "val"),
        "test": _load_approved_partition(Path(split_dir), "test"),
    }
    expected_labels = {label for _, label in train_rows}
    retained_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for partition, rows in partition_rows.items():
        labels = {label for _, label in rows}
        if labels != expected_labels:
            raise RuntimeError(
                f"approved Option-A {partition} labels do not match the training split"
            )
        for image_id, label in rows:
            if image_id in seen_ids:
                raise RuntimeError("approved Option-A image ids overlap across partitions")
            seen_ids.add(image_id)
            retained_counts[label] += 1
    return dict(retained_counts)


def _is_disease_class(label: str) -> bool:
    parts = label.split("_")
    return len(parts) >= 3 and parts[1] != "00"


def _holdback_count(class_count: int) -> int:
    count = math.floor(class_count * KNOWN_HOLDBACK_FRACTION)
    if count < 1 or count >= class_count:
        raise RuntimeError(
            "each initially-known class needs at least four training examples for 25% holdback"
        )
    return count


def _holdback_rank(seed: int, label: str, image_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{label}\0{image_id}".encode("utf-8")).digest()


def _build_holdback_manifests(
    train_rows: Sequence[tuple[str, str]], known_classes: Sequence[str]
) -> dict[str, list[str]]:
    ids_by_class: dict[str, list[str]] = {label: [] for label in known_classes}
    for image_id, label in train_rows:
        if label in ids_by_class:
            ids_by_class[label].append(image_id)
    manifests: dict[str, list[str]] = {}
    for seed in STREAM_SEEDS:
        selected: list[str] = []
        for label in known_classes:
            ranked = sorted(
                ids_by_class[label],
                key=lambda image_id: (_holdback_rank(seed, label, image_id), image_id),
            )
            selected.extend(ranked[: _holdback_count(len(ranked))])
        manifests[str(seed)] = sorted(selected)
    return manifests


def derive_section6a_plan(
    split_dir: Path, train_rows: Sequence[tuple[str, str]]
) -> Section6aPlan:
    """Mechanically derive every locked Section 6a class and schedule choice."""
    trusted_counts = _load_retained_counts(Path(split_dir), train_rows)
    disease_classes = [label for label in trusted_counts if _is_disease_class(label)]
    if len(disease_classes) < NOVEL_CLASS_COUNT:
        raise RuntimeError("approved split has fewer than four retained disease classes")
    novel_classes = tuple(
        sorted(disease_classes, key=lambda label: (trusted_counts[label], label))[
            :NOVEL_CLASS_COUNT
        ]
    )
    known_classes = tuple(sorted(set(trusted_counts) - set(novel_classes)))
    entry_order = tuple(
        sorted(novel_classes, key=lambda label: (-trusted_counts[label], label))
    )
    train_counts = Counter(label for _, label in train_rows)
    raw_manifests = _build_holdback_manifests(train_rows, known_classes)
    manifests = {seed: tuple(image_ids) for seed, image_ids in raw_manifests.items()}
    stream_length = len(manifests["0"]) + sum(
        train_counts[label] for label in novel_classes
    )
    introduction_steps = tuple(
        math.floor(stream_length * fraction) for fraction in ENTRY_FRACTIONS
    )
    return Section6aPlan(
        trusted_counts=trusted_counts,
        known_classes=known_classes,
        novel_classes=novel_classes,
        entry_order=entry_order,
        manifests=manifests,
        stream_length=stream_length,
        introduction_steps=introduction_steps,
    )


def _summary_text(payload: Mapping[str, Any], trusted_counts: Mapping[str, int]) -> str:
    protocol = payload["protocol"]
    stream_seeds = ", ".join(str(seed) for seed in STREAM_SEEDS)
    thresholds = ", ".join(f"{value:.2f}" for value in CONFIDENCE_THRESHOLDS)
    lines = [
        "=== SSL-HDC SECTION 6a STREAM SPEC SUMMARY ===",
        f"Tag: {payload['tag']}",
        "Smoke: false",
        "Novel classes (entry order):",
    ]
    for item, fraction in zip(protocol["class_introductions"], ENTRY_FRACTIONS, strict=True):
        label = item["class_label"]
        lines.append(
            f"  {fraction:.0%}: {label} (trusted count={trusted_counts[label]}, step={item['at_step']})"
        )
    lines.extend(
        [
            f"Initially known classes: {len(protocol['initially_known_classes'])}",
            f"Known-class holdback: {KNOWN_HOLDBACK_FRACTION:.0%} per class "
            "(floor for indivisible counts)",
            f"Stream length: {protocol['stream_length']}",
            f"Stream seeds: {stream_seeds}",
            f"Confidence thresholds: {thresholds}",
            "Label accounting: every low-confidence request and novel registration counts",
            "=== END STREAM SPEC SUMMARY ===",
        ]
    )
    return "\n".join(lines) + "\n"


def build_stream_spec(
    *, split_dir: Path, output_path: Path, summary_log_path: Path
) -> dict[str, Any]:
    train_rows = load_approved_train(Path(split_dir))
    plan = derive_section6a_plan(Path(split_dir), train_rows)
    introductions = [
        {"class_label": label, "at_step": step}
        for label, step in zip(plan.entry_order, plan.introduction_steps, strict=True)
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "tag": "option_a_section6a_v1",
        "smoke": False,
        "execution": dict(RUNNER_SETTINGS["execution"]),
        "seeds": {"stream_order": list(STREAM_SEEDS), "projection": 0, "learner": 0, "replay": 0},
        "protocol": {
            "initially_known_classes": list(plan.known_classes),
            "class_introductions": introductions,
            "bootstrap": {
                "known_class_holdback_fraction": KNOWN_HOLDBACK_FRACTION,
                "held_back_image_ids_by_seed": {
                    seed: list(image_ids) for seed, image_ids in plan.manifests.items()
                },
                "counts_toward_budget": False,
            },
            "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
            "label_budget": {
                "max_total_requests": plan.stream_length,
                "count_low_confidence": True,
                "count_novel_registration": True,
            },
            "rolling_window": 50,
            "stream_length": plan.stream_length,
        },
        "hdc": dict(RUNNER_SETTINGS["hdc"]),
        "baseline": dict(RUNNER_SETTINGS["baseline"]),
        "metrics": dict(RUNNER_SETTINGS["metrics"]),
    }
    output_path = Path(output_path)
    summary_log_path = Path(summary_log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = _summary_text(payload, plan.trusted_counts)
    summary_log_path.write_text(summary, encoding="utf-8")
    print(summary, end="")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_stream_spec(
        split_dir=args.split_dir,
        output_path=DEFAULT_OUTPUT,
        summary_log_path=DEFAULT_SUMMARY_LOG,
    )


if __name__ == "__main__":
    main()
