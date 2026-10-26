"""Deterministic, duplicate-safe preparation of replication development splits.

This module deliberately has no reader for a replication test partition.  It
creates train/validation CSVs plus an integrity manifest whose test rows are
sealed: consumers must not use this module to access them.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MIN_CLASS_SUPPORT = 30
SPLIT_RATIOS: Mapping[str, float] = {"train": 0.60, "val": 0.20, "test": 0.20}
SPLIT_NAMES = tuple(SPLIT_RATIOS)
CSV_FIELDS = (
    "image",
    "label",
    "split_partition",
    "split_policy",
    "split_seed",
    "supervisor_approved",
    "provenance_version",
    "source_csv",
    "image_path",
)
REPLICATION_POLICY = "replication-60-20-20-content-grouped-v1"
REPLICATION_PROVENANCE_VERSION = "ssl-hdc-replication-split-v1"


@dataclass(frozen=True, order=True)
class ImageRecord:
    """One acquired image, with a content SHA-256 calculated before splitting."""

    image_id: str
    label: str
    image_path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.image_id or not self.label or not self.image_path:
            raise ValueError("image_id, label, and image_path are required")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("sha256 must be a lowercase 64-character digest")


@dataclass(frozen=True)
class PreparedSplits:
    """In-memory partitions; the test member is never written as a CSV."""

    partitions: Mapping[str, tuple[ImageRecord, ...]]
    report: Mapping[str, object]


def sha256_file(path: Path) -> str:
    """Return a content SHA-256 without decoding the image."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sha256_manifest(records: Iterable[ImageRecord], path: Path) -> None:
    """Write the acquisition-time per-file hash manifest once, in stable order.

    The manifest is metadata only; it neither copies nor opens image content.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite SHA-256 manifest: {path}")
    normalized = _validate_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in normalized:
            handle.write(
                json.dumps(
                    {
                        "image": record.image_id,
                        "image_path": record.image_path,
                        "label": record.label,
                        "sha256": record.sha256,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _validate_records(records: Iterable[ImageRecord]) -> tuple[ImageRecord, ...]:
    result = tuple(sorted(records))
    if not result:
        raise ValueError("no image records supplied")
    image_ids = [record.image_id for record in result]
    paths = [record.image_path for record in result]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("duplicate image_id")
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate image_path")
    return result


def group_exact_duplicates(records: Iterable[ImageRecord]) -> dict[str, tuple[ImageRecord, ...]]:
    """Group images by exact content and reject cross-label duplicate content."""
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in _validate_records(records):
        grouped[record.sha256].append(record)
    result: dict[str, tuple[ImageRecord, ...]] = {}
    for digest, group in sorted(grouped.items()):
        labels = {record.label for record in group}
        if len(labels) != 1:
            raise ValueError(f"exact duplicate content has conflicting labels: {digest}")
        result[digest] = tuple(sorted(group))
    return result


def _rank(seed: int, label: str, digest: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{label}\0{digest}".encode("utf-8")).digest()


def _assign_label_groups(
    groups: Sequence[tuple[str, tuple[ImageRecord, ...]]], seed: int, label: str
) -> dict[str, list[ImageRecord]]:
    """Assign whole content groups with deterministic nearest-target balancing."""
    total = sum(len(group) for _, group in groups)
    targets = {name: total * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: 0 for name in SPLIT_NAMES}
    assigned = {name: [] for name in SPLIT_NAMES}
    for digest, group in sorted(groups, key=lambda item: (_rank(seed, label, item[0]), item[0])):
        size = len(group)
        # Minimum resulting squared error, then fixed partition order, makes ties stable.
        partition = min(
            SPLIT_NAMES,
            key=lambda name: (
                sum(
                    ((counts[other] + (size if other == name else 0)) - targets[other]) ** 2
                    for other in SPLIT_NAMES
                ),
                SPLIT_NAMES.index(name),
            ),
        )
        counts[partition] += size
        assigned[partition].extend(group)
    return assigned


def build_replication_splits(
    records: Iterable[ImageRecord], *, seed: int, min_class_support: int = MIN_CLASS_SUPPORT
) -> PreparedSplits:
    """Apply the locked support rule and split without allowing duplicate leakage."""
    if min_class_support < 1:
        raise ValueError("min_class_support must be positive")
    normalized = _validate_records(records)
    all_counts = Counter(record.label for record in normalized)
    retained_labels = {label for label, count in all_counts.items() if count >= min_class_support}
    if len(retained_labels) < 2:
        raise ValueError("at least two classes must satisfy the support rule")
    retained = tuple(record for record in normalized if record.label in retained_labels)
    duplicate_groups = group_exact_duplicates(retained)
    by_label: dict[str, list[tuple[str, tuple[ImageRecord, ...]]]] = defaultdict(list)
    for digest, group in duplicate_groups.items():
        by_label[group[0].label].append((digest, group))

    partitions: dict[str, list[ImageRecord]] = {name: [] for name in SPLIT_NAMES}
    for label, groups in sorted(by_label.items()):
        allocated = _assign_label_groups(groups, seed, label)
        for name in SPLIT_NAMES:
            partitions[name].extend(allocated[name])
    frozen_partitions = {name: tuple(sorted(rows)) for name, rows in partitions.items()}
    _assert_disjoint_content(frozen_partitions)
    duplicate_group_sizes = sorted(len(group) for group in duplicate_groups.values() if len(group) > 1)
    report: dict[str, object] = {
        "policy": REPLICATION_POLICY,
        "seed": seed,
        "min_class_support": min_class_support,
        "input_count": len(normalized),
        "retained_count": len(retained),
        "retained_class_count": len(retained_labels),
        "excluded_classes": dict(sorted((label, count) for label, count in all_counts.items() if label not in retained_labels)),
        "split_counts": {name: len(rows) for name, rows in frozen_partitions.items()},
        "class_distribution": {
            name: dict(sorted(Counter(record.label for record in rows).items()))
            for name, rows in frozen_partitions.items()
        },
        "exact_duplicate_groups": len(duplicate_groups),
        "duplicate_group_count": len(duplicate_group_sizes),
        "duplicate_image_count": sum(duplicate_group_sizes),
    }
    return PreparedSplits(partitions=frozen_partitions, report=report)


def _assert_disjoint_content(partitions: Mapping[str, Sequence[ImageRecord]]) -> None:
    owners: dict[str, str] = {}
    for partition, rows in partitions.items():
        for record in rows:
            prior = owners.setdefault(record.sha256, partition)
            if prior != partition:
                raise AssertionError(f"content digest crosses partitions: {record.sha256}")


def assert_not_sealed_test_path(path: Path) -> None:
    """Reject test CSV paths before they can be opened by replication code."""
    candidate = Path(path)
    if candidate.name.lower() == "test.csv" or "sealed_test" in candidate.name.lower():
        raise PermissionError("replication test split is sealed and must not be opened")


def _csv_row(record: ImageRecord, partition: str, seed: int, source_manifest: str) -> dict[str, object]:
    return {
        "image": record.image_id,
        "label": record.label,
        "split_partition": partition,
        "split_policy": REPLICATION_POLICY,
        "split_seed": seed,
        "supervisor_approved": "pending",
        "provenance_version": REPLICATION_PROVENANCE_VERSION,
        "source_csv": source_manifest,
        "image_path": record.image_path,
    }


def write_prepared_splits(prepared: PreparedSplits, output_dir: Path, *, source_manifest: str) -> None:
    """Write only development splits and a sealed manifest; refuse overwrite."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite replication split directory: {output_dir}")
    if set(prepared.partitions) != set(SPLIT_NAMES):
        raise ValueError("prepared partitions must be train, val, and test")
    output_dir.mkdir(parents=True)
    seed = int(prepared.report["seed"])
    for partition in ("train", "val"):
        with (output_dir / f"{partition}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(_csv_row(row, partition, seed, source_manifest) for row in prepared.partitions[partition])
    sealed = {
        "status": "SEALED_DO_NOT_OPEN",
        "test_count": len(prepared.partitions["test"]),
        "test_content_sha256": hashlib.sha256(
            "\n".join(row.sha256 for row in prepared.partitions["test"]).encode("ascii")
        ).hexdigest(),
        "test_image_ids_sha256": hashlib.sha256(
            "\n".join(row.image_id for row in prepared.partitions["test"]).encode("utf-8")
        ).hexdigest(),
    }
    (output_dir / "sealed_test_receipt.json").write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "distribution_report.json").write_text(
        json.dumps(prepared.report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
