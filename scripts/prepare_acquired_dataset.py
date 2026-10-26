from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssl_hdc_pilot.replication.data import (
    CSV_FIELDS,
    ImageRecord,
    PreparedSplits,
    REPLICATION_POLICY,
    build_replication_splits,
    group_exact_duplicates,
    write_prepared_splits,
)
from ssl_hdc_pilot.scripts.build_trusted_splits import OFFICIAL_SEED


SOURCES: dict[str, dict[str, Any]] = {
    "PlantDoc": {
        "source_url": "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip",
        "source_revision": "5467f6012d78d1c446145d5f582da6096f852ae8",
        "bytes": 984_413_268,
        "sha256": "345548bb3659519f425608b428d8a2ca338032b7b02c3cb3c3bb2954fc8986d9",
        "hash_verification": "Claude verified; upstream publishes no archive hash",
        "expected_raw": (2_578, 28),
        "expected_retained": (2_576, 27),
    },
    "PlantWild": {
        "source_url": "https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/527a72eb8f00c95e41698bb09d982c7d4625dce3/plantwild.zip",
        "source_revision": "527a72eb8f00c95e41698bb09d982c7d4625dce3",
        "bytes": 2_708_493_416,
        "sha256": "6fe6ca45914a453e9b4907f5c2b115fc8cb8481609bd9f8f93008dabf7aca9a3",
        "hash_verification": "Claude verified against the Hugging Face LFS oid",
        "expected_raw": (18_542, 89),
        "expected_retained": (18_542, 89),
    },
    "PlantVillage": {
        "source_url": "https://data.mendeley.com/public-api/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded",
        "source_revision": "Mendeley tywbtsjrjv v1",
        "source_file_id": "d5652a28-c1d8-4b76-97f3-72fb80f94efc",
        "source_filename": "Plant_leaf_diseases_dataset_without_augmentation.zip",
        "bytes": 868_032_562,
        "sha256": "ac3432453984d02a86197987e775a5429d0d59e7cc7c35bcf5a8f50349b90ff0",
        "hash_verification": "Claude verified against Mendeley's published hash",
        "expected_raw": (54_303, 38),
        "expected_retained": (54_303, 38),
    },
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
BUFFER_SIZE = 1024 * 1024
WINDOWS_INVALID_CHARS = set('<>:"\\|?*')
WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def utc_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def receipt_payload(dataset: str, archive: Path) -> dict[str, Any]:
    source = SOURCES[dataset]
    actual_bytes = archive.stat().st_size
    if actual_bytes != source["bytes"]:
        raise RuntimeError(
            f"{dataset} archive byte mismatch: expected={source['bytes']}, actual={actual_bytes}"
        )
    payload = {
        "dataset": dataset,
        "archive": archive.name,
        "bytes": actual_bytes,
        "sha256": source["sha256"],
        "source_url": source["source_url"],
        "source_revision": source["source_revision"],
        "downloaded_utc": utc_mtime(archive),
        "hash_verification": source["hash_verification"],
    }
    for key in ("source_file_id", "source_filename"):
        if key in source:
            payload[key] = source[key]
    return payload


def write_receipt(dataset: str) -> Path:
    dataset_root = ROOT / "ssl_hdc_pilot/replication/data" / dataset
    archive = dataset_root / "source_download.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"missing archive: {archive}")
    receipt = dataset_root / "acquisition_receipt.json"
    temporary = ROOT / ".codex-tmp" / f"{dataset}_acquisition_receipt.json.tmp"
    temporary.write_text(
        json.dumps(receipt_payload(dataset, archive), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, receipt)
    return receipt


def validate_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RuntimeError(f"unsafe ZIP member path: {name}")
    return path


def renamed_component(component: str, token: str) -> str:
    path = PurePosixPath(component)
    return f"{path.stem}__{token}{path.suffix}"


def windows_safe_component(component: str) -> tuple[str, bool]:
    """Encode characters Windows cannot represent without changing archive identity."""
    trailing_start = len(component.rstrip(" ."))
    pieces = []
    for index, char in enumerate(component):
        if (
            ord(char) < 32
            or char in WINDOWS_INVALID_CHARS
            or index >= trailing_start
        ):
            pieces.append(f"__{ord(char):02x}")
        else:
            pieces.append(char)
    safe = "".join(pieces)
    if safe.split(".", 1)[0].casefold() in WINDOWS_RESERVED_NAMES:
        token = hashlib.sha256(component.encode("utf-8")).hexdigest()[:8]
        safe = renamed_component(safe, f"reserved_{token}")
    return safe, safe != component


def copy_and_hash(source: BinaryIO, destination: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        while True:
            block = source.read(BUFFER_SIZE)
            if not block:
                break
            digest.update(block)
            output.write(block)
            byte_count += len(block)
    return digest.hexdigest(), byte_count


def hash_member(handle: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with handle.open(info) as source:
        for block in iter(lambda: source.read(BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def class_label(dataset: str, archive_path: PurePosixPath) -> str | None:
    if archive_path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    parts = archive_path.parts
    if dataset == "PlantDoc":
        for marker in ("train", "test"):
            if marker in parts:
                index = parts.index(marker)
                if len(parts) == index + 3:
                    return parts[index + 1]
        return None
    if dataset == "PlantWild":
        if len(parts) == 4 and parts[0:2] == ("plantwild", "images"):
            return parts[2]
        return None
    marker = "Plant_leave_diseases_dataset_without_augmentation"
    if marker not in parts:
        return None
    index = parts.index(marker)
    if len(parts) != index + 3:
        return None
    label = parts[index + 1]
    if label == "Background_without_leaves":
        return None
    # Two stray non-PlantVillage files ("image (1).*") sit in the Mendeley archive; excluding
    # them reproduces the published 54,303-image inventory (Claude ruling, 2026-09-17).
    if parts[index + 2] in {"image (1).png", "image (1).jpeg"}:
        return None
    return label


def source_split(dataset: str, archive_path: PurePosixPath) -> str:
    if dataset != "PlantDoc":
        return "not_supplied"
    for marker in ("train", "test"):
        if marker in archive_path.parts:
            return marker
    raise RuntimeError(f"PlantDoc image lacks supplied source split: {archive_path}")


def archive_existing(path: Path, dataset: str, role: str) -> None:
    if not path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / ".codex-tmp/obsolete_extractions" / f"{dataset}_{role}_{stamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    print(f"PRESERVED_OLD role={role} source={path} destination={destination}", flush=True)


def extract_collision_safe(
    dataset: str, archive: Path, staging: Path, final_root: Path
) -> tuple[list[ImageRecord], list[dict[str, Any]], dict[str, str], dict[str, int]]:
    directory_names: dict[tuple[str, str], tuple[str, str]] = {}
    file_targets: dict[str, dict[str, Any]] = {}
    collision_rows: list[dict[str, Any]] = []
    collision_recorded: set[tuple[str, str]] = set()
    image_rows: list[ImageRecord] = []
    source_splits: dict[str, str] = {}
    byte_sizes: dict[str, int] = {}

    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        for position, info in enumerate(infos, start=1):
            archive_path = validate_member_path(info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"symbolic-link ZIP member is not allowed: {info.filename}")
            if info.is_dir():
                continue

            safe_parent: list[str] = []
            directory_collision = False
            windows_path_rewritten = False
            original_prefix: list[str] = []
            for component in archive_path.parts[:-1]:
                original_prefix.append(component)
                base_component, component_rewritten = windows_safe_component(component)
                windows_path_rewritten = windows_path_rewritten or component_rewritten
                parent_key = "/".join(part.casefold() for part in safe_parent)
                key = (parent_key, base_component.casefold())
                prior = directory_names.get(key)
                if prior is None:
                    safe_component = base_component
                    directory_names[key] = (component, safe_component)
                elif prior[0] == component:
                    safe_component = prior[1]
                else:
                    token = hashlib.sha256("/".join(original_prefix).encode("utf-8")).hexdigest()[:8]
                    safe_component = renamed_component(base_component, token)
                    directory_collision = True
                safe_parent.append(safe_component)

            original_filename = archive_path.name
            safe_filename, filename_rewritten = windows_safe_component(original_filename)
            windows_path_rewritten = windows_path_rewritten or filename_rewritten
            proposed = PurePosixPath(*safe_parent, safe_filename)
            collision_base_filename = safe_filename
            target_key = proposed.as_posix().casefold()
            first = file_targets.get(target_key)
            member_digest: str | None = None
            collision_kind: str | None = None
            collision_key: str | None = None
            if first is not None:
                collision_key = f"file:{target_key}"
                member_digest = hash_member(handle, info)
                safe_filename = renamed_component(collision_base_filename, member_digest[:8])
                proposed = PurePosixPath(*safe_parent, safe_filename)
                target_key = proposed.as_posix().casefold()
                suffix_length = 8
                while target_key in file_targets:
                    suffix_length += 4
                    safe_filename = renamed_component(
                        collision_base_filename, member_digest[:suffix_length]
                    )
                    proposed = PurePosixPath(*safe_parent, safe_filename)
                    target_key = proposed.as_posix().casefold()
                collision_kind = "case_insensitive_file_path"
                if windows_path_rewritten:
                    collision_kind += "+windows_invalid_path"
                first_key = (first["archive_path"], first["sha256"])
                if first_key not in collision_recorded:
                    collision_rows.append(
                        {
                            **first,
                            "collision_kind": collision_kind,
                            "collision_key": collision_key,
                        }
                    )
                    collision_recorded.add(first_key)
            elif directory_collision:
                collision_kind = "case_insensitive_directory_path"
                collision_key = f"directory:{'/'.join(archive_path.parts[:-1]).casefold()}"
            elif windows_path_rewritten:
                collision_kind = "windows_invalid_path"
                collision_key = f"windows:{archive_path.as_posix()}"

            destination = staging / Path(*proposed.parts)
            with handle.open(info) as source:
                digest, byte_count = copy_and_hash(source, destination)
            if member_digest is not None and digest != member_digest:
                raise RuntimeError(f"member hash changed while extracting: {info.filename}")
            if byte_count != info.file_size:
                raise RuntimeError(
                    f"member byte mismatch: {info.filename}; expected={info.file_size}, actual={byte_count}"
                )
            metadata = {
                "archive_path": archive_path.as_posix(),
                "extracted_path": proposed.as_posix(),
                "sha256": digest,
                "bytes": byte_count,
            }
            file_targets[target_key] = metadata
            if collision_kind is not None:
                collision_rows.append(
                    {
                        **metadata,
                        "collision_kind": collision_kind,
                        "collision_key": collision_key,
                    }
                )
                collision_recorded.add((metadata["archive_path"], digest))

            label = class_label(dataset, archive_path)
            if label is not None:
                final_path = final_root / Path(*proposed.parts)
                image_id = f"{archive_path.as_posix()}#{digest}"
                image_rows.append(
                    ImageRecord(
                        image_id=image_id,
                        label=label,
                        image_path=final_path.relative_to(ROOT).as_posix(),
                        sha256=digest,
                    )
                )
                source_splits[image_id] = source_split(dataset, archive_path)
                byte_sizes[image_id] = byte_count
            if position % 5_000 == 0 or position == len(infos):
                print(
                    f"EXTRACT_PROGRESS dataset={dataset} members={position}/{len(infos)} "
                    f"images={len(image_rows)} collisions={len(collision_rows)}",
                    flush=True,
                )
    return image_rows, collision_rows, source_splits, byte_sizes


def split_label_conflicts(
    records: list[ImageRecord],
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Exclude every record whose exact byte content maps to more than one label.

    Applied before any duplicate grouping. ALL records sharing a conflicting
    sha256 are removed from the run entirely (not just the extra copies).
    """
    by_digest: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_digest[record.sha256].append(record)
    conflicting_digests = {
        digest
        for digest, group in by_digest.items()
        if len({row.label for row in group}) > 1
    }
    clean = [row for row in records if row.sha256 not in conflicting_digests]
    excluded = [row for row in records if row.sha256 in conflicting_digests]
    return clean, excluded


def write_label_conflicts(
    dataset_root: Path, excluded: list[ImageRecord], byte_sizes: dict[str, int]
) -> Path:
    path = dataset_root / "label_conflicts.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite label conflict record: {path}")
    by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in excluded:
        suffix = f"#{record.sha256}"
        archive_path = (
            record.image_id[: -len(suffix)]
            if record.image_id.endswith(suffix)
            else record.image_id
        )
        by_digest[record.sha256].append(
            {
                "image_id": record.image_id,
                "archive_path": archive_path,
                "label": record.label,
                "bytes": byte_sizes[record.image_id],
            }
        )
    payload = {
        "total_excluded_images": len(excluded),
        "total_conflicting_groups": len(by_digest),
        "conflicts": {
            digest: sorted(rows, key=lambda row: row["image_id"])
            for digest, rows in sorted(by_digest.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def split_rank(seed: int, label: str, digest: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{label}\0{digest}".encode("utf-8")).digest()


def build_plantdoc_splits(
    records: list[ImageRecord], source_splits: dict[str, str]
) -> PreparedSplits:
    """Keep supplied-test images and their exact duplicate groups sealed."""
    all_counts = Counter(record.label for record in records)
    retained_labels = {label for label, count in all_counts.items() if count >= 30}
    retained = tuple(sorted(row for row in records if row.label in retained_labels))
    duplicate_groups = group_exact_duplicates(retained)
    by_label: dict[str, list[tuple[str, tuple[ImageRecord, ...]]]] = defaultdict(list)
    for digest, group in duplicate_groups.items():
        by_label[group[0].label].append((digest, group))

    partitions: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}
    forced_test_digests: set[str] = set()
    supplied_test_images = 0
    duplicate_promotions = 0
    for label, groups in sorted(by_label.items()):
        total = sum(len(group) for _, group in groups)
        forced = []
        candidates = []
        for digest, group in groups:
            supplied = [row for row in group if source_splits[row.image_id] == "test"]
            if supplied:
                forced.append((digest, group))
                forced_test_digests.add(digest)
                supplied_test_images += len(supplied)
                duplicate_promotions += len(group) - len(supplied)
            else:
                candidates.append((digest, group))

        ordered = sorted(candidates, key=lambda item: (split_rank(OFFICIAL_SEED, label, item[0]), item[0]))
        test_rows = [row for _, group in forced for row in group]
        test_target = total * 0.20
        development_groups = []
        for digest, group in ordered:
            before = abs(len(test_rows) - test_target)
            after = abs(len(test_rows) + len(group) - test_target)
            if len(test_rows) < test_target and after <= before:
                test_rows.extend(group)
            else:
                development_groups.append((digest, group))

        train_rows: list[ImageRecord] = []
        val_rows: list[ImageRecord] = []
        train_target = total * 0.60
        val_target = total * 0.20
        for _, group in development_groups:
            size = len(group)
            train_error = (len(train_rows) + size - train_target) ** 2 + (
                len(val_rows) - val_target
            ) ** 2
            val_error = (len(train_rows) - train_target) ** 2 + (
                len(val_rows) + size - val_target
            ) ** 2
            (train_rows if train_error <= val_error else val_rows).extend(group)
        partitions["train"].extend(train_rows)
        partitions["val"].extend(val_rows)
        partitions["test"].extend(test_rows)

    frozen = {name: tuple(sorted(rows)) for name, rows in partitions.items()}
    development = (*frozen["train"], *frozen["val"])
    if any(source_splits[row.image_id] == "test" for row in development):
        raise RuntimeError("PlantDoc supplied-test image leaked into train/validation")
    if any(row.sha256 in forced_test_digests for row in development):
        raise RuntimeError("PlantDoc supplied-test duplicate group leaked into train/validation")
    duplicate_sizes = sorted(len(group) for group in duplicate_groups.values() if len(group) > 1)
    report = {
        "policy": f"{REPLICATION_POLICY}+plantdoc-supplied-test-isolation-v1",
        "seed": OFFICIAL_SEED,
        "min_class_support": 30,
        "input_count": len(records),
        "retained_count": len(retained),
        "retained_class_count": len(retained_labels),
        "excluded_classes": dict(
            sorted((label, count) for label, count in all_counts.items() if label not in retained_labels)
        ),
        "split_counts": {name: len(rows) for name, rows in frozen.items()},
        "class_distribution": {
            name: dict(sorted(Counter(row.label for row in rows).items()))
            for name, rows in frozen.items()
        },
        "exact_duplicate_groups": len(duplicate_groups),
        "duplicate_group_count": len(duplicate_sizes),
        "duplicate_image_count": sum(duplicate_sizes),
        "supplied_test_image_count": supplied_test_images,
        "supplied_test_duplicate_group_count": len(forced_test_digests),
        "train_source_images_promoted_with_test_duplicates": duplicate_promotions,
        "supplied_test_leakage_to_development": 0,
    }
    return PreparedSplits(partitions=frozen, report=report)


def write_source_manifest(
    records: list[ImageRecord], source_splits: dict[str, str], path: Path
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite image manifest: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in sorted(records):
            handle.write(
                json.dumps(
                    {
                        "image": record.image_id,
                        "image_path": record.image_path,
                        "label": record.label,
                        "sha256": record.sha256,
                        "source_split": source_splits[record.image_id],
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def write_all_csv(splits: Path) -> Path:
    output = splits / "all.csv"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite development manifest: {output}")
    with output.open("x", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for name in ("train.csv", "val.csv"):
            with (splits / name).open(newline="", encoding="utf-8") as source:
                rows = csv.DictReader(source)
                if tuple(rows.fieldnames or ()) != CSV_FIELDS:
                    raise RuntimeError(f"unexpected split schema: {name}")
                writer.writerows(rows)
    return output


def validate_counts(
    dataset: str,
    records: list[ImageRecord],
    prepared: Any,
    *,
    small_class_exclusions: int,
    label_conflict_exclusions: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    raw = (len(records), len({row.label for row in records}))
    retained = (
        int(prepared.report["retained_count"]),
        int(prepared.report["retained_class_count"]),
    )
    expected_raw = tuple(SOURCES[dataset]["expected_raw"])
    expected_retained = tuple(SOURCES[dataset]["expected_retained"])
    if raw != expected_raw:
        raise RuntimeError(f"{dataset} raw count mismatch: expected={expected_raw}, actual={raw}")
    reconciled_retained_images = raw[0] - small_class_exclusions - label_conflict_exclusions
    print(
        f"RETAINED_RECONCILIATION dataset={dataset} raw_count={raw[0]} "
        f"minus small_class_exclusions={small_class_exclusions} "
        f"minus label_conflict_exclusions={label_conflict_exclusions} "
        f"equals={reconciled_retained_images} actual_retained_count={retained[0]}",
        flush=True,
    )
    if retained[0] != reconciled_retained_images:
        raise RuntimeError(
            f"{dataset} retained count does not reconcile: raw_count={raw[0]} - "
            f"small_class_exclusions={small_class_exclusions} - "
            f"label_conflict_exclusions={label_conflict_exclusions} = "
            f"{reconciled_retained_images}, but actual retained_count={retained[0]}"
        )
    if label_conflict_exclusions == 0 and retained != expected_retained:
        raise RuntimeError(
            f"{dataset} retained count mismatch: expected={expected_retained}, actual={retained}"
        )
    return raw, retained


def prepare(dataset: str) -> None:
    dataset_root = ROOT / "ssl_hdc_pilot/replication/data" / dataset
    archive = dataset_root / "source_download.zip"
    receipt = write_receipt(dataset)
    extracted = dataset_root / "extracted"
    staging = ROOT / ".codex-tmp/extraction" / dataset
    archive_existing(staging, dataset, "partial")
    archive_existing(extracted, dataset, "prior")

    records, collisions, source_splits, byte_sizes = extract_collision_safe(
        dataset, archive, staging, extracted
    )
    clean_records, conflict_records = split_label_conflicts(records)
    label_conflicts_path = write_label_conflicts(dataset_root, conflict_records, byte_sizes)
    prepared = (
        build_plantdoc_splits(clean_records, source_splits)
        if dataset == "PlantDoc"
        else build_replication_splits(clean_records, seed=OFFICIAL_SEED)
    )
    small_class_exclusions = sum(prepared.report["excluded_classes"].values())
    raw_counts, retained_counts = validate_counts(
        dataset,
        records,
        prepared,
        small_class_exclusions=small_class_exclusions,
        label_conflict_exclusions=len(conflict_records),
    )
    shutil.move(str(staging), str(extracted))

    manifest = dataset_root / "sha256_manifest.jsonl"
    write_source_manifest(records, source_splits, manifest)
    collisions_path = dataset_root / "collisions.json"
    if collisions_path.exists():
        raise FileExistsError(f"refusing to overwrite collision record: {collisions_path}")
    collisions_path.write_text(
        json.dumps(collisions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    splits = dataset_root / "splits"
    write_prepared_splits(
        prepared,
        splits,
        source_manifest=manifest.relative_to(ROOT).as_posix(),
    )
    all_csv = write_all_csv(splits)
    if (splits / "test.csv").exists():
        raise RuntimeError("sealed-test violation: test.csv was written")
    development_rows = len(prepared.partitions["train"]) + len(prepared.partitions["val"])
    with all_csv.open(newline="", encoding="utf-8") as handle:
        development_manifest = list(csv.DictReader(handle))
    all_rows = len(development_manifest)
    if all_rows != development_rows:
        raise RuntimeError(
            f"development manifest count mismatch: expected={development_rows}, actual={all_rows}"
        )
    if dataset == "PlantDoc":
        leaked = [
            row["image"]
            for row in development_manifest
            if source_splits[row["image"]] == "test"
        ]
        if leaked:
            raise RuntimeError(
                f"PlantDoc supplied-test leakage into splits/all.csv: {len(leaked)} rows"
            )
    preparation_receipt = dataset_root / "preparation_receipt.json"
    preparation_receipt.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "prepared_utc": datetime.now(timezone.utc).isoformat(),
                "acquisition_receipt": receipt.relative_to(ROOT).as_posix(),
                "archive_hash_recomputed": False,
                "raw_images": raw_counts[0],
                "raw_classes": raw_counts[1],
                "small_class_exclusions": small_class_exclusions,
                "label_conflict_exclusions": len(conflict_records),
                "label_conflicts_path": label_conflicts_path.relative_to(ROOT).as_posix(),
                "retained_images": retained_counts[0],
                "retained_classes": retained_counts[1],
                "excluded_classes": prepared.report["excluded_classes"],
                "split_seed": OFFICIAL_SEED,
                "split_counts": prepared.report["split_counts"],
                "duplicate_group_count": prepared.report["duplicate_group_count"],
                "duplicate_image_count": prepared.report["duplicate_image_count"],
                "supplied_test_image_count": prepared.report.get(
                    "supplied_test_image_count"
                ),
                "supplied_test_duplicate_group_count": prepared.report.get(
                    "supplied_test_duplicate_group_count"
                ),
                "train_source_images_promoted_with_test_duplicates": prepared.report.get(
                    "train_source_images_promoted_with_test_duplicates"
                ),
                "supplied_test_leakage_to_development": prepared.report.get(
                    "supplied_test_leakage_to_development"
                ),
                "collision_group_count": len(
                    {row["collision_key"] for row in collisions}
                ),
                "collision_record_count": len(collisions),
                "development_rows": development_rows,
                "test_csv_written": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"PREPARED dataset={dataset} raw={raw_counts} retained={retained_counts} "
        f"splits={prepared.report['split_counts']} duplicates={prepared.report['duplicate_group_count']} "
        f"collisions={len(collisions)} label_conflict_exclusions={len(conflict_records)} "
        f"all_csv_rows={development_rows}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=(*SOURCES, "all"))
    parser.add_argument("--receipts-only", action="store_true")
    args = parser.parse_args()
    datasets = tuple(SOURCES) if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        receipt = write_receipt(dataset)
        print(f"RECEIPT dataset={dataset} path={receipt.relative_to(ROOT)}", flush=True)
        if not args.receipts_only:
            prepare(dataset)


if __name__ == "__main__":
    main()
