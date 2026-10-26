"""Build preregistered trusted-label splits after supervisor policy approval."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "Data" / "test.csv"
DEFAULT_OUTPUT = REPO_ROOT / "ssl_hdc_pilot" / "trusted_splits"
EXPECTED_TRUSTED_COUNT = 5_767
EXPECTED_SOURCE_CLASS_COUNT = 25
EXPECTED_HEADLINE_CLASS_COUNT = 19
OFFICIAL_SEED = 20_260_716
PROVENANCE_VERSION = "ssl-hdc-trusted-split-v1"
CANONICAL_SOURCE_CSV = "Data/test.csv"
PROVENANCE_FIELDS = [
    "split_partition",
    "split_policy",
    "split_seed",
    "supervisor_approved",
    "provenance_version",
    "source_csv",
    "image_path",
]
EXPECTED_TAIL_CLASSES = {
    "6_a11_2": 12,
    "6_a12_1": 13,
    "6_b4_3": 14,
    "6_b4_1": 18,
    "6_b5_1": 21,
    "6_a12_2": 29,
}
MANDATORY_DISCLOSURES = [
    "19-class and 25-class macro-F1 are not comparable",
    "both Option A and Option B results must be reported regardless of outcome",
    "Option B tail classes have no untouched holdout; report confidence intervals",
]


def _normalize_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        image_id = (row.get("image") or row.get("image_id") or "").strip()
        label = (row.get("label") or row.get("crop_disease_stage") or "").strip()
        if not image_id or not label:
            raise ValueError("trusted rows require image/image_id and label columns")
        if image_id in seen:
            raise ValueError(f"duplicate trusted image id: {image_id}")
        seen.add(image_id)
        normalized.append({"image": image_id, "label": label})
    if not normalized:
        raise ValueError("trusted input is empty")
    return normalized


def _distribution(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["label"] for row in rows).items()))


def build_option_a(
    rows: Sequence[dict[str, str]], *, seed: int, min_class_count: int = 30
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    """Build the fixed 60/20/20 headline split after dropping rare classes."""
    normalized = _normalize_rows(rows)
    counts = Counter(row["label"] for row in normalized)
    kept_labels = {label for label, count in counts.items() if count >= min_class_count}
    retained = [row for row in normalized if row["label"] in kept_labels]
    dropped = dict(sorted((label, count) for label, count in counts.items() if label not in kept_labels))
    if len(kept_labels) < 2:
        raise ValueError("Option A needs at least two retained classes")

    indices = np.arange(len(retained))
    labels = np.asarray([row["label"] for row in retained])
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=seed + 1,
        stratify=labels[train_val_idx],
    )
    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    partitions = {
        name: [retained[int(index)] for index in sorted(partition_indices)]
        for name, partition_indices in split_indices.items()
    }
    report = {
        "policy": "option-a-19class",
        "seed": seed,
        "min_class_count": min_class_count,
        "input_count": len(normalized),
        "source_class_count": len(counts),
        "retained_count": len(retained),
        "retained_class_count": len(kept_labels),
        "dropped_classes": dropped,
        "split_counts": {name: len(partition) for name, partition in partitions.items()},
        "class_distribution": {
            name: _distribution(partition) for name, partition in partitions.items()
        },
        "disclosures": MANDATORY_DISCLOSURES,
    }
    return partitions, report


def build_option_b(
    rows: Sequence[dict[str, str]], *, seed: int, folds: int = 5, repeats: int = 10
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build all-class repeated stratified fold assignments for the appendix."""
    normalized = _normalize_rows(rows)
    counts = Counter(row["label"] for row in normalized)
    if folds < 2 or repeats < 1:
        raise ValueError("Option B requires at least 2 folds and 1 repeat")
    if min(counts.values()) < folds:
        raise ValueError("every class must contain at least one sample per fold")

    labels = np.asarray([row["label"] for row in normalized])
    splitter = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=repeats,
        random_state=seed,
    )
    assignments: list[dict[str, Any]] = []
    fold_distributions: dict[str, dict[str, int]] = {}
    for split_index, (_, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        repeat = split_index // folds
        fold = split_index % folds
        selected = [normalized[int(index)] for index in test_idx]
        fold_distributions[f"repeat_{repeat:02d}_fold_{fold:02d}"] = _distribution(selected)
        for row in selected:
            assignments.append(
                {
                    "image": row["image"],
                    "label": row["label"],
                    "repeat": repeat,
                    "fold": fold,
                }
            )
    report = {
        "policy": "option-b-cv",
        "seed": seed,
        "folds": folds,
        "repeats": repeats,
        "input_count": len(normalized),
        "class_count": len(counts),
        "class_distribution": dict(sorted(counts.items())),
        "fold_distribution": fold_distributions,
        "disclosures": MANDATORY_DISCLOSURES,
    }
    return assignments, report


def validate_write_permission(*, dry_run: bool, supervisor_approved: bool) -> None:
    if not dry_run and not supervisor_approved:
        raise RuntimeError(
            "official trusted-split writing requires explicit supervisor approval; "
            "use --dry-run for inspection"
        )


def validate_official_input_path(input_csv: Path, *, dry_run: bool) -> None:
    if not dry_run and input_csv.resolve() != DEFAULT_INPUT.resolve():
        raise RuntimeError(
            "official trusted splits must be built directly from canonical Data/test.csv; "
            "copies are permitted only with --dry-run"
        )


def validate_official_report(
    *,
    policy: str,
    report: dict[str, Any],
    seed: int,
    min_class_count: int,
    folds: int,
    repeats: int,
) -> None:
    if report.get("input_count") != EXPECTED_TRUSTED_COUNT:
        raise RuntimeError("official split requires exactly 5,767 trusted rows")
    if seed != OFFICIAL_SEED:
        raise RuntimeError("official splits are locked to fixed seed 20260716")
    if policy == "option-a-19class":
        if min_class_count != 30:
            raise RuntimeError("official Option A is locked to the under-30 tail policy")
        if report.get("source_class_count") != EXPECTED_SOURCE_CLASS_COUNT:
            raise RuntimeError("official Option A requires all 25 source classes")
        if report.get("retained_class_count") != EXPECTED_HEADLINE_CLASS_COUNT:
            raise RuntimeError("official Option A requires exactly 19 retained classes")
        if report.get("dropped_classes") != EXPECTED_TAIL_CLASSES:
            raise RuntimeError("official Option A tail classes/counts differ from preregistration")
    elif policy == "option-b-cv":
        if folds != 5 or repeats != 10:
            raise RuntimeError("official Option B is locked to 5 folds x 10 repeats")
        if report.get("class_count") != EXPECTED_SOURCE_CLASS_COUNT:
            raise RuntimeError("official Option B requires all 25 classes")
    else:
        raise ValueError(f"unknown tail policy: {policy}")


def _write_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_official_provenance(
    rows: Sequence[dict[str, Any]], *, partition: str, policy: str
) -> list[dict[str, Any]]:
    """Attach immutable provenance used by downstream trusted-split consumers."""
    return [
        {
            **row,
            "split_partition": partition,
            "split_policy": policy,
            "split_seed": OFFICIAL_SEED,
            "supervisor_approved": "true",
            "provenance_version": PROVENANCE_VERSION,
            "source_csv": CANONICAL_SOURCE_CSV,
            "image_path": f"Data/test/{row['image']}/{row['image']}.jpg",
        }
        for row in rows
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=["option-a-19class", "option-b-cv"],
        default="option-a-19class",
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=OFFICIAL_SEED)
    parser.add_argument("--min-class-count", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--supervisor-approved", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_write_permission(
        dry_run=args.dry_run,
        supervisor_approved=args.supervisor_approved,
    )
    validate_official_input_path(args.input_csv, dry_run=args.dry_run)
    with args.input_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if args.policy == "option-a-19class":
        partitions, report = build_option_a(
            rows,
            seed=args.seed,
            min_class_count=args.min_class_count,
        )
        if not args.dry_run:
            validate_official_report(
                policy=args.policy,
                report=report,
                seed=args.seed,
                min_class_count=args.min_class_count,
                folds=args.folds,
                repeats=args.repeats,
            )
            for name, partition in partitions.items():
                rows_with_provenance = add_official_provenance(
                    partition,
                    partition=name,
                    policy=args.policy,
                )
                _write_rows(
                    args.output_dir / f"{name}.csv",
                    rows_with_provenance,
                    ["image", "label", *PROVENANCE_FIELDS],
                )
    else:
        assignments, report = build_option_b(
            rows,
            seed=args.seed,
            folds=args.folds,
            repeats=args.repeats,
        )
        if not args.dry_run:
            validate_official_report(
                policy=args.policy,
                report=report,
                seed=args.seed,
                min_class_count=args.min_class_count,
                folds=args.folds,
                repeats=args.repeats,
            )
            _write_rows(
                args.output_dir / "option_b_fold_assignments.csv",
                add_official_provenance(
                    assignments,
                    partition="cv-assignment",
                    policy=args.policy,
                ),
                ["image", "label", "repeat", "fold", *PROVENANCE_FIELDS],
            )

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "distribution_report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({**report, "dry_run": args.dry_run}, indent=2))


if __name__ == "__main__":
    main()
