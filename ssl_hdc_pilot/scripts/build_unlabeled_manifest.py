"""Build the only sanctioned unlabeled image manifest for Arm-D adaptation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_ROOT = REPO_ROOT / "Data" / "train"
DEFAULT_VAL_ROOT = REPO_ROOT / "Data" / "val"
DEFAULT_TRUSTED_ROOT = REPO_ROOT / "Data" / "test"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "ssl_hdc_pilot" / "results" / "unlabeled_manifest.csv"
DEFAULT_SUMMARY_JSON = (
    REPO_ROOT / "ssl_hdc_pilot" / "results" / "unlabeled_manifest_summary.json"
)


def _folder_ids(root: Path) -> set[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"image pool directory does not exist: {root}")
    return {path.name for path in root.iterdir() if path.is_dir()}


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _numeric_sort(ids: Iterable[str]) -> list[str]:
    def key(image_id: str) -> tuple[int, int | str]:
        return (0, int(image_id)) if image_id.isdigit() else (1, image_id)

    return sorted(ids, key=key)


def build_manifest(
    *,
    train_root: Path,
    val_root: Path,
    trusted_root: Path,
    output_csv: Path,
    summary_json: Path,
    expected_train_count: int | None = None,
    expected_val_overlap_count: int | None = None,
    expected_trusted_count: int | None = None,
) -> dict[str, object]:
    """Enumerate train once, mark its val duplicate subset, exclude trusted IDs."""
    train_ids = _folder_ids(train_root)
    val_ids = _folder_ids(val_root)
    trusted_ids = _folder_ids(trusted_root)
    val_overlap = train_ids & val_ids
    trusted_overlap = train_ids & trusted_ids
    sanctioned_ids = train_ids - trusted_ids

    expected = {
        "train folder": (len(train_ids), expected_train_count),
        "val overlap": (len(val_overlap), expected_val_overlap_count),
        "trusted folder": (len(trusted_ids), expected_trusted_count),
    }
    mismatches = [
        f"{name}: observed {observed}, expected {wanted}"
        for name, (observed, wanted) in expected.items()
        if wanted is not None and observed != wanted
    ]
    if mismatches:
        raise RuntimeError("dataset count guard failed; " + "; ".join(mismatches))
    if val_ids - train_ids:
        raise RuntimeError(
            f"Data/val is not a pure duplicate subset of Data/train: "
            f"{len(val_ids - train_ids)} val-only IDs"
        )

    rows: list[dict[str, str | int]] = []
    for image_id in _numeric_sort(sanctioned_ids):
        image_path = train_root / image_id / f"{image_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"missing expected image: {image_path}")
        rows.append(
            {
                "image_id": image_id,
                "image_path": _display_path(image_path),
                "source_pool": "Data/train",
                "is_val_duplicate": int(image_id in val_overlap),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "image_path", "source_pool", "is_val_duplicate"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "train_folder_count": len(train_ids),
        "val_folder_count": len(val_ids),
        "val_overlap_count": len(val_overlap),
        "val_is_duplicate_subset_of_train": val_ids <= train_ids,
        "trusted_folder_count": len(trusted_ids),
        "trusted_overlap_excluded_count": len(trusted_overlap),
        "manifest_count": len(rows),
        "reconciliation": (
            f"{len(train_ids)} train folders - {len(trusted_overlap)} trusted overlaps "
            f"= {len(rows)} sanctioned images; {len(val_overlap)} val IDs are already "
            "inside train and are not added twice"
        ),
        "labels_used": False,
        "trusted_ids_present_in_manifest": False,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--val-root", type=Path, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--trusted-root", type=Path, default=DEFAULT_TRUSTED_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--expected-train-count", type=int, default=51_906)
    parser.add_argument("--expected-val-overlap-count", type=int, default=5_193)
    parser.add_argument("--expected-trusted-count", type=int, default=5_767)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_manifest(
        train_root=args.train_root,
        val_root=args.val_root,
        trusted_root=args.trusted_root,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        expected_train_count=args.expected_train_count,
        expected_val_overlap_count=args.expected_val_overlap_count,
        expected_trusted_count=args.expected_trusted_count,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
