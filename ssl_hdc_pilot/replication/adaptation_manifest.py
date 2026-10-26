"""Build the label-free, train-only DINOv3 adaptation manifest for one replication dataset.

Each replication dataset (PlantDoc, PlantWild, PlantVillage) has an approved
60/20/20 content-grouped split under ``data/<Dataset>/splits/``.  This module
turns the ``train.csv`` partition of that split into
``data/<Dataset>/adaptation_train_manifest.csv``: a label-free manifest that
carries only ``image_id``, ``image_path``, and ``source_pool`` -- the three
columns ``ssl_hdc_pilot.scripts.adapt_dinov3.load_sanctioned_manifest``
requires, with the forbidden ``label``/``crop_disease_stage`` columns dropped.

``image_id`` is each row's ``image`` field verbatim (not a derived id): this
is the same convention ``extract_embeddings.py``'s ``load_records()`` already
falls back to for these split CSVs (they have no ``image_id`` column), and it
is what ``adapt_dinov3.py``'s ``--replication-dataset`` mode cross-checks
against ``train.csv``/``val.csv`` identities, so it must match exactly.

Consumed by ``adapt_dinov3.py --replication-dataset <Dataset>`` (added by the
coordinator's ruling): that opt-in mode keys the expected row count,
``source_pool``, and image-path root to this dataset's own train split
instead of the original single-dataset study's fixed 51,906-row Data/train
pool. Without ``--replication-dataset``, adapt_dinov3.py's behaviour is
unchanged and this manifest is (deliberately) not loadable by it.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPLICATION_DATA_ROOT = REPO_ROOT / "ssl_hdc_pilot" / "replication" / "data"
DATASETS = ("PlantDoc", "PlantWild", "PlantVillage")
MANIFEST_FIELDS = ("image_id", "image_path", "source_pool")

INCOMPATIBILITY_NOTE = (
    "Matches adapt_dinov3.load_sanctioned_manifest's required column schema "
    "(image_id, image_path, source_pool; no label columns). Loadable via "
    "adapt_dinov3.py --replication-dataset <Dataset>, which keys the expected "
    "row count/source_pool/path-root to this dataset instead of the original "
    "study's fixed 51906-row Data/train pool. Without that flag, "
    "adapt_dinov3.py's original behaviour is unchanged and this manifest is "
    "not loadable by it (by design)."
)


@dataclass(frozen=True)
class ManifestBuildResult:
    dataset: str
    manifest_csv: Path
    summary_json: Path
    row_count: int
    train_csv_row_count: int
    skipped_existing: bool


def _read_split_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"replication split CSV is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"image", "label", "split_partition", "image_path"}
        if not required <= fields:
            raise RuntimeError(f"{path} is missing columns: {sorted(required - fields)}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"replication split CSV has no rows: {path}")
    return rows


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_adaptation_manifest(
    dataset: str, *, data_root: Path = REPLICATION_DATA_ROOT
) -> ManifestBuildResult:
    if dataset not in DATASETS:
        raise ValueError(f"unknown replication dataset: {dataset!r}; expected one of {DATASETS}")

    dataset_root = data_root / dataset
    train_csv = dataset_root / "splits" / "train.csv"
    all_csv = dataset_root / "splits" / "all.csv"
    manifest_csv = dataset_root / "adaptation_train_manifest.csv"
    summary_json = dataset_root / "adaptation_train_manifest_summary.json"

    train_rows = _read_split_csv(train_csv)
    non_train = [row for row in train_rows if row["split_partition"] != "train"]
    if non_train:
        raise RuntimeError(
            f"{dataset} train.csv contains {len(non_train)} rows that are not "
            "split_partition=='train'"
        )

    all_rows = _read_split_csv(all_csv)
    all_train_rows = [row for row in all_rows if row["split_partition"] == "train"]
    if len(all_train_rows) != len(train_rows):
        raise RuntimeError(
            f"{dataset}: train.csv has {len(train_rows)} rows but all.csv has "
            f"{len(all_train_rows)} rows marked split_partition=='train'"
        )
    if {row["image"] for row in all_train_rows} != {row["image"] for row in train_rows}:
        raise RuntimeError(
            f"{dataset}: train.csv identities do not match the train-partition "
            "rows of all.csv"
        )

    extracted_root = (dataset_root / "extracted").resolve()
    seen_images: set[str] = set()
    manifest_rows: list[dict[str, str]] = []
    for row in sorted(train_rows, key=lambda r: r["image"]):
        image_value = row["image"].strip()
        if not image_value or image_value in seen_images:
            raise RuntimeError(f"{dataset}: invalid or duplicate image id: {image_value!r}")
        seen_images.add(image_value)

        raw_path = row["image_path"].strip()
        if not raw_path:
            raise RuntimeError(f"{dataset}: row for {image_value!r} has no image_path")
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = REPO_ROOT / image_path
        resolved = image_path.resolve()
        if not resolved.is_relative_to(extracted_root):
            raise RuntimeError(
                f"{dataset}: train image escapes its extracted pool: {resolved}"
            )
        if not resolved.is_file():
            raise FileNotFoundError(f"{dataset}: missing train image: {resolved}")

        # image_id is the split CSV's own "image" field verbatim (same
        # convention extract_embeddings.py's load_records() already falls
        # back to for these CSVs, and the same one adapt_dinov3.py's
        # replication mode cross-checks against train.csv/val.csv). This
        # must match exactly for that cross-check to work; a derived id
        # (e.g. a filename stem) would never be found in train_ids.
        manifest_rows.append(
            {
                "image_id": image_value,
                "image_path": _display_path(resolved),
                "source_pool": f"ssl_hdc_pilot/replication/data/{dataset}/train",
            }
        )

    if len(manifest_rows) != len(train_rows):
        raise RuntimeError(f"{dataset}: manifest row count drifted from train.csv row count")

    skipped_existing = False
    if manifest_csv.is_file():
        existing = _read_split_csv_generic(manifest_csv, MANIFEST_FIELDS)
        if existing == manifest_rows:
            skipped_existing = True
        else:
            raise FileExistsError(
                f"refusing to overwrite a differing adaptation manifest: {manifest_csv}"
            )
    else:
        _write_manifest_csv(manifest_csv, manifest_rows)

    summary = {
        "dataset": dataset,
        "row_count": len(manifest_rows),
        "train_csv_row_count": len(train_rows),
        "train_csv": _display_path(train_csv),
        "all_csv": _display_path(all_csv),
        "labels_used": False,
        "adapter_expected_columns": list(MANIFEST_FIELDS),
        "loadable_via": "adapt_dinov3.py --replication-dataset <Dataset>",
        "note": INCOMPATIBILITY_NOTE,
    }
    if not summary_json.is_file():
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return ManifestBuildResult(
        dataset=dataset,
        manifest_csv=manifest_csv,
        summary_json=summary_json,
        row_count=len(manifest_rows),
        train_csv_row_count=len(train_rows),
        skipped_existing=skipped_existing,
    )


def _read_split_csv_generic(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise RuntimeError(f"unexpected manifest columns in {path}: {reader.fieldnames}")
        return list(reader)


def _write_manifest_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, default=REPLICATION_DATA_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_adaptation_manifest(args.dataset, data_root=args.data_root)
    print(
        json.dumps(
            {
                "dataset": result.dataset,
                "manifest_csv": str(result.manifest_csv),
                "summary_json": str(result.summary_json),
                "row_count": result.row_count,
                "train_csv_row_count": result.train_csv_row_count,
                "skipped_existing": result.skipped_existing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
