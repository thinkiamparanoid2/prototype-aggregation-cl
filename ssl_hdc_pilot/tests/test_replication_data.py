from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from ssl_hdc_pilot.replication.data import (
    CSV_FIELDS,
    ImageRecord,
    assert_not_sealed_test_path,
    build_replication_splits,
    group_exact_duplicates,
    write_sha256_manifest,
    write_prepared_splits,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _records() -> list[ImageRecord]:
    rows: list[ImageRecord] = []
    for label, count in (("healthy", 35), ("rust", 34), ("tail", 29)):
        for index in range(count):
            digest_source = f"{label}-{index}"
            rows.append(ImageRecord(f"{label}-{index}", label, f"images/{label}/{index}.jpg", _digest(digest_source)))
    # An exact, same-label duplicate must stay with its source image.
    rows.append(ImageRecord("healthy-copy", "healthy", "images/healthy/copy.jpg", _digest("healthy-0")))
    return rows


class ReplicationDataTests(unittest.TestCase):
    def test_support_rule_duplicate_grouping_and_determinism(self) -> None:
        first = build_replication_splits(_records(), seed=917)
        second = build_replication_splits(_records(), seed=917)
        self.assertEqual(first, second)
        self.assertEqual(first.report["excluded_classes"], {"tail": 29})
        self.assertEqual(first.report["retained_count"], 70)
        owners = {
            record.image_id: partition
            for partition, rows in first.partitions.items()
            for record in rows
        }
        self.assertEqual(owners["healthy-0"], owners["healthy-copy"])
        self.assertEqual(sum(len(rows) for rows in first.partitions.values()), 70)

    def test_cross_label_content_duplicate_is_rejected(self) -> None:
        records = _records()
        records[-1] = ImageRecord("bad", "rust", "images/rust/bad.jpg", _digest("healthy-0"))
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            group_exact_duplicates(records)

    def test_test_path_guard_rejects_opening_any_test_csv(self) -> None:
        with self.assertRaisesRegex(PermissionError, "sealed"):
            assert_not_sealed_test_path(Path("new_dataset/test.csv"))
        with self.assertRaisesRegex(PermissionError, "sealed"):
            assert_not_sealed_test_path(Path("new_dataset/sealed_test_receipt.json"))
        assert_not_sealed_test_path(Path("new_dataset/val.csv"))

    def test_writer_has_trusted_schema_never_writes_test_csv_and_refuses_overwrite(self) -> None:
        prepared = build_replication_splits(_records(), seed=917)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "plantdoc"
            write_prepared_splits(prepared, output, source_manifest="manifests/plantdoc.jsonl")
            self.assertFalse((output / "test.csv").exists())
            self.assertTrue((output / "sealed_test_receipt.json").is_file())
            with (output / "train.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(tuple(csv.DictReader(handle).fieldnames or ()), CSV_FIELDS)
            with self.assertRaises(FileExistsError):
                write_prepared_splits(prepared, output, source_manifest="manifests/plantdoc.jsonl")

    def test_source_hash_manifest_is_stable_and_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "plantdoc.sha256.jsonl"
            write_sha256_manifest(reversed(_records()), manifest)
            rows = [json_row for json_row in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), len(_records()))
            self.assertIn('"sha256"', rows[0])
            with self.assertRaises(FileExistsError):
                write_sha256_manifest(_records(), manifest)


if __name__ == "__main__":
    unittest.main()
