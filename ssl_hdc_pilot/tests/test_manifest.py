from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ssl_hdc_pilot.scripts.build_unlabeled_manifest import build_manifest


class UnlabeledManifestTests(unittest.TestCase):
    def test_manifest_keeps_train_once_documents_val_overlap_and_excludes_trusted(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            train_root = root / "train"
            val_root = root / "val"
            test_root = root / "test"
            for image_id in ("1", "2", "3"):
                sample = train_root / image_id
                sample.mkdir(parents=True)
                (sample / f"{image_id}.jpg").write_bytes(b"jpg")
            (val_root / "2").mkdir(parents=True)
            (test_root / "9").mkdir(parents=True)

            output_csv = root / "manifest.csv"
            summary_json = root / "summary.json"
            summary = build_manifest(
                train_root=train_root,
                val_root=val_root,
                trusted_root=test_root,
                output_csv=output_csv,
                summary_json=summary_json,
            )

            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["image_id"] for row in rows], ["1", "2", "3"])
            self.assertEqual([row["is_val_duplicate"] for row in rows], ["0", "1", "0"])
            self.assertTrue(all(row["image_path"].endswith(f"{row['image_id']}.jpg") for row in rows))
            self.assertEqual(summary["train_folder_count"], 3)
            self.assertEqual(summary["val_overlap_count"], 1)
            self.assertEqual(summary["trusted_overlap_excluded_count"], 0)
            self.assertEqual(summary["manifest_count"], 3)
            self.assertEqual(json.loads(summary_json.read_text(encoding="utf-8")), summary)

    def test_manifest_never_contains_a_trusted_id(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            for pool in ("train", "test"):
                sample = root / pool / "7"
                sample.mkdir(parents=True)
                (sample / "7.jpg").write_bytes(b"jpg")
            (root / "val").mkdir()

            summary = build_manifest(
                train_root=root / "train",
                val_root=root / "val",
                trusted_root=root / "test",
                output_csv=root / "manifest.csv",
                summary_json=root / "summary.json",
            )

            self.assertEqual(summary["trusted_overlap_excluded_count"], 1)
            with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])


if __name__ == "__main__":
    unittest.main()
