from __future__ import annotations

import unittest
from collections import Counter

from ssl_hdc_pilot.scripts.build_trusted_splits import build_option_a, build_option_b


def _rows(counts: dict[str, int]) -> list[dict[str, str]]:
    rows = []
    index = 0
    for label, count in counts.items():
        for _ in range(count):
            rows.append({"image": str(index), "label": label})
            index += 1
    return rows


class TrustedSplitTests(unittest.TestCase):
    def test_option_a_is_deterministic_stratified_and_drops_only_tail(self) -> None:
        rows = _rows({"common_a": 10, "common_b": 10, "tail": 2})
        first, first_report = build_option_a(rows, seed=42, min_class_count=5)
        second, second_report = build_option_a(rows, seed=42, min_class_count=5)

        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)
        self.assertEqual(first_report["dropped_classes"], {"tail": 2})
        self.assertIn("both Option A and Option B", first_report["disclosures"][1])
        self.assertEqual(
            {name: len(partition) for name, partition in first.items()},
            {"train": 12, "val": 4, "test": 4},
        )
        for partition in first.values():
            self.assertEqual(Counter(row["label"] for row in partition), {"common_a": len(partition) // 2, "common_b": len(partition) // 2})
        all_ids = [row["image"] for partition in first.values() for row in partition]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_option_b_repeated_folds_are_deterministic_and_cover_each_id_once_per_repeat(self) -> None:
        rows = _rows({"a": 10, "b": 10, "c": 10})
        first, first_report = build_option_b(rows, seed=7, folds=5, repeats=2)
        second, second_report = build_option_b(rows, seed=7, folds=5, repeats=2)

        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)
        self.assertIn("not comparable", first_report["disclosures"][0])
        self.assertEqual(len(first), 60)
        for repeat in (0, 1):
            repeat_rows = [row for row in first if row["repeat"] == repeat]
            self.assertEqual(len({row["image"] for row in repeat_rows}), 30)
            for fold in range(5):
                fold_rows = [row for row in repeat_rows if row["fold"] == fold]
                self.assertEqual(Counter(row["label"] for row in fold_rows), {"a": 2, "b": 2, "c": 2})


if __name__ == "__main__":
    unittest.main()
