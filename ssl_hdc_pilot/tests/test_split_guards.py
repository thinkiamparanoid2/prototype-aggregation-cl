from __future__ import annotations

import unittest
from pathlib import Path

from ssl_hdc_pilot.scripts.build_trusted_splits import (
    DEFAULT_INPUT,
    EXPECTED_TAIL_CLASSES,
    validate_official_input_path,
    validate_official_report,
    validate_write_permission,
)


class TrustedSplitGuardTests(unittest.TestCase):
    def test_official_write_requires_explicit_supervisor_approval(self) -> None:
        validate_write_permission(dry_run=True, supervisor_approved=False)
        with self.assertRaisesRegex(RuntimeError, "supervisor approval"):
            validate_write_permission(dry_run=False, supervisor_approved=False)

    def test_official_input_must_be_the_canonical_trusted_csv(self) -> None:
        validate_official_input_path(DEFAULT_INPUT, dry_run=False)
        validate_official_input_path(Path("smoke_copy.csv"), dry_run=True)
        with self.assertRaisesRegex(RuntimeError, "Data/test.csv"):
            validate_official_input_path(Path("smoke_copy.csv"), dry_run=False)

    def test_option_a_official_report_must_match_preregistered_19_class_policy(self) -> None:
        report = {
            "input_count": 5767,
            "source_class_count": 25,
            "retained_class_count": 19,
            "dropped_classes": EXPECTED_TAIL_CLASSES,
        }
        validate_official_report(
            policy="option-a-19class",
            report=report,
            seed=20260716,
            min_class_count=30,
            folds=5,
            repeats=10,
        )
        with self.assertRaisesRegex(RuntimeError, "19 retained classes"):
            validate_official_report(
                policy="option-a-19class",
                report={**report, "retained_class_count": 18},
                seed=20260716,
                min_class_count=30,
                folds=5,
                repeats=10,
            )
        with self.assertRaisesRegex(RuntimeError, "fixed seed 20260716"):
            validate_official_report(
                policy="option-a-19class",
                report=report,
                seed=123,
                min_class_count=30,
                folds=5,
                repeats=10,
            )

    def test_option_b_official_report_is_locked_to_25_classes_and_5x10(self) -> None:
        report = {"input_count": 5767, "class_count": 25}
        validate_official_report(
            policy="option-b-cv",
            report=report,
            seed=20260716,
            min_class_count=30,
            folds=5,
            repeats=10,
        )
        with self.assertRaisesRegex(RuntimeError, "5 folds x 10 repeats"):
            validate_official_report(
                policy="option-b-cv",
                report=report,
                seed=20260716,
                min_class_count=30,
                folds=4,
                repeats=10,
            )


if __name__ == "__main__":
    unittest.main()
