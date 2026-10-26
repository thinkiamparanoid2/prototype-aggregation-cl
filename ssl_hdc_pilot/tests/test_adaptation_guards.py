from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from ssl_hdc_pilot.scripts.adapt_dinov3 import (
    AdaptationConfig,
    RESULTS_ROOT,
    load_module_state,
    prepare_run,
    require_finite_loss,
    resume_config_payload,
    save_baseline_checkpoint,
    save_training_checkpoint,
    validate_config,
    validate_output_dir,
    validate_smoke_stop_after_epoch,
    write_history,
)
from ssl_hdc_pilot.scripts.extract_embeddings import LADDER_MODEL_SPECS


class AdaptationGuardTests(unittest.TestCase):
    def test_safe_smoke_budget_is_accepted(self) -> None:
        config = AdaptationConfig(smoke=True, max_images=256, max_steps=5, epochs=1)
        validate_config(config)

    def test_official_adaptable_model_ids_are_exactly_the_three_ladder_ids(self) -> None:
        adaptable = {
            model_id for model_id, spec in LADDER_MODEL_SPECS.items() if spec.adaptable
        }
        self.assertEqual(
            adaptable,
            {
                "facebook/dinov3-vits16-pretrain-lvd1689m",
                "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "facebook/dinov3-vitl16-pretrain-lvd1689m",
            },
        )
        for model_id in adaptable:
            validate_config(AdaptationConfig(model_id=model_id))
            validate_config(
                AdaptationConfig(
                    model_id=model_id,
                    smoke=True,
                    max_images=32,
                    max_steps=1,
                    epochs=1,
                )
            )

    def test_frozen_only_ceiling_rung_is_rejected_for_adaptation(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen-only ceiling rung"):
            validate_config(
                AdaptationConfig(model_id="facebook/dinov3-vit7b16-pretrain-lvd1689m")
            )
        with self.assertRaisesRegex(ValueError, "encoder-size ladder"):
            validate_config(
                AdaptationConfig(
                    model_id="facebook/not-authorized",
                )
            )

    def test_official_ladder_recipe_is_immutable(self) -> None:
        for field, value in (
            ("epochs", 4),
            ("learning_rate", 5e-5),
            ("ema_momentum", 0.997),
            ("batch_size", 16),
            ("device", "auto"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError,
                "preregistered ladder recipe",
            ):
                validate_config(AdaptationConfig(**{field: value}))
        validate_config(
            AdaptationConfig(
                smoke=True,
                max_images=2,
                max_steps=1,
                epochs=1,
                batch_size=2,
            )
        )

    def test_official_num_workers_is_a_hardware_knob_not_a_locked_recipe_field(self) -> None:
        # num_workers is deliberately excluded from the locked-recipe equality
        # check (see validate_config's comment): it is a DataLoader
        # hardware-throughput knob, not part of the scientific recipe, so an
        # official (non-smoke) run may set it to any non-negative value.
        validate_config(AdaptationConfig(num_workers=0))
        validate_config(AdaptationConfig(num_workers=3))
        validate_config(AdaptationConfig(num_workers=5))
        with self.assertRaisesRegex(ValueError, "num_workers cannot be negative"):
            validate_config(AdaptationConfig(num_workers=-1))
        # every other locked field must still be enforced alongside a
        # non-default num_workers.
        with self.assertRaisesRegex(ValueError, "preregistered ladder recipe"):
            validate_config(AdaptationConfig(num_workers=5, batch_size=16))

    def test_two_epoch_continuation_smoke_stays_within_ten_total_steps(self) -> None:
        validate_config(
            AdaptationConfig(smoke=True, max_images=8, max_steps=1, epochs=2)
        )
        with self.assertRaisesRegex(ValueError, "10 total steps"):
            validate_config(
                AdaptationConfig(smoke=True, max_images=8, max_steps=6, epochs=2)
            )

    def test_smoke_stop_control_preserves_resume_config(self) -> None:
        smoke = AdaptationConfig(smoke=True, max_images=8, max_steps=1, epochs=2)
        validate_smoke_stop_after_epoch(smoke, 1)
        validate_smoke_stop_after_epoch(smoke, None)
        with self.assertRaisesRegex(ValueError, "before the final epoch"):
            validate_smoke_stop_after_epoch(smoke, 2)
        with self.assertRaisesRegex(ValueError, "smoke-only"):
            validate_smoke_stop_after_epoch(AdaptationConfig(), 1)

    def test_same_config_resumes_last_complete_epoch_and_mismatch_hard_errors(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            manifest.write_text("image_id,image_path,source_pool\n1,x,Data/train\n", encoding="utf-8")
            config = AdaptationConfig(smoke=True, max_images=8, max_steps=2, epochs=1)
            payload = resume_config_payload(config, manifest)

            run = root / "arm_d_vitb16"
            self.assertEqual(prepare_run(run, payload).last_complete_epoch, -1)
            checkpoint = save_baseline_checkpoint(
                run,
                torch.nn.Linear(2, 2),
                metadata={
                    "model_id": AdaptationConfig().model_id,
                    "labels_used": False,
                    "trusted_pixels_used": False,
                },
            )
            load_module_state(
                checkpoint / "encoder.npz",
                torch.nn.Linear(2, 2),
                torch.device("cpu"),
            )
            self.assertEqual(prepare_run(run, payload).last_complete_epoch, 0)

            corrupt_run = root / "corrupt" / "arm_d_vitb16"
            prepare_run(corrupt_run, payload)
            corrupt_checkpoint = corrupt_run / "checkpoint_epoch0"
            corrupt_checkpoint.mkdir()
            (corrupt_checkpoint / "encoder.npz").write_bytes(b"not-an-npz")
            (corrupt_checkpoint / "checkpoint_metadata.json").write_text(
                json.dumps({"epoch": 0, "complete": True}), encoding="utf-8"
            )
            self.assertEqual(prepare_run(corrupt_run, payload).last_complete_epoch, -1)

            changed = resume_config_payload(
                AdaptationConfig(
                    smoke=True,
                    max_images=8,
                    max_steps=2,
                    epochs=1,
                    learning_rate=5e-5,
                ),
                manifest,
            )
            with self.assertRaisesRegex(RuntimeError, "config mismatch"):
                prepare_run(run, changed)

            changed_device = resume_config_payload(
                AdaptationConfig(
                    smoke=True,
                    max_images=8,
                    max_steps=2,
                    epochs=1,
                    device="cpu",
                ),
                manifest,
            )
            with self.assertRaisesRegex(RuntimeError, "config mismatch"):
                prepare_run(run, changed_device)

    def test_model_id_is_bound_to_its_size_specific_output_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "image_id,image_path,source_pool\n1,x,Data/train\n",
                encoding="utf-8",
            )
            vitb = "facebook/dinov3-vitb16-pretrain-lvd1689m"
            smoke_config = AdaptationConfig(
                model_id=vitb,
                smoke=True,
                max_images=2,
                max_steps=1,
                epochs=1,
            )
            payload = resume_config_payload(smoke_config, manifest)
            with self.assertRaisesRegex(RuntimeError, "arm_d_vitb16"):
                prepare_run(root / "arm_d_vits16", payload)
            self.assertEqual(
                prepare_run(root / "arm_d_vitb16", payload).last_complete_epoch,
                -1,
            )
            official_payload = resume_config_payload(
                AdaptationConfig(model_id=vitb),
                manifest,
            )
            with self.assertRaisesRegex(RuntimeError, "official vitb16 run"):
                validate_output_dir(root / "arm_d_vitb16", official_payload)
            self.assertEqual(
                validate_output_dir(
                    RESULTS_ROOT / "arm_d_vitb16",
                    official_payload,
                ),
                vitb,
            )

    def test_resume_refuses_checkpoint_cadence_gap(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "image_id,image_path,source_pool\n1,x,Data/train\n",
                encoding="utf-8",
            )
            payload = resume_config_payload(
                AdaptationConfig(smoke=True, max_images=8, max_steps=1, epochs=2),
                manifest,
            )
            run = root / "arm_d_vitb16"
            prepare_run(run, payload)
            student_encoder = torch.nn.Linear(2, 2)
            teacher_encoder = torch.nn.Linear(2, 2)
            student_head = torch.nn.Linear(2, 2)
            teacher_head = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(
                list(student_encoder.parameters()) + list(student_head.parameters()),
                lr=1e-5,
            )
            metadata = {
                "model_id": AdaptationConfig().model_id,
                "labels_used": False,
                "trusted_pixels_used": False,
            }
            save_baseline_checkpoint(run, teacher_encoder, metadata=metadata)
            save_training_checkpoint(
                output_dir=run,
                epoch=2,
                student_encoder=student_encoder,
                teacher_encoder=teacher_encoder,
                student_head=student_head,
                teacher_head=teacher_head,
                optimizer=optimizer,
                scaler_state={},
                center=torch.zeros(1, 2),
                metadata=metadata,
            )

            with self.assertRaisesRegex(RuntimeError, "checkpoint cadence gap"):
                prepare_run(run, payload)

    def test_smoke_budget_and_low_lr_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 512"):
            validate_config(AdaptationConfig(smoke=True, max_images=513, max_steps=5, epochs=1))
        with self.assertRaisesRegex(ValueError, "at most 10 steps"):
            validate_config(AdaptationConfig(smoke=True, max_images=256, max_steps=11, epochs=1))
        with self.assertRaisesRegex(ValueError, "low-LR ceiling"):
            validate_config(AdaptationConfig(smoke=True, max_images=256, max_steps=5, epochs=1, learning_rate=2e-4))
        with self.assertRaisesRegex(ValueError, "exactly 2 global and 4 local"):
            validate_config(AdaptationConfig(smoke=True, max_images=256, max_steps=5, epochs=1, local_crops=5))

    def test_required_mitigations_cannot_be_disabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "AMP"):
            validate_config(AdaptationConfig(smoke=True, max_images=256, max_steps=5, epochs=1, amp=False))
        with self.assertRaisesRegex(ValueError, "gradient checkpointing"):
            validate_config(AdaptationConfig(smoke=True, max_images=256, max_steps=5, epochs=1, gradient_checkpointing=False))
        with self.assertRaisesRegex(ValueError, "EMA"):
            validate_config(AdaptationConfig(smoke=True, max_images=256, max_steps=5, epochs=1, ema_momentum=0.9))

    def test_non_finite_loss_aborts_before_training_state_is_updated(self) -> None:
        require_finite_loss(torch.tensor(1.25), epoch=1, step=2)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(FloatingPointError, "epoch=1 step=2"):
                require_finite_loss(torch.tensor(value), epoch=1, step=2)

    def test_history_persists_peak_cuda_memory_allocated(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            history_path = Path(tmp) / "history.csv"
            write_history(
                history_path,
                [
                    {
                        "epoch": 1,
                        "loss": 0.5,
                        "steps": 2,
                        "images": 4,
                        "elapsed_seconds": 1.0,
                        "seconds_per_step": 0.5,
                        "peak_cuda_memory_allocated_bytes": 123_456,
                        "device": "cuda",
                    }
                ],
            )

            with history_path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["peak_cuda_memory_allocated_bytes"], "123456")


if __name__ == "__main__":
    unittest.main()
