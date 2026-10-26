from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from ssl_hdc_pilot.scripts.adapt_dinov3 import (
    load_module_state,
    load_training_checkpoint,
    save_training_checkpoint,
)


class AdaptationCheckpointTests(unittest.TestCase):
    def test_npz_checkpoint_roundtrip_restores_training_state(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            output_dir = Path(tmp)
            student_encoder = torch.nn.Linear(2, 2)
            teacher_encoder = copy.deepcopy(student_encoder)
            student_head = torch.nn.Linear(2, 3)
            teacher_head = copy.deepcopy(student_head)
            optimizer = torch.optim.AdamW(
                list(student_encoder.parameters()) + list(student_head.parameters()),
                lr=1e-5,
            )
            loss = student_head(student_encoder(torch.ones(2, 2))).square().mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                teacher_encoder.weight.add_(5.0)
            expected_weight = student_encoder.weight.detach().clone()
            expected_teacher = teacher_encoder.weight.detach().clone()
            expected_student_head = copy.deepcopy(student_head.state_dict())
            expected_teacher_head = copy.deepcopy(teacher_head.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            center = torch.arange(3, dtype=torch.float32).reshape(1, 3)

            checkpoint = save_training_checkpoint(
                output_dir=output_dir,
                epoch=1,
                student_encoder=student_encoder,
                teacher_encoder=teacher_encoder,
                student_head=student_head,
                teacher_head=teacher_head,
                optimizer=optimizer,
                scaler_state={"scale": 65536.0, "growth_tracker": 1},
                center=center,
                metadata={"loss": 0.25},
            )
            self.assertEqual(checkpoint.name, "checkpoint_epoch1")
            self.assertTrue((checkpoint / "encoder.npz").is_file())
            self.assertTrue((checkpoint / "training_state.npz").is_file())
            exported_encoder = torch.nn.Linear(2, 2)
            load_module_state(
                checkpoint / "encoder.npz",
                exported_encoder,
                torch.device("cpu"),
            )
            torch.testing.assert_close(exported_encoder.weight, expected_teacher)

            with torch.no_grad():
                student_encoder.weight.zero_()
                teacher_encoder.weight.zero_()
                for parameter in student_head.parameters():
                    parameter.zero_()
                for parameter in teacher_head.parameters():
                    parameter.zero_()
            optimizer.state.clear()
            restored = load_training_checkpoint(
                checkpoint_dir=checkpoint,
                student_encoder=student_encoder,
                teacher_encoder=teacher_encoder,
                student_head=student_head,
                teacher_head=teacher_head,
                optimizer=optimizer,
                device=torch.device("cpu"),
            )

            torch.testing.assert_close(student_encoder.weight, expected_weight)
            torch.testing.assert_close(teacher_encoder.weight, expected_teacher)
            torch.testing.assert_close(student_head.state_dict(), expected_student_head)
            torch.testing.assert_close(teacher_head.state_dict(), expected_teacher_head)
            torch.testing.assert_close(optimizer.state_dict(), expected_optimizer)
            torch.testing.assert_close(restored.center, center)
            self.assertEqual(restored.scaler_state["scale"], 65536.0)
            self.assertEqual(restored.metadata["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
