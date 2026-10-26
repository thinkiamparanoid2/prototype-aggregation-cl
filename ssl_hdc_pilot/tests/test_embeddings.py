from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from ssl_hdc_pilot.scripts.extract_embeddings import (
    HEADLINE_MODEL_ID,
    LADDER_MODEL_SPECS,
    ImageRecord,
    extract_records,
    validate_request,
    write_embedding_artifacts,
)
from ssl_hdc_pilot.scripts.checkpoint_encoder import load_checkpoint_export


class _FakeProcessor:
    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": torch.ones(len(images), 3, 4, 4)}

    def to_dict(self):
        return {"size": {"height": 224, "width": 224}, "do_normalize": True}


class _FakeModel(torch.nn.Module):
    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        hidden = torch.arange(batch * 3 * 4, dtype=torch.float16).reshape(batch, 3, 4)
        return SimpleNamespace(last_hidden_state=hidden)


class _CheckpointModel(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.weight = torch.nn.Parameter(torch.zeros(1))


def _write_checkpoint_export(
    root: Path,
    *,
    run_model_id: str = HEADLINE_MODEL_ID,
    metadata_model_id: str = HEADLINE_MODEL_ID,
) -> Path:
    run_root = root / "arm_d_vitb16"
    checkpoint = run_root / "checkpoint_epoch3"
    checkpoint.mkdir(parents=True)
    (run_root / "run_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "training_config": {
                    "model_id": run_model_id,
                    "epochs": 5,
                    "smoke": False,
                },
            }
        ),
        encoding="utf-8",
    )
    np.savez(checkpoint / "encoder.npz", weight=np.zeros(1, dtype=np.float32))
    np.savez(checkpoint / "training_state.npz", marker=np.ones(1, dtype=np.float32))
    (checkpoint / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "complete": True,
                "epoch": 3,
                "model_id": metadata_model_id,
                "exported_encoder": "ema_teacher",
                "labels_used": False,
                "trusted_pixels_used": False,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


class EmbeddingContractTests(unittest.TestCase):
    def test_extracts_cls_vectors_as_float32_in_input_order(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            records = []
            for image_id in ("a", "b", "c"):
                path = root / f"{image_id}.jpg"
                Image.new("RGB", (8, 8), color=(1, 2, 3)).save(path)
                records.append(ImageRecord(image_id=image_id, image_path=path))

            image_ids, embeddings = extract_records(
                records=records,
                model=_FakeModel(),
                processor=_FakeProcessor(),
                device=torch.device("cpu"),
                batch_size=2,
            )

            self.assertEqual(image_ids, ["a", "b", "c"])
            self.assertEqual(embeddings.shape, (3, 4))
            self.assertEqual(embeddings.dtype, np.float32)
            np.testing.assert_array_equal(embeddings[0], np.array([0, 1, 2, 3], dtype=np.float32))

    def test_npz_writer_rejects_non_float32_embeddings(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "float32"):
                write_embedding_artifacts(
                    image_ids=["1"],
                    embeddings=np.zeros((1, 384), dtype=np.float64),
                    output_npz=root / "bad.npz",
                    metadata_json=root / "bad.json",
                    metadata={"model_id": HEADLINE_MODEL_ID},
                )

    def test_budget_and_model_size_guards(self) -> None:
        self.assertEqual(
            set(LADDER_MODEL_SPECS),
            {
                "facebook/dinov3-vits16-pretrain-lvd1689m",
                "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "facebook/dinov3-vit7b16-pretrain-lvd1689m",
            },
        )
        for model_id in LADDER_MODEL_SPECS:
            validate_request(model_id=model_id, smoke=False, max_images=None)
            validate_request(model_id=model_id, smoke=True, max_images=50)
        with self.assertRaisesRegex(ValueError, "--max-images is smoke-only"):
            validate_request(model_id=HEADLINE_MODEL_ID, smoke=False, max_images=50)
        with self.assertRaisesRegex(ValueError, "at most 100"):
            validate_request(model_id=HEADLINE_MODEL_ID, smoke=True, max_images=101)
        with self.assertRaisesRegex(ValueError, "encoder-size ladder"):
            validate_request(model_id="facebook/not-authorized", smoke=False, max_images=None)

    def test_writer_binds_model_id_to_expected_embedding_dimension(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            for model_id, spec in LADDER_MODEL_SPECS.items():
                payload = write_embedding_artifacts(
                    image_ids=["1"],
                    embeddings=np.zeros((1, spec.embedding_dim), dtype=np.float32),
                    output_npz=root / f"{spec.rung}.npz",
                    metadata_json=root / f"{spec.rung}.json",
                    metadata={"model_id": model_id},
                )
                self.assertEqual(payload["model_id"], model_id)
                self.assertEqual(payload["embedding_dim"], spec.embedding_dim)
            with self.assertRaisesRegex(ValueError, "expected 768"):
                write_embedding_artifacts(
                    image_ids=["1"],
                    embeddings=np.zeros((1, 384), dtype=np.float32),
                    output_npz=root / "wrong.npz",
                    metadata_json=root / "wrong.json",
                    metadata={"model_id": HEADLINE_MODEL_ID},
                )

    def test_checkpoint_export_rejects_model_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            checkpoint = _write_checkpoint_export(
                root,
                metadata_model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
            )
            with self.assertRaisesRegex(RuntimeError, "model id"):
                load_checkpoint_export(
                    checkpoint,
                    _CheckpointModel(768),
                    torch.device("cpu"),
                    results_root=root,
                )

    def test_checkpoint_export_rejects_recorded_dimension_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            checkpoint = _write_checkpoint_export(root)
            with self.assertRaisesRegex(RuntimeError, "expected 768"):
                load_checkpoint_export(
                    checkpoint,
                    _CheckpointModel(384),
                    torch.device("cpu"),
                    results_root=root,
                )

    def test_checkpoint_export_returns_auditable_provenance(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            checkpoint = _write_checkpoint_export(root)
            provenance = load_checkpoint_export(
                checkpoint,
                _CheckpointModel(768),
                torch.device("cpu"),
                results_root=root,
            )
            self.assertEqual(provenance.rung, "vitb16")
            self.assertEqual(provenance.epoch, 3)
            self.assertEqual(len(provenance.run_config_sha256), 64)
            self.assertEqual(provenance.exported_encoder, "ema_teacher")


if __name__ == "__main__":
    unittest.main()
