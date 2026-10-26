"""Validate and strict-load Arm-D EMA-teacher encoder exports."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ssl_hdc_pilot.scripts.encoder_models import EncoderModelSpec, get_model_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "ssl_hdc_pilot" / "results"
EXPORTED_ENCODER_ROLE = "ema_teacher"


@dataclass(frozen=True)
class CheckpointProvenance:
    checkpoint_dir: Path
    model_id: str
    rung: str
    embedding_dim: int
    epoch: int
    run_config_sha256: str
    exported_encoder: str

    def as_metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": str(self.checkpoint_dir.resolve()),
            "model_id": self.model_id,
            "rung": self.rung,
            "embedding_dim": self.embedding_dim,
            "epoch": self.epoch,
            "run_config_sha256": self.run_config_sha256,
            "exported_encoder": self.exported_encoder,
        }


def load_module_state(path: Path, module: torch.nn.Module, device: torch.device) -> None:
    with np.load(path, allow_pickle=False) as archive:
        state = {
            name: torch.from_numpy(archive[name].copy()).to(device)
            for name in archive.files
        }
    module.load_state_dict(state, strict=True)


def inspect_checkpoint_export(
    checkpoint_dir: Path,
    *,
    results_root: Path = RESULTS_ROOT,
) -> CheckpointProvenance:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory is missing: {checkpoint_dir}")
    match = re.fullmatch(r"checkpoint_epoch([0-9]+)", checkpoint_dir.name)
    if match is None:
        raise RuntimeError("checkpoint directory must be named checkpoint_epochN")
    epoch = int(match.group(1))
    run_root = checkpoint_dir.parent
    config_path = run_root / "run_config.json"
    if not config_path.is_file():
        raise RuntimeError("checkpoint run lacks its immutable run_config.json")
    try:
        config_bytes = config_path.read_bytes()
        run_config = json.loads(config_bytes)
        training_config = run_config["training_config"]
        model_id = str(training_config["model_id"])
        model_spec = get_model_spec(model_id)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("checkpoint run config lacks a valid model id") from error
    expected_root = (Path(results_root) / model_spec.arm_d_directory).resolve()
    if run_root.resolve() != expected_root:
        raise RuntimeError(f"{model_id} checkpoint run must use directory {expected_root}")
    if training_config.get("smoke") is not False:
        raise RuntimeError("official extraction refuses smoke checkpoints")
    metadata_path = checkpoint_dir / "checkpoint_metadata.json"
    encoder_path = checkpoint_dir / "encoder.npz"
    if not metadata_path.is_file() or not encoder_path.is_file():
        raise RuntimeError("checkpoint lacks complete EMA-teacher export files")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("checkpoint metadata is invalid") from error
    if metadata.get("complete") is not True or metadata.get("epoch") != epoch:
        raise RuntimeError("checkpoint metadata does not match its epoch directory")
    if metadata.get("model_id") != model_id:
        raise RuntimeError("checkpoint model id does not match run config")
    if metadata.get("exported_encoder") != EXPORTED_ENCODER_ROLE:
        raise RuntimeError("checkpoint does not export the EMA teacher")
    if metadata.get("labels_used") is not False or metadata.get("trusted_pixels_used") is not False:
        raise RuntimeError("checkpoint lacks label-free training provenance")
    if epoch == 0 and metadata.get("frozen_baseline") is not True:
        raise RuntimeError("checkpoint_epoch0 is not marked as the frozen baseline")
    if epoch > 0 and not (checkpoint_dir / "training_state.npz").is_file():
        raise RuntimeError("adapted checkpoint lacks complete training state")
    return CheckpointProvenance(
        checkpoint_dir=checkpoint_dir,
        model_id=model_spec.model_id,
        rung=model_spec.rung,
        embedding_dim=model_spec.embedding_dim,
        epoch=epoch,
        run_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        exported_encoder=EXPORTED_ENCODER_ROLE,
    )


def _validate_model_dimension(model: torch.nn.Module, spec: EncoderModelSpec) -> None:
    actual = getattr(getattr(model, "config", None), "hidden_size", None)
    if actual != spec.embedding_dim:
        raise RuntimeError(
            f"checkpoint model {spec.model_id} expected {spec.embedding_dim}-d encoder, got {actual}"
        )


def load_checkpoint_export(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    *,
    results_root: Path = RESULTS_ROOT,
) -> CheckpointProvenance:
    provenance = inspect_checkpoint_export(checkpoint_dir, results_root=results_root)
    spec = get_model_spec(provenance.model_id)
    _validate_model_dimension(model, spec)
    load_module_state(Path(checkpoint_dir) / "encoder.npz", model, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("checkpoint encoder frozen guard failed")
    return provenance


def load_checkpoint_encoder(
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, CheckpointProvenance]:
    provenance = inspect_checkpoint_export(checkpoint_dir)
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        provenance.model_id, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        provenance.model_id, local_files_only=True
    )
    provenance = load_checkpoint_export(checkpoint_dir, model, device)
    return model, processor, provenance
