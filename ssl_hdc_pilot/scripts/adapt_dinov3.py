"""Label-free DINOv3 domain adaptation on the sanctioned unlabeled manifest."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ssl_hdc_pilot.scripts.checkpoint_encoder import (
    EXPORTED_ENCODER_ROLE,
    load_module_state,
)
from ssl_hdc_pilot.scripts.encoder_models import (
    HEADLINE_MODEL_ID,
    LADDER_MODEL_SPECS,
    get_model_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "ssl_hdc_pilot" / "results" / "unlabeled_manifest.csv"
RESULTS_ROOT = REPO_ROOT / "ssl_hdc_pilot" / "results"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "arm_d_vitb16"
TRAIN_ROOT = REPO_ROOT / "Data" / "train"
TRUSTED_ROOT = REPO_ROOT / "Data" / "test"
EXPECTED_MANIFEST_COUNT = 51_906
# Opt-in replication mode (--replication-dataset). Without the flag, every
# check below still uses EXPECTED_MANIFEST_COUNT/TRAIN_ROOT/TRUSTED_ROOT
# exactly as before; these constants only relax the original-study-specific
# checks when a replication dataset is explicitly named.
REPLICATION_DATASETS = ("PlantDoc", "PlantWild", "PlantVillage")
REPLICATION_DATA_ROOT = REPO_ROOT / "ssl_hdc_pilot" / "replication" / "data"
REPLICATION_ADAPTED_ROOT = REPO_ROOT / "ssl_hdc_pilot" / "replication" / "adapted"
PROJECTION_HIDDEN_DIM = 1_024
PROJECTION_BOTTLENECK_DIM = 256
PROJECTION_OUTPUT_DIM = 4_096
STUDENT_TEMPERATURE = 0.1
TEACHER_TEMPERATURE = 0.04
CENTER_MOMENTUM = 0.9
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class AdaptationConfig:
    model_id: str = HEADLINE_MODEL_ID
    smoke: bool = False
    max_images: int | None = None
    max_steps: int | None = None
    epochs: int = 5
    learning_rate: float = 1e-5
    ema_momentum: float = 0.996
    global_crops: int = 2
    local_crops: int = 4
    amp: bool = True
    gradient_checkpointing: bool = True
    batch_size: int = 32
    num_workers: int = 0
    weight_decay: float = 0.04
    seed: int = 20_260_716
    device: str = "cuda"


@dataclass(frozen=True)
class ResumePlan:
    output_dir: Path
    last_complete_epoch: int


@dataclass(frozen=True)
class TrainingRestore:
    center: torch.Tensor
    scaler_state: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ManifestRecord:
    image_id: str
    image_path: Path


class MultiCropTransform:
    """Two 224px global views and four 96px local views."""

    def __init__(self) -> None:
        color = transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
        )
        common = [
            transforms.RandomHorizontalFlip(),
            color,
            transforms.RandomGrayscale(p=0.2),
        ]
        normalize = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        self.global_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
                *common,
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
                *normalize,
            ]
        )
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(96, scale=(0.05, 0.4)),
                *common,
                transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
                *normalize,
            ]
        )

    def __call__(self, image: Image.Image) -> list[torch.Tensor]:
        return [self.global_transform(image) for _ in range(2)] + [
            self.local_transform(image) for _ in range(4)
        ]


class ManifestDataset(Dataset):
    def __init__(self, records: list[ManifestRecord]) -> None:
        self.records = records
        self.transform = MultiCropTransform()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[list[torch.Tensor], str]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            views = self.transform(image.convert("RGB"))
        return views, record.image_id


class DINOProjectionHead(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, PROJECTION_HIDDEN_DIM, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(PROJECTION_HIDDEN_DIM, PROJECTION_BOTTLENECK_DIM, bias=False),
        )
        self.last_layer = torch.nn.Linear(
            PROJECTION_BOTTLENECK_DIM, PROJECTION_OUTPUT_DIM, bias=False
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.last_layer(F.normalize(self.mlp(features), dim=-1))


def validate_config(config: AdaptationConfig) -> None:
    spec = LADDER_MODEL_SPECS.get(config.model_id)
    if spec is None:
        raise ValueError("Arm-D model id is not in the preregistered encoder-size ladder")
    if not spec.adaptable:
        raise ValueError(
            f"{config.model_id} is a frozen-only ceiling rung; it cannot be Arm-D adapted"
        )
    if not 0.0 < config.learning_rate <= 1e-4:
        raise ValueError("Arm-D low-LR ceiling is 1e-4")
    if not 0.99 <= config.ema_momentum < 1.0:
        raise ValueError("EMA momentum must be in [0.99, 1.0)")
    if config.global_crops != 2 or config.local_crops != 4:
        raise ValueError("multi-crop requires exactly 2 global and 4 local crops")
    if not config.amp:
        raise ValueError("AMP is mandatory for Arm-D")
    if not config.gradient_checkpointing:
        raise ValueError("gradient checkpointing is mandatory for Arm-D")
    if not 1 <= config.epochs <= 10:
        raise ValueError("Arm-D is a short schedule of 1 to 10 epochs")
    if not 1 <= config.batch_size <= 64:
        raise ValueError("batch_size must be in [1, 64]")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if not 0.0 <= config.weight_decay <= 0.1:
        raise ValueError("weight_decay must be in [0, 0.1]")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if config.smoke:
        if not 1 <= config.epochs <= 2:
            raise ValueError("Arm-D smoke may run only 1 or 2 epochs")
        if config.max_images is None or not 1 <= config.max_images <= 512:
            raise ValueError("Arm-D smoke may use at most 512 images")
        if config.max_steps is None or not 1 <= config.max_steps <= 10:
            raise ValueError("Arm-D smoke may run at most 10 steps")
        if config.epochs * config.max_steps > 10:
            raise ValueError("Arm-D smoke may run at most 10 total steps")
    elif config.max_images is not None or config.max_steps is not None:
        raise ValueError("max-images/max-steps are smoke-only controls")
    if not config.smoke:
        # num_workers is deliberately NOT in this lock: it is a DataLoader
        # hardware-throughput knob (how many OS processes decode/augment
        # images), not part of the scientific recipe. It does not change
        # amp/gradient_checkpointing/batch_size/crops/lr/momentum/epochs/
        # device, and every optimization step still consumes the exact same
        # per-sample augmented views; it only changes which OS process runs
        # each __getitem__ call. The one caveat (recipe-equivalent, not
        # bitwise-identical): torchvision's random transforms draw from
        # process-local RNG state, so num_workers > 0 spreads augmentation
        # draws across worker-process RNG streams instead of the single
        # main-process stream used at num_workers=0 -- the *distribution* of
        # crops/flips/jitter is unchanged, but the exact sequence is not
        # reproduced bit-for-bit relative to a num_workers=0 run with the
        # same seed. It is still guarded to be non-negative above.
        locked_recipe = {
            "epochs": 5,
            "learning_rate": 1e-5,
            "ema_momentum": 0.996,
            "global_crops": 2,
            "local_crops": 4,
            "amp": True,
            "gradient_checkpointing": True,
            "batch_size": 32,
            "device": "cuda",
        }
        actual = {
            field: getattr(config, field)
            for field in locked_recipe
        }
        if actual != locked_recipe:
            raise ValueError(
                "official Arm-D runs must use the preregistered ladder recipe"
            )


def validate_smoke_stop_after_epoch(
    config: AdaptationConfig, stop_after_epoch: int | None
) -> None:
    if stop_after_epoch is None:
        return
    if not config.smoke:
        raise ValueError("--smoke-stop-after-epoch is a smoke-only control")
    if not 1 <= stop_after_epoch < config.epochs:
        raise ValueError("smoke stop must be at least 1 and before the final epoch")


def _replication_split_ids(dataset: str, split_name: str) -> set[str]:
    path = REPLICATION_DATA_ROOT / dataset / "splits" / f"{split_name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"replication {split_name} split is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return {row["image"].strip() for row in reader}


def load_sanctioned_manifest(
    manifest: Path,
    *,
    smoke: bool,
    max_images: int | None,
    replication_dataset: str | None = None,
) -> list[ManifestRecord]:
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        forbidden = {"label", "crop_disease_stage"} & fields
        if forbidden:
            raise RuntimeError(f"Arm-D manifest must be label-free; found {sorted(forbidden)}")
        required = {"image_id", "image_path", "source_pool"}
        if not required <= fields:
            raise RuntimeError(f"Arm-D manifest is missing columns: {sorted(required - fields)}")
        rows = list(reader)

    if replication_dataset is None:
        expected_count = EXPECTED_MANIFEST_COUNT
        expected_source_pool = "Data/train"
        train_root = TRAIN_ROOT.resolve()
        train_ids: set[str] | None = None
        val_ids: set[str] | None = None
    else:
        if replication_dataset not in REPLICATION_DATASETS:
            raise ValueError(f"unknown --replication-dataset: {replication_dataset!r}")
        train_ids = _replication_split_ids(replication_dataset, "train")
        val_ids = _replication_split_ids(replication_dataset, "val")
        expected_count = len(train_ids)
        expected_source_pool = f"ssl_hdc_pilot/replication/data/{replication_dataset}/train"
        train_root = (REPLICATION_DATA_ROOT / replication_dataset).resolve()
    if len(rows) != expected_count:
        raise RuntimeError(
            f"sanctioned manifest must contain {expected_count} rows, got {len(rows)}"
        )

    trusted_ids = {path.name for path in TRUSTED_ROOT.iterdir() if path.is_dir()}
    records: list[ManifestRecord] = []
    seen: set[str] = set()
    for row in rows:
        image_id = row["image_id"].strip()
        if not image_id or image_id in seen:
            raise RuntimeError(f"invalid or duplicate manifest image id: {image_id!r}")
        if image_id in trusted_ids:
            raise RuntimeError(f"trusted image id found in Arm-D manifest: {image_id}")
        if row["source_pool"].strip() != expected_source_pool:
            raise RuntimeError(f"non-sanctioned source_pool for {image_id}")
        image_path = Path(row["image_path"].strip())
        if not image_path.is_absolute():
            image_path = REPO_ROOT / image_path
        resolved = image_path.resolve()
        if not resolved.is_relative_to(train_root):
            raise RuntimeError(f"Arm-D image path escapes {train_root}: {resolved}")
        if replication_dataset is None:
            if resolved.name != f"{image_id}.jpg" or not resolved.is_file():
                raise FileNotFoundError(f"invalid Arm-D image path for {image_id}: {resolved}")
        else:
            if not resolved.is_file():
                raise FileNotFoundError(f"invalid Arm-D image path for {image_id}: {resolved}")
            if image_id not in train_ids:
                raise RuntimeError(
                    f"replication image id is not in {replication_dataset}'s train split: {image_id}"
                )
            if image_id in val_ids:
                raise RuntimeError(
                    f"replication image id belongs to {replication_dataset}'s val split: {image_id}"
                )
        seen.add(image_id)
        records.append(ManifestRecord(image_id=image_id, image_path=resolved))
    if smoke:
        records = records[: int(max_images or 0)]
    return records


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _encoder_logits(
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    views: list[torch.Tensor],
) -> list[torch.Tensor]:
    batch_size = views[0].shape[0]
    logits: list[torch.Tensor] = []
    groups = [views[:2]]
    if len(views) > 2:
        groups.append(views[2:])
    for size_group in groups:
        pixels = torch.cat(size_group, dim=0)
        output = encoder(pixel_values=pixels, interpolate_pos_encoding=True)
        projected = head(output.last_hidden_state[:, 0, :])
        logits.extend(projected.split(batch_size, dim=0))
    return logits


def dino_self_distillation_loss(
    student_logits: list[torch.Tensor],
    teacher_logits: list[torch.Tensor],
    center: torch.Tensor,
) -> torch.Tensor:
    student_log_probs = [
        F.log_softmax(logits.float() / STUDENT_TEMPERATURE, dim=-1)
        for logits in student_logits
    ]
    teacher_probs = [
        F.softmax((logits.float() - center) / TEACHER_TEMPERATURE, dim=-1).detach()
        for logits in teacher_logits
    ]
    losses = []
    for teacher_index, teacher_probability in enumerate(teacher_probs):
        for student_index, student_log_probability in enumerate(student_log_probs):
            if student_index == teacher_index:
                continue
            losses.append(
                torch.sum(-teacher_probability * student_log_probability, dim=-1).mean()
            )
    return torch.stack(losses).mean()


def require_finite_loss(loss: torch.Tensor, *, epoch: int, step: int) -> None:
    if not bool(torch.isfinite(loss).all().item()):
        raise FloatingPointError(
            f"Arm-D non-finite loss at epoch={epoch} step={step}; aborting before update"
        )


@torch.no_grad()
def update_teacher_ema(
    student_encoder: torch.nn.Module,
    teacher_encoder: torch.nn.Module,
    student_head: torch.nn.Module,
    teacher_head: torch.nn.Module,
    momentum: float,
) -> None:
    for student, teacher in zip(student_encoder.parameters(), teacher_encoder.parameters()):
        teacher.mul_(momentum).add_(student.detach(), alpha=1.0 - momentum)
    for student, teacher in zip(student_head.parameters(), teacher_head.parameters()):
        teacher.mul_(momentum).add_(student.detach(), alpha=1.0 - momentum)
    for student, teacher in zip(student_encoder.buffers(), teacher_encoder.buffers()):
        teacher.copy_(student)
    for student, teacher in zip(student_head.buffers(), teacher_head.buffers()):
        teacher.copy_(student)


@torch.no_grad()
def update_center(center: torch.Tensor, teacher_logits: list[torch.Tensor]) -> torch.Tensor:
    batch_center = torch.cat(teacher_logits, dim=0).float().mean(dim=0, keepdim=True)
    return center * CENTER_MOMENTUM + batch_center * (1.0 - CENTER_MOMENTUM)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resume_config_payload(
    config: AdaptationConfig,
    manifest: Path,
    *,
    replication_dataset: str | None = None,
) -> dict[str, Any]:
    if not manifest.is_file():
        raise FileNotFoundError(f"sanctioned manifest is missing: {manifest}")
    training_config = asdict(config)
    if replication_dataset is not None:
        # Added only when set, so a payload built with replication_dataset=None
        # is byte-for-byte identical to the pre-existing schema (no new key),
        # keeping old run_config.json files on disk resumable unchanged.
        training_config["replication_dataset"] = replication_dataset
    return {
        "schema_version": 1,
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "training_config": training_config,
    }


def _valid_npz(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return bool(archive.files)
    except (OSError, ValueError):
        return False


def _state_keys_sha256(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def _checkpoint_is_complete(
    path: Path,
    epoch: int,
    expected_model_id: str,
) -> bool:
    metadata_path = path / "checkpoint_metadata.json"
    encoder_path = path / "encoder.npz"
    if not metadata_path.is_file() or not _valid_npz(encoder_path):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if metadata.get("complete") is not True or metadata.get("epoch") != epoch:
        return False
    if metadata.get("exported_encoder") != EXPORTED_ENCODER_ROLE:
        return False
    required_provenance = {
        "model_id": expected_model_id,
        "labels_used": False,
        "trusted_pixels_used": False,
    }
    if any(metadata.get(key) != value for key, value in required_provenance.items()):
        return False
    with np.load(encoder_path, allow_pickle=False) as archive:
        encoder_keys = list(archive.files)
    if metadata.get("encoder_state_key_count") != len(encoder_keys):
        return False
    if metadata.get("encoder_state_keys_sha256") != _state_keys_sha256(encoder_keys):
        return False
    if epoch == 0 and metadata.get("frozen_baseline") is not True:
        return False
    return epoch == 0 or _valid_npz(path / "training_state.npz")


def _last_complete_epoch(output_dir: Path, expected_model_id: str) -> int:
    complete: list[int] = []
    for path in output_dir.glob("checkpoint_epoch*"):
        suffix = path.name.removeprefix("checkpoint_epoch")
        if not path.is_dir() or not suffix.isdigit():
            continue
        epoch = int(suffix)
        if _checkpoint_is_complete(path, epoch, expected_model_id):
            complete.append(epoch)
    complete.sort()
    if complete and complete != list(range(complete[-1] + 1)):
        raise RuntimeError(
            f"Arm-D checkpoint cadence gap; found complete epochs {complete}"
        )
    return complete[-1] if complete else -1


def validate_output_dir(output_dir: Path, payload: dict[str, Any]) -> str:
    try:
        training_config = payload["training_config"]
        model_id = str(training_config["model_id"])
        smoke = training_config["smoke"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Arm-D run config lacks model or smoke identity") from error
    model_spec = get_model_spec(model_id)
    if output_dir.name != model_spec.arm_d_directory:
        raise RuntimeError(
            f"{model_id} must use output directory {model_spec.arm_d_directory}"
        )
    replication_dataset = training_config.get("replication_dataset")
    if smoke is not False and smoke is not True:
        raise RuntimeError("Arm-D run config smoke identity must be boolean")
    if replication_dataset is not None:
        if replication_dataset not in REPLICATION_DATASETS:
            raise RuntimeError("invalid replication_dataset in run config")
        expected = (REPLICATION_ADAPTED_ROOT / replication_dataset / model_spec.arm_d_directory).resolve()
        if output_dir.resolve() != expected:
            raise RuntimeError(
                f"{model_id} replication run for {replication_dataset} must use {expected}"
            )
    elif smoke is False:
        expected = (RESULTS_ROOT / model_spec.arm_d_directory).resolve()
        if output_dir.resolve() != expected:
            raise RuntimeError(
                f"official {model_spec.rung} run must use {expected}"
            )
    return model_id


def prepare_run(output_dir: Path, payload: dict[str, Any]) -> ResumePlan:
    model_id = validate_output_dir(output_dir, payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("Arm-D resume config mismatch; use a new output directory")
    else:
        unexpected = [path for path in output_dir.iterdir() if path.name != "run_config.json"]
        if unexpected:
            raise RuntimeError("Arm-D output directory is non-empty but has no run config")
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return ResumePlan(
        output_dir=output_dir,
        last_complete_epoch=_last_complete_epoch(output_dir, model_id),
    )


def _save_module_state(path: Path, module: torch.nn.Module) -> dict[str, Any]:
    arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in module.state_dict().items()
    }
    np.savez(path, **arrays)
    keys = list(arrays)
    return {
        "encoder_state_key_count": len(keys),
        "encoder_state_keys_sha256": _state_keys_sha256(keys),
    }


def _optimizer_payload(optimizer: torch.optim.Optimizer) -> dict[str, np.ndarray]:
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    parameter_index = {id(parameter): index for index, parameter in enumerate(parameters)}
    groups = []
    for group in optimizer.param_groups:
        settings = {key: value for key, value in group.items() if key != "params"}
        settings["parameter_indices"] = [parameter_index[id(parameter)] for parameter in group["params"]]
        groups.append(settings)
    arrays: dict[str, np.ndarray] = {
        "optimizer_groups_json": np.asarray(json.dumps(groups)),
    }
    for parameter, state in optimizer.state.items():
        index = parameter_index[id(parameter)]
        for name, value in state.items():
            if torch.is_tensor(value):
                arrays[f"optimizer::{index}::{name}"] = value.detach().cpu().numpy()
            else:
                arrays[f"optimizer::{index}::{name}"] = np.asarray(value)
    return arrays


def _restore_optimizer(
    archive: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    parameter_index = {id(parameter): index for index, parameter in enumerate(parameters)}
    saved_groups = json.loads(str(archive["optimizer_groups_json"].item()))
    if len(saved_groups) != len(optimizer.param_groups):
        raise RuntimeError("optimizer group count mismatch on resume")
    optimizer.state.clear()
    for saved, current in zip(saved_groups, optimizer.param_groups):
        indices = saved.pop("parameter_indices")
        current_indices = [parameter_index[id(parameter)] for parameter in current["params"]]
        if indices != current_indices:
            raise RuntimeError("optimizer parameter order mismatch on resume")
        for key, value in saved.items():
            current[key] = tuple(value) if key == "betas" else value
    for key in archive.files:
        if not key.startswith("optimizer::"):
            continue
        _, raw_index, state_name = key.split("::", 2)
        parameter = parameters[int(raw_index)]
        optimizer.state.setdefault(parameter, {})[state_name] = torch.from_numpy(
            archive[key].copy()
        ).to(device)


def save_training_checkpoint(
    *,
    output_dir: Path,
    epoch: int,
    student_encoder: torch.nn.Module,
    teacher_encoder: torch.nn.Module,
    student_head: torch.nn.Module,
    teacher_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler_state: dict[str, Any],
    center: torch.Tensor,
    metadata: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / f"checkpoint_epoch{epoch}"
    if checkpoint_dir.exists():
        raise RuntimeError(f"refusing to overwrite complete checkpoint: {checkpoint_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=f".checkpoint_epoch{epoch}_", dir=output_dir))
    try:
        encoder_schema = _save_module_state(temporary / "encoder.npz", teacher_encoder)
        arrays: dict[str, np.ndarray] = {
            "center": center.detach().cpu().numpy(),
            "scaler_state_json": np.asarray(json.dumps(scaler_state)),
        }
        for prefix, module in (
            ("student_encoder", student_encoder),
            ("student_head", student_head),
            ("teacher_head", teacher_head),
        ):
            arrays.update(
                {
                    f"{prefix}::{name}": tensor.detach().cpu().numpy()
                    for name, tensor in module.state_dict().items()
                }
            )
        arrays.update(_optimizer_payload(optimizer))
        np.savez(temporary / "training_state.npz", **arrays)
        checkpoint_metadata = {
            **metadata,
            **encoder_schema,
            "exported_encoder": EXPORTED_ENCODER_ROLE,
            "epoch": epoch,
            "complete": True,
        }
        (temporary / "checkpoint_metadata.json").write_text(
            json.dumps(checkpoint_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(checkpoint_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return checkpoint_dir


def load_training_checkpoint(
    *,
    checkpoint_dir: Path,
    student_encoder: torch.nn.Module,
    teacher_encoder: torch.nn.Module,
    student_head: torch.nn.Module,
    teacher_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> TrainingRestore:
    metadata = json.loads(
        (checkpoint_dir / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("complete") is not True:
        raise RuntimeError(f"checkpoint is incomplete: {checkpoint_dir}")
    load_module_state(checkpoint_dir / "encoder.npz", teacher_encoder, device)
    with np.load(checkpoint_dir / "training_state.npz", allow_pickle=False) as archive:
        for prefix, module in (
            ("student_encoder", student_encoder),
            ("student_head", student_head),
            ("teacher_head", teacher_head),
        ):
            state = {
                key.removeprefix(f"{prefix}::"): torch.from_numpy(archive[key].copy()).to(device)
                for key in archive.files
                if key.startswith(f"{prefix}::")
            }
            module.load_state_dict(state, strict=True)
        _restore_optimizer(archive, optimizer, device)
        center = torch.from_numpy(archive["center"].copy()).to(device)
        scaler_state = json.loads(str(archive["scaler_state_json"].item()))
    return TrainingRestore(center=center, scaler_state=scaler_state, metadata=metadata)


def save_baseline_checkpoint(
    output_dir: Path,
    encoder: torch.nn.Module,
    *,
    metadata: dict[str, Any],
) -> Path:
    model_id = str(metadata.get("model_id", ""))
    get_model_spec(model_id)
    checkpoint_dir = output_dir / "checkpoint_epoch0"
    if checkpoint_dir.exists():
        if _checkpoint_is_complete(checkpoint_dir, 0, model_id):
            return checkpoint_dir
        raise RuntimeError(f"frozen baseline checkpoint is incomplete: {checkpoint_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint_epoch0_", dir=output_dir))
    try:
        encoder_schema = _save_module_state(temporary / "encoder.npz", encoder)
        payload = {
            **metadata,
            **encoder_schema,
            "exported_encoder": EXPORTED_ENCODER_ROLE,
            "epoch": 0,
            "complete": True,
            "frozen_baseline": True,
        }
        (temporary / "checkpoint_metadata.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(checkpoint_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return checkpoint_dir


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_student_teacher(
    model_id: str,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, DINOProjectionHead, DINOProjectionHead]:
    from transformers import AutoModel

    student_encoder = AutoModel.from_pretrained(model_id, local_files_only=True)
    student_encoder.gradient_checkpointing_enable()
    teacher_encoder = copy.deepcopy(student_encoder)
    teacher_encoder.gradient_checkpointing_disable()
    hidden_size = int(student_encoder.config.hidden_size)
    student_head = DINOProjectionHead(hidden_size)
    teacher_head = copy.deepcopy(student_head)
    for module in (teacher_encoder, teacher_head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    student_encoder.to(device)
    teacher_encoder.to(device)
    student_head.to(device)
    teacher_head.to(device)
    return student_encoder, teacher_encoder, student_head, teacher_head


def make_loader(
    records: list[ManifestRecord],
    config: AdaptationConfig,
    epoch: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed + epoch)
    return DataLoader(
        ManifestDataset(records),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def train_one_epoch(
    *,
    records: list[ManifestRecord],
    config: AdaptationConfig,
    epoch: int,
    student_encoder: torch.nn.Module,
    teacher_encoder: torch.nn.Module,
    student_head: DINOProjectionHead,
    teacher_head: DINOProjectionHead,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    center: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    student_encoder.train()
    student_head.train()
    teacher_encoder.eval()
    teacher_head.eval()
    loader = make_loader(records, config, epoch, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    total_loss = 0.0
    steps = 0
    images = 0
    for step, (views, image_ids) in enumerate(loader, start=1):
        if config.smoke and step > int(config.max_steps or 0):
            break
        views = [view.to(device, non_blocking=device.type == "cuda") for view in views]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda" and config.amp,
        ):
            student_logits = _encoder_logits(student_encoder, student_head, views)
            with torch.no_grad():
                teacher_logits = _encoder_logits(teacher_encoder, teacher_head, views[:2])
            loss = dino_self_distillation_loss(student_logits, teacher_logits, center)
        require_finite_loss(loss, epoch=epoch, step=step)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(student_encoder.parameters()) + list(student_head.parameters()), 3.0
        )
        scaler.step(optimizer)
        scaler.update()
        update_teacher_ema(
            student_encoder,
            teacher_encoder,
            student_head,
            teacher_head,
            config.ema_momentum,
        )
        center = update_center(center, teacher_logits)
        total_loss += float(loss.detach().cpu())
        steps += 1
        images += len(image_ids)
        print(
            f"epoch={epoch} step={step} images={images} loss={float(loss.detach().cpu()):.6f}",
            flush=True,
        )
    if steps == 0:
        raise RuntimeError("Arm-D epoch completed zero optimization steps")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_cuda_memory_allocated_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    return center, {
        "epoch": epoch,
        "loss": total_loss / steps,
        "steps": steps,
        "images": images,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / steps,
        "peak_cuda_memory_allocated_bytes": peak_cuda_memory_allocated_bytes,
        "device": str(device),
    }


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "loss",
        "steps",
        "images",
        "elapsed_seconds",
        "seconds_per_step",
        "peak_cuda_memory_allocated_bytes",
        "device",
    ]
    temporary = path.with_name(f"{path.stem}.tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=HEADLINE_MODEL_ID)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20_260_716)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--global-crops", type=int, default=2)
    parser.add_argument("--local-crops", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke-stop-after-epoch", type=int)
    parser.add_argument("--replication-dataset", choices=REPLICATION_DATASETS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AdaptationConfig(
        model_id=args.model_id,
        smoke=args.smoke,
        max_images=args.max_images,
        max_steps=args.max_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        ema_momentum=args.ema_momentum,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        global_crops=args.global_crops,
        local_crops=args.local_crops,
    )
    validate_config(config)
    validate_smoke_stop_after_epoch(config, args.smoke_stop_after_epoch)
    payload = resume_config_payload(
        config, args.manifest, replication_dataset=args.replication_dataset
    )
    plan = prepare_run(args.output_dir, payload)
    print(f"Arm-D resume plan: last_complete_epoch={plan.last_complete_epoch}", flush=True)
    if plan.last_complete_epoch >= config.epochs:
        print("Arm-D run is already complete; nothing to resume.", flush=True)
        return
    if (
        args.smoke_stop_after_epoch is not None
        and plan.last_complete_epoch >= args.smoke_stop_after_epoch
    ):
        print("Requested smoke stop epoch is already complete; nothing to resume.", flush=True)
        return

    device_name = (
        "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    )
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)
    records = load_sanctioned_manifest(
        args.manifest,
        smoke=config.smoke,
        max_images=config.max_images,
        replication_dataset=args.replication_dataset,
    )
    seed_everything(config.seed)
    student_encoder, teacher_encoder, student_head, teacher_head = build_student_teacher(
        config.model_id, device
    )
    optimizer = torch.optim.AdamW(
        list(student_encoder.parameters()) + list(student_head.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.amp)
    center = torch.zeros((1, PROJECTION_OUTPUT_DIM), device=device)
    fingerprint = _payload_fingerprint(payload)
    common_metadata = {
        "model_id": config.model_id,
        "config_fingerprint": fingerprint,
        "manifest_sha256": payload["manifest_sha256"],
        "labels_used": False,
        "trusted_pixels_used": False,
    }

    last_complete_epoch = plan.last_complete_epoch
    process_resumed_from_epoch = last_complete_epoch
    if last_complete_epoch < 0:
        save_baseline_checkpoint(
            plan.output_dir,
            teacher_encoder,
            metadata=common_metadata,
        )
        last_complete_epoch = 0
        process_resumed_from_epoch = 0
        print("Saved frozen baseline checkpoint_epoch0.", flush=True)
    elif last_complete_epoch == 0:
        load_module_state(
            plan.output_dir / "checkpoint_epoch0" / "encoder.npz",
            teacher_encoder,
            device,
        )
        student_encoder.load_state_dict(teacher_encoder.state_dict(), strict=True)
        print("Strict-loaded frozen baseline checkpoint_epoch0.", flush=True)
    elif last_complete_epoch > 0:
        restored = load_training_checkpoint(
            checkpoint_dir=plan.output_dir / f"checkpoint_epoch{last_complete_epoch}",
            student_encoder=student_encoder,
            teacher_encoder=teacher_encoder,
            student_head=student_head,
            teacher_head=teacher_head,
            optimizer=optimizer,
            device=device,
        )
        center = restored.center
        scaler.load_state_dict(restored.scaler_state)
        print(f"Restored checkpoint_epoch{last_complete_epoch}.", flush=True)

    history_path = plan.output_dir / "history.csv"
    history = [
        row for row in read_history(history_path) if int(row["epoch"]) <= last_complete_epoch
    ]
    final_epoch = args.smoke_stop_after_epoch or config.epochs
    for epoch in range(last_complete_epoch + 1, final_epoch + 1):
        seed_everything(config.seed + epoch)
        center, metrics = train_one_epoch(
            records=records,
            config=config,
            epoch=epoch,
            student_encoder=student_encoder,
            teacher_encoder=teacher_encoder,
            student_head=student_head,
            teacher_head=teacher_head,
            optimizer=optimizer,
            scaler=scaler,
            center=center,
            device=device,
        )
        history.append(metrics)
        write_history(history_path, history)
        checkpoint_dir = save_training_checkpoint(
            output_dir=plan.output_dir,
            epoch=epoch,
            student_encoder=student_encoder,
            teacher_encoder=teacher_encoder,
            student_head=student_head,
            teacher_head=teacher_head,
            optimizer=optimizer,
            scaler_state=scaler.state_dict(),
            center=center,
            metadata={
                **common_metadata,
                "metrics": metrics,
                "smoke": config.smoke,
                "manifest_images_loaded": len(records),
                "global_crops": config.global_crops,
                "local_crops": config.local_crops,
                "amp": config.amp,
                "gradient_checkpointing": config.gradient_checkpointing,
                "ema_momentum": config.ema_momentum,
                "process_resumed_from_epoch": process_resumed_from_epoch,
            },
        )
        print(f"Saved complete checkpoint: {checkpoint_dir}", flush=True)
    if args.smoke_stop_after_epoch is not None:
        print(
            f"Smoke stopped after checkpoint_epoch{args.smoke_stop_after_epoch} as requested.",
            flush=True,
        )


if __name__ == "__main__":
    main()
