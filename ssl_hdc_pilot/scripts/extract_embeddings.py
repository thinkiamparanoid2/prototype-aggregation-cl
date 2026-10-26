"""Extract frozen DINOv3 CLS embeddings without using labels."""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from ssl_hdc_pilot.scripts.checkpoint_encoder import load_checkpoint_encoder
from ssl_hdc_pilot.scripts.encoder_models import (
    EncoderModelSpec,
    HEADLINE_MODEL_ID,
    LADDER_MODEL_SPECS,
    VITB16_MODEL_ID,
    VITL16_MODEL_ID,
    VITS16_MODEL_ID,
    get_model_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    image_path: Path


def validate_request(
    *,
    model_id: str,
    smoke: bool,
    max_images: int | None,
) -> None:
    get_model_spec(model_id)
    if smoke:
        if max_images is None:
            raise ValueError("--smoke requires --max-images")
        if not 1 <= max_images <= 100:
            raise ValueError("embedding smokes may use at most 100 images")
    elif max_images is not None:
        raise ValueError("--max-images is smoke-only; omit it for official extraction")


def load_records(input_csv: Path, image_root: Path | None) -> list[ImageRecord]:
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input CSV has no image records: {input_csv}")

    records: list[ImageRecord] = []
    seen: set[str] = set()
    for row in rows:
        image_id = (row.get("image_id") or row.get("image") or "").strip()
        if not image_id:
            raise ValueError("input CSV needs an image_id or image column")
        if image_id in seen:
            raise ValueError(f"duplicate image id in input CSV: {image_id}")
        seen.add(image_id)

        stored_path = (row.get("image_path") or "").strip()
        if stored_path:
            image_path = Path(stored_path)
            if not image_path.is_absolute():
                image_path = REPO_ROOT / image_path
        elif image_root is not None:
            image_path = image_root / image_id / f"{image_id}.jpg"
        else:
            raise ValueError("rows without image_path require --image-root")
        if not image_path.is_file():
            raise FileNotFoundError(f"missing image for {image_id}: {image_path}")
        records.append(ImageRecord(image_id=image_id, image_path=image_path))
    return records


DTYPE_BY_NAME = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def load_frozen_encoder(
    model_id: str, device: torch.device, dtype: torch.dtype = torch.float32
) -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
    # Load directly in the target dtype so the 7B ceiling rung never needs its
    # ~27 GB float32 footprint resident before casting (bf16 is ~13.4 GB).
    model = AutoModel.from_pretrained(
        model_id, local_files_only=True, torch_dtype=dtype
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen encoder guard failed")
    return model, processor


@torch.inference_mode()
def extract_records(
    *,
    records: Sequence[ImageRecord],
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[list[str], np.ndarray]:
    if not records:
        raise ValueError("cannot extract an empty image list")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    # Match float inputs (pixel_values) to the model's parameter dtype so a
    # bf16/fp16 encoder does not hit a dtype mismatch at the patch embedding.
    # Fall back to float32 for parameter-less stub models used in tests.
    model_dtype = next(model.parameters(), torch.empty(0)).dtype
    chunks: list[np.ndarray] = []
    image_ids: list[str] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images = []
        for record in batch:
            with Image.open(record.image_path) as image:
                images.append(image.convert("RGB").copy())
        inputs = processor(images=images, return_tensors="pt")
        moved: dict[str, Any] = {}
        for name, value in inputs.items():
            if torch.is_tensor(value):
                value = value.to(device, non_blocking=device.type == "cuda")
                if value.is_floating_point():
                    value = value.to(model_dtype)
            moved[name] = value
        inputs = moved
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[1] < 1:
            raise RuntimeError(f"unexpected DINOv3 hidden-state shape: {tuple(hidden.shape)}")
        cls = hidden[:, 0, :].float().cpu().numpy().astype(np.float32, copy=False)
        chunks.append(cls)
        image_ids.extend(record.image_id for record in batch)
    embeddings = np.concatenate(chunks, axis=0)
    return image_ids, embeddings


def write_embedding_artifacts(
    *,
    image_ids: Sequence[str],
    embeddings: np.ndarray,
    output_npz: Path,
    metadata_json: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if embeddings.dtype != np.float32:
        raise ValueError(f"embeddings must be float32, got {embeddings.dtype}")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(image_ids):
        raise ValueError("embedding array must be [number_of_ids, embedding_dim]")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("image ids must be unique")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain non-finite values")
    if output_npz.suffix.lower() != ".npz" or metadata_json.suffix.lower() != ".json":
        raise ValueError("embedding artifacts must use NPZ and JSON extensions")

    model_spec = get_model_spec(str(metadata.get("model_id", "")))
    if embeddings.shape[1] != model_spec.embedding_dim:
        raise ValueError(
            f"{model_spec.model_id} expected {model_spec.embedding_dim}-d embeddings, "
            f"got {embeddings.shape[1]}"
        )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        image_ids=np.asarray(list(image_ids), dtype=np.str_),
        embeddings=embeddings,
    )
    payload = {
        **metadata,
        "rung": model_spec.rung,
        "image_count": len(image_ids),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_dtype": "float32",
        "pooling": "CLS token (last_hidden_state[:, 0, :])",
        "labels_used": False,
        "encoder_frozen": True,
        "encoder_eval_mode": True,
    }
    metadata_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="frozen-encoder compute dtype; use bfloat16 for the 7B ceiling rung",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_provenance = None
    requested_model_id = args.model_id or HEADLINE_MODEL_ID
    if args.checkpoint_dir is not None:
        from ssl_hdc_pilot.scripts.checkpoint_encoder import inspect_checkpoint_export

        checkpoint_provenance = inspect_checkpoint_export(args.checkpoint_dir)
        if args.model_id is not None and args.model_id != checkpoint_provenance.model_id:
            raise RuntimeError("--model-id does not match checkpoint run config")
        requested_model_id = checkpoint_provenance.model_id
    validate_request(
        model_id=requested_model_id,
        smoke=args.smoke,
        max_images=args.max_images,
    )
    records = load_records(args.input_csv, args.image_root)
    if args.max_images is not None:
        records = records[: args.max_images]
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    encoder_dtype = DTYPE_BY_NAME[args.dtype]
    if args.checkpoint_dir is None:
        model, processor = load_frozen_encoder(
            requested_model_id, device, dtype=encoder_dtype
        )
    else:
        if args.dtype != "float32":
            raise ValueError("--dtype applies to frozen extraction only, not checkpoints")
        model, processor, checkpoint_provenance = load_checkpoint_encoder(
            args.checkpoint_dir, device
        )
    started = time.perf_counter()
    image_ids, embeddings = extract_records(
        records=records,
        model=model,
        processor=processor,
        device=device,
        batch_size=args.batch_size,
    )
    seconds = time.perf_counter() - started
    payload = write_embedding_artifacts(
        image_ids=image_ids,
        embeddings=embeddings,
        output_npz=args.output_npz,
        metadata_json=args.metadata_json,
        metadata={
            "model_id": requested_model_id,
            "encoder_source": "arm_d_checkpoint" if checkpoint_provenance else "frozen_pretrained",
            "checkpoint_provenance": (
                checkpoint_provenance.as_metadata() if checkpoint_provenance else None
            ),
            "device": str(device),
            "encoder_dtype": args.dtype,
            "batch_size": args.batch_size,
            "source_csv": str(args.input_csv),
            "preprocessing": processor.to_dict(),
            "local_files_only": True,
            "smoke": args.smoke,
            "elapsed_seconds": seconds,
        },
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
