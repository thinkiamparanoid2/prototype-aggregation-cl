"""Resumable driver: DINOv3 self-supervised adaptation + epoch probe-gate.

For one (dataset, encoder size) replication arm this drives, in order:

  1. manifest      -- build/verify the label-free train-only manifest.
  2. space_guard    -- refuse to start unless G: has >= 8 GiB + this arm's
                        expected peak checkpoint footprint free.
  3. adapt              -- launch ``python -m ssl_hdc_pilot.scripts.adapt_dinov3
                            --replication-dataset <D>`` with the locked recipe
                            (epochs=5, defaults unchanged) and this size's
                            --model-id.
  4. delete_training_states -- once all epochs finish training, delete every
                            epoch's training_state.npz (full training state;
                            only ever needed to resume, never for extraction).
                            encoder.npz (the EMA-teacher export used for
                            probing/extraction) is kept for every epoch.
  5. probe_extract      -- extract train/val embeddings for epochs 0..5 from
                            their checkpoints (frozen baseline + 5 adapted).
  6. gate               -- reuse the original fixed logistic-regression probe
                            gate (``ssl_hdc_pilot.scripts.probe_gate``,
                            tolerance 0.005) to score every epoch, then select
                            the checkpoint to keep from epochs 1..5 ONLY
                            (never epoch 0 -- see "Selection rule" below).
  7. final_embed        -- extract the selected checkpoint's embeddings over
                            the WHOLE dataset (all.csv) and write
                            embeddings/<Dataset>/adapted_<size>.npz(+.json),
                            in the same format as the frozen embedding files.
  8. verify             -- row count, id set, embedding dim, and finiteness
                            of the written adapted embeddings vs all.csv.
  9. cleanup            -- delete every non-selected epoch checkpoint
                            directory (their encoder.npz; training_state.npz
                            is already gone from step 4); keep only the
                            selected one.
 10. receipt             -- ssl_hdc_pilot/replication/adapted/<Dataset>/<size>/receipt.json

Every step is idempotent: it checks for a complete, verified prior output
before doing any work, and refuses to overwrite a differing finished
artifact. Re-running this driver after a crash (or after a step fails)
picks up where it left off; it never re-does a finished step and never
prompts interactively.

Selection rule (deliberately excludes epoch 0)
===============================================
The replication plan requires a real *adapted* arm for every (dataset,
size), even in the (rare) case where the frozen baseline would score
highest on the probe. So the checkpoint this driver actually keeps,
extracts final embeddings from, and reports as ``selected_checkpoint`` is
``argmax(val_macro_f1)`` over epochs **1..5 only** (ties resolved to the
earliest epoch by ``probe_gate.evaluate_embedding_gate``'s own left-to-right
``max()`` semantics over ascending-epoch input -- unmodified, not
reimplemented here). The frozen baseline's score and the *original* gate
verdict (best adapted >= frozen - 0.005) are still fully recorded in
``gate_result.json`` for disclosure; they just no longer decide what gets
extracted/kept.

Replication mode in adapt_dinov3.py
====================================
Step 3 shells out to ``ssl_hdc_pilot/scripts/adapt_dinov3.py --replication-dataset <D>``,
which (as of this driver) has an opt-in replication mode: without the flag
its behaviour, constants, and checks are byte-for-byte unchanged from the
original single-dataset study; with it, the manifest row-count/path/
source_pool checks are keyed to that dataset's own train split instead of
the original study's 51,906-row Data/train pool, and the required output
directory becomes ``ssl_hdc_pilot/replication/adapted/<D>/arm_d_<size>``
instead of the single global ``ssl_hdc_pilot/results/arm_d_<size>``.

``checkpoint_encoder.inspect_checkpoint_export`` (used by steps 5 and 7 via
``load_checkpoint_export``) still unconditionally refuses any checkpoint
whose ``run_config.json`` has ``smoke != False`` -- untouched, per the
coordinator's ruling. A ``--smoke`` checkpoint can therefore never be probed
or extracted through the official path; only real (non-smoke) runs ever
reach steps 5+.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from ssl_hdc_pilot.replication.adaptation_manifest import (
    DATASETS,
    build_adaptation_manifest,
)
from ssl_hdc_pilot.scripts import checkpoint_encoder as ce
from ssl_hdc_pilot.scripts import extract_embeddings as ee
from ssl_hdc_pilot.scripts import probe_gate as pg
from ssl_hdc_pilot.scripts.encoder_models import (
    EncoderModelSpec,
    VITB16_MODEL_ID,
    VITL16_MODEL_ID,
    VITS16_MODEL_ID,
    get_model_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPLICATION_ROOT = REPO_ROOT / "ssl_hdc_pilot" / "replication"
DATA_ROOT = REPLICATION_ROOT / "data"
EMBEDDINGS_ROOT = REPLICATION_ROOT / "embeddings"
ADAPTED_ROOT = REPLICATION_ROOT / "adapted"
PYTHON = REPO_ROOT / ".venv-e1" / "Scripts" / "python.exe"

SIZE_MODEL_IDS = {
    "vits16": VITS16_MODEL_ID,
    "vitb16": VITB16_MODEL_ID,
    "vitl16": VITL16_MODEL_ID,
}
SIZES = tuple(SIZE_MODEL_IDS)
EPOCHS = 5
SPACE_GUARD_DRIVE = "G:/"
MIN_FREE_GIB = 8.0

# --- Hardware profile (measured 2026-09-17 on this rig) ---------------------
# RTX 5070 (12 GiB VRAM), Ryzen 5 3600 (6 cores / 12 threads), 16 GiB RAM.
# adapt_dinov3.py's DataLoader (make_loader) recreates its worker pool every
# epoch (Windows uses the multiprocessing "spawn" start method, which
# re-imports torch/transformers per worker process -- that is the fixed
# per-epoch startup tax below). Measured directly via adapt_dinov3's own
# train_one_epoch()/make_loader(), called in-process (bypassing the CLI, no
# checkpoint written) on PlantDoc's real 1535-image train pool, batch_size=32
# (the locked recipe), amp+gradient_checkpointing on, one epoch = 48 steps:
#   num_workers=0: ~0.55 steps/sec steady-state, no per-epoch startup tax,
#                  GPU util mostly 0-2% between bursts (CPU decode-bound) ->
#                  ~87s/epoch.
#   num_workers=3: ~16s one-time worker-spawn tax/epoch, then ~1.49 steps/sec
#                  steady-state -> ~47s/epoch.
#   num_workers=5: ~24s one-time worker-spawn tax/epoch, then ~2.4 steps/sec
#                  steady-state -> ~44s/epoch on PlantDoc; the startup tax is
#                  a fixed ~24s regardless of dataset size, so it amortizes
#                  away on PlantWild (337 steps/epoch) and PlantVillage
#                  (1019 steps/epoch), where num_workers=5's steady-state
#                  throughput advantage over num_workers=3 dominates.
#   vitl16 @ num_workers=5, 15 steps: GPU util saturates to ~95-100% once
#                  the pipeline is warm (compute-bound, not data-starved);
#                  peak CUDA memory 7.41 GiB of 12 GiB (headroom OK), peak
#                  process RSS ~3.2 GiB of 16 GiB RAM (headroom OK).
# Chosen default: 5 (best steady-state throughput on every dataset size in
# the ladder; the fixed per-epoch respawn tax only matters on the smallest
# dataset, PlantDoc, where it is still a net win over num_workers=0).
# num_workers only changes which OS process runs each augmentation
# __getitem__ call, not amp/batch_size/crops/lr/momentum/epochs/device; see
# adapt_dinov3.validate_config's comment for the augmentation-RNG-stream
# caveat this implies (recipe-equivalent, not bitwise-identical).
DEFAULT_NUM_WORKERS = 5
# Measured directly for vits16 (~22M params): encoder.npz (EMA-teacher
# export) = 86,448,700 bytes; training_state.npz (full training state:
# student encoder+head, teacher head, AdamW optimizer state, center, scaler;
# only written for epoch >= 1) = 286,690,124 bytes (~3.32x the encoder).
# vitb16 (86M params) and vitl16 (300M params) are NOT measured -- both
# byte counts are extrapolated linearly by parameter count (fp32 storage is
# ~4 bytes/param, matching the vits16 measurement almost exactly), per the
# coordinator's ruling.
#
# Peak-per-arm occurs transiently, right when the 5th epoch's checkpoint
# finishes saving (before delete_training_states runs): epoch0's encoder +
# 5 * (encoder + training_state). The checkpoint-policy refinement (deleting
# every training_state.npz immediately once all epochs finish, instead of
# carrying them through probe_extract/gate/final_embed) does not lower this
# transient peak, but it does mean disk usage drops sharply right after
# training instead of staying elevated through the rest of the arm -- so the
# window during which the guard's headroom is actually needed is now just
# the adapt step, not the whole run. Values below include ~20% padding for
# probe/final embedding files and rounding.
_VITS16_ENCODER_BYTES = 86_448_700
_VITS16_TRAINING_STATE_BYTES = 286_690_124
_PARAM_COUNTS = {"vits16": 22_000_000, "vitb16": 86_000_000, "vitl16": 300_000_000}


def _peak_gib(size: str) -> float:
    ratio = _PARAM_COUNTS[size] / _PARAM_COUNTS["vits16"]
    encoder_bytes = _VITS16_ENCODER_BYTES * ratio
    training_state_bytes = _VITS16_TRAINING_STATE_BYTES * ratio
    peak_bytes = encoder_bytes + EPOCHS * (encoder_bytes + training_state_bytes)
    return (peak_bytes / 1024**3) * 1.2  # +20% safety padding


EXPECTED_PEAK_GIB = {size: round(_peak_gib(size), 1) for size in SIZES}

KNOWN_BLOCKER = (
    "adapt_dinov3.py's replication mode (--replication-dataset) still failed; see "
    "the 'adapt' step's stderr_tail above for the exact error."
)


@dataclass(frozen=True)
class ArmPaths:
    dataset: str
    size: str
    model_id: str
    model_spec: EncoderModelSpec
    run_dir: Path
    manifest_csv: Path
    train_csv: Path
    val_csv: Path
    all_csv: Path
    checkpoints_root: Path
    checkpoint_dir: Path
    probe_dir: Path
    gate_json: Path
    final_npz: Path
    final_json: Path
    receipt_json: Path


def arm_paths(
    dataset: str,
    size: str,
    *,
    adapted_root: Path = ADAPTED_ROOT,
    embeddings_root: Path = EMBEDDINGS_ROOT,
) -> ArmPaths:
    if dataset not in DATASETS:
        raise ValueError(f"unknown replication dataset: {dataset!r}; expected one of {DATASETS}")
    if size not in SIZES:
        raise ValueError(f"unknown encoder size: {size!r}; expected one of {SIZES}")
    model_id = SIZE_MODEL_IDS[size]
    model_spec = get_model_spec(model_id)
    run_dir = Path(adapted_root) / dataset / size
    # adapt_dinov3.py's replication mode UNCONDITIONALLY requires the
    # checkpoint output dir to be exactly ADAPTED_ROOT/<dataset>/arm_d_<size>
    # -- this check is not parametrized and cannot be relaxed for a sandboxed
    # test, so checkpoints_root/checkpoint_dir always use the real
    # ADAPTED_ROOT regardless of the adapted_root override above. That
    # override only relocates this driver's own bookkeeping (run_dir: probe
    # embeddings, gate_result.json, receipt.json) and the final embeddings
    # (embeddings_root), which have no such constraint.
    checkpoints_root = ADAPTED_ROOT / dataset
    return ArmPaths(
        dataset=dataset,
        size=size,
        model_id=model_id,
        model_spec=model_spec,
        run_dir=run_dir,
        manifest_csv=DATA_ROOT / dataset / "adaptation_train_manifest.csv",
        train_csv=DATA_ROOT / dataset / "splits" / "train.csv",
        val_csv=DATA_ROOT / dataset / "splits" / "val.csv",
        all_csv=DATA_ROOT / dataset / "splits" / "all.csv",
        checkpoints_root=checkpoints_root,
        checkpoint_dir=checkpoints_root / model_spec.arm_d_directory,
        probe_dir=run_dir / "probe_epochs",
        gate_json=run_dir / "gate_result.json",
        final_npz=Path(embeddings_root) / dataset / f"adapted_{size}.npz",
        final_json=Path(embeddings_root) / dataset / f"adapted_{size}.json",
        receipt_json=run_dir / "receipt.json",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(csv_path: Path, *, with_labels: bool) -> tuple[list[ee.ImageRecord], np.ndarray | None]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"replication split CSV has no rows: {csv_path}")
    records: list[ee.ImageRecord] = []
    labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        image_id = row["image"].strip()
        if not image_id or image_id in seen:
            raise RuntimeError(f"invalid or duplicate image id in {csv_path}: {image_id!r}")
        seen.add(image_id)
        raw_path = row["image_path"].strip()
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = REPO_ROOT / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"missing image for {image_id}: {image_path}")
        records.append(ee.ImageRecord(image_id=image_id, image_path=image_path))
        if with_labels:
            labels.append(row["label"].strip())
    return records, (np.asarray(labels) if with_labels else None)


# --- Low-RAM streaming extraction ------------------------------------------
#
# extract_embeddings.extract_records (unmodified, not touched here) opens a
# whole batch of full-resolution PIL images into a Python list before ever
# calling the processor. For PlantDoc's big web photos at batch_size=64 that
# briefly holds 64 full-resolution decoded images at once, which caused an
# OS low-memory kill on this 16 GB-RAM box. This driver-local replacement
# keeps extract_records' exact conventions -- same ImageRecord/processor/
# model-forward/CLS-pooling, same output order and ids, same GPU batch size
# 64 in float32 under torch.inference_mode -- but decodes and preprocesses
# one image at a time (dropping the full-resolution PIL image the instant
# its processor call returns, keeping only the small 224x224 pixel tensor),
# using a small thread pool feeding a bounded in-flight pipeline instead of
# materializing a whole batch of decoded images up front.
STREAM_EXTRACTION_THREADS = 6


def _decode_and_preprocess(record: "ee.ImageRecord", processor: Any) -> torch.Tensor:
    """Decode one image and run the processor on it alone. The PIL image is
    dropped (context manager exit + no surviving reference) as soon as this
    call returns; only the resulting 224x224 float32 pixel tensor survives."""
    with Image.open(record.image_path) as image:
        rgb = image.convert("RGB").copy()
    inputs = processor(images=[rgb], return_tensors="pt")
    return inputs["pixel_values"][0]


@torch.inference_mode()
def extract_records_streaming(
    *,
    records: Sequence["ee.ImageRecord"],
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    batch_size: int = 64,
    num_threads: int = STREAM_EXTRACTION_THREADS,
) -> tuple[list[str], np.ndarray]:
    if not records:
        raise ValueError("cannot extract an empty image list")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    # Same dtype-matching rule as extract_records: match pixel_values to the
    # model's parameter dtype (float32 for every caller in this driver).
    model_dtype = next(model.parameters(), torch.empty(0)).dtype
    total = len(records)
    max_in_flight = max(num_threads * 2, batch_size)

    image_ids: list[str] = []
    chunks: list[np.ndarray] = []
    batch_tensors: list[torch.Tensor] = []
    batch_ids: list[str] = []

    def _run_batch() -> None:
        nonlocal batch_tensors, batch_ids
        pixel_values = torch.stack(batch_tensors, dim=0)
        pixel_values = pixel_values.to(device, non_blocking=device.type == "cuda")
        if pixel_values.is_floating_point():
            pixel_values = pixel_values.to(model_dtype)
        outputs = model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[1] < 1:
            raise RuntimeError(f"unexpected DINOv3 hidden-state shape: {tuple(hidden.shape)}")
        cls = hidden[:, 0, :].float().cpu().numpy().astype(np.float32, copy=False)
        chunks.append(cls)
        image_ids.extend(batch_ids)
        batch_tensors = []
        batch_ids = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        pending: deque = deque()
        next_submit = 0

        def _submit_next() -> None:
            nonlocal next_submit
            if next_submit < total:
                pending.append(
                    executor.submit(_decode_and_preprocess, records[next_submit], processor)
                )
                next_submit += 1

        # Prime the pipeline: only ever up to max_in_flight images decoded
        # ahead of GPU consumption, regardless of dataset size -- this is
        # the "bounded in-flight" queue that keeps RAM flat even on
        # PlantVillage's ~32k-image full-dataset extraction.
        while next_submit < total and len(pending) < max_in_flight:
            _submit_next()

        for index in range(total):
            tensor = pending.popleft().result()
            _submit_next()
            batch_tensors.append(tensor)
            batch_ids.append(records[index].image_id)
            if len(batch_tensors) == batch_size:
                _run_batch()
        if batch_tensors:
            _run_batch()

    embeddings = np.concatenate(chunks, axis=0)
    return image_ids, embeddings


def ensure_manifest(dataset: str) -> dict[str, Any]:
    result = build_adaptation_manifest(dataset)
    return {
        "row_count": result.row_count,
        "train_csv_row_count": result.train_csv_row_count,
        "skipped_existing": result.skipped_existing,
    }


def check_disk_guard(size: str, *, drive: str = SPACE_GUARD_DRIVE) -> dict[str, Any]:
    usage = shutil.disk_usage(drive)
    free_gib = usage.free / 1024**3
    peak_gib = EXPECTED_PEAK_GIB[size]
    required_gib = MIN_FREE_GIB + peak_gib
    if free_gib < required_gib:
        raise RuntimeError(
            f"free space on {drive} is {free_gib:.2f} GiB, below the required "
            f"{required_gib:.2f} GiB (8 GiB floor + {peak_gib:.1f} GiB expected peak for {size})"
        )
    return {"drive": drive, "free_gib": free_gib, "required_gib": required_gib, "peak_gib": peak_gib}


def _epoch_checkpoint_present(checkpoint_dir: Path, epoch: int, model_id: str) -> bool:
    """Encoder-only completeness check: true once encoder.npz + a matching
    complete checkpoint_metadata.json exist, regardless of whether
    training_state.npz is still there. Deliberately does NOT reuse
    adapt_dinov3._checkpoint_is_complete, which requires training_state.npz
    for epoch > 0 -- exactly the file delete_training_states removes once an
    arm finishes, which would otherwise make an already-finished arm look
    incomplete forever and cause run_adaptation to re-invoke the CLI against
    a checkpoint directory that already exists (hard erroring on
    "refusing to overwrite complete checkpoint")."""
    path = checkpoint_dir / f"checkpoint_epoch{epoch}"
    metadata_path = path / "checkpoint_metadata.json"
    if not metadata_path.is_file() or not (path / "encoder.npz").is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("complete") is True
        and metadata.get("epoch") == epoch
        and metadata.get("model_id") == model_id
    )


def _adaptation_already_complete(checkpoint_dir: Path, model_id: str, target_epoch: int) -> bool:
    return checkpoint_dir.is_dir() and all(
        _epoch_checkpoint_present(checkpoint_dir, epoch, model_id)
        for epoch in range(0, target_epoch + 1)
    )


def run_adaptation(
    paths: ArmPaths,
    *,
    device: str,
    epochs: int = EPOCHS,
    smoke: bool = False,
    max_images: int | None = None,
    max_steps: int | None = None,
    smoke_stop_after_epoch: int | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict[str, Any]:
    target_epoch = smoke_stop_after_epoch or epochs
    if _adaptation_already_complete(paths.checkpoint_dir, paths.model_id, target_epoch):
        return {"status": "SKIPPED_ALREADY_COMPLETE", "target_epoch": target_epoch}

    cmd = [
        str(PYTHON), "-m", "ssl_hdc_pilot.scripts.adapt_dinov3",
        "--manifest", str(paths.manifest_csv),
        "--output-dir", str(paths.checkpoint_dir),
        "--model-id", paths.model_id,
        "--epochs", str(epochs),
        "--device", device,
        "--replication-dataset", paths.dataset,
        "--num-workers", str(num_workers),
    ]
    if smoke:
        cmd.append("--smoke")
        if max_images is not None:
            cmd += ["--max-images", str(max_images)]
        if max_steps is not None:
            cmd += ["--max-steps", str(max_steps)]
        if smoke_stop_after_epoch is not None:
            cmd += ["--smoke-stop-after-epoch", str(smoke_stop_after_epoch)]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        stderr_tail = result.stderr.strip().splitlines()[-8:]
        return {
            "status": "BLOCKED",
            "command": cmd,
            "returncode": result.returncode,
            "stderr_tail": stderr_tail,
            "known_blocker": KNOWN_BLOCKER,
        }
    return {"status": "COMPLETE", "command": cmd, "stdout_tail": result.stdout.strip().splitlines()[-8:]}


def delete_training_states(paths: ArmPaths, *, last_epoch: int) -> dict[str, Any]:
    """Delete every epoch's training_state.npz once training is complete.

    training_state.npz (full training state: student encoder+head, teacher
    head, AdamW optimizer state, center, scaler) is only ever read to resume
    a crashed run; once all epochs are done there is nothing left to resume,
    and neither probe_extract nor final_embed ever touches it (both load
    only encoder.npz, the EMA-teacher export). Idempotent: already-deleted
    files are simply skipped.
    """
    deleted: list[str] = []
    already_gone: list[str] = []
    for epoch in range(1, last_epoch + 1):
        training_state = paths.checkpoint_dir / f"checkpoint_epoch{epoch}" / "training_state.npz"
        if training_state.is_file():
            training_state.unlink()
            deleted.append(str(training_state))
        else:
            already_gone.append(str(training_state))
    return {"deleted": deleted, "already_gone": already_gone}


def _probe_npz_valid(path: Path, *, expected_rows: int, expected_dim: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"image_ids", "embeddings"}:
                return False
            embeddings = archive["embeddings"]
            if embeddings.shape != (expected_rows, expected_dim):
                return False
            return bool(np.isfinite(embeddings).all())
    except (OSError, ValueError):
        return False


def probe_extract(
    paths: ArmPaths, *, device: torch.device, last_epoch: int = EPOCHS
) -> dict[str, dict[str, Path]]:
    paths.probe_dir.mkdir(parents=True, exist_ok=True)
    train_records, _ = _load_records(paths.train_csv, with_labels=False)
    val_records, _ = _load_records(paths.val_csv, with_labels=False)
    dim = paths.model_spec.embedding_dim

    model = None
    processor = None
    results: dict[str, dict[str, Path]] = {}
    for epoch in range(0, last_epoch + 1):
        name = f"checkpoint_epoch{epoch}"
        train_npz = paths.probe_dir / f"{name}_train.npz"
        val_npz = paths.probe_dir / f"{name}_val.npz"
        if _probe_npz_valid(train_npz, expected_rows=len(train_records), expected_dim=dim) and _probe_npz_valid(
            val_npz, expected_rows=len(val_records), expected_dim=dim
        ):
            results[name] = {"train": train_npz, "val": val_npz}
            continue

        if model is None:
            from transformers import AutoImageProcessor, AutoModel

            processor = AutoImageProcessor.from_pretrained(paths.model_id, local_files_only=True)
            model = AutoModel.from_pretrained(paths.model_id, local_files_only=True).to(device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

        checkpoint_dir = paths.checkpoint_dir / name
        provenance = ce.load_checkpoint_export(checkpoint_dir, model, device, results_root=paths.checkpoints_root)
        if provenance.model_id != paths.model_id:
            raise RuntimeError(f"{name}: checkpoint model id does not match {paths.model_id}")

        train_ids, train_x = extract_records_streaming(
            records=train_records, model=model, processor=processor, device=device, batch_size=64
        )
        val_ids, val_x = extract_records_streaming(
            records=val_records, model=model, processor=processor, device=device, batch_size=64
        )
        _write_probe_npz(train_npz, train_ids, train_x)
        _write_probe_npz(val_npz, val_ids, val_x)
        results[name] = {"train": train_npz, "val": val_npz}
    return results


def _write_probe_npz(path: Path, image_ids: list[str], embeddings: np.ndarray) -> None:
    if not np.isfinite(embeddings).all():
        raise ValueError(f"non-finite probe embeddings: {path}")
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, image_ids=np.asarray(image_ids, dtype=np.str_), embeddings=embeddings)
    tmp.replace(path)


def run_gate(paths: ArmPaths, probe_results: dict[str, dict[str, Path]]) -> dict[str, Any]:
    train_records, train_labels = _load_records(paths.train_csv, with_labels=True)
    val_records, val_labels = _load_records(paths.val_csv, with_labels=True)
    train_ids_order = [record.image_id for record in train_records]
    val_ids_order = [record.image_id for record in val_records]

    probe_hashes: dict[str, str] = {}
    checkpoint_embeddings: dict[str, dict[str, np.ndarray]] = {}
    for name, files in sorted(probe_results.items()):
        with np.load(files["train"]) as train_archive, np.load(files["val"]) as val_archive:
            if list(train_archive["image_ids"]) != train_ids_order:
                raise RuntimeError(f"{name}: probe train ids do not match train.csv order")
            if list(val_archive["image_ids"]) != val_ids_order:
                raise RuntimeError(f"{name}: probe val ids do not match val.csv order")
            checkpoint_embeddings[name] = {
                "train": train_archive["embeddings"],
                "val": val_archive["embeddings"],
            }
        probe_hashes[name] = _sha256_file(files["train"]) + ":" + _sha256_file(files["val"])

    if paths.gate_json.is_file():
        existing = json.loads(paths.gate_json.read_text(encoding="utf-8"))
        if existing.get("probe_file_hashes") == probe_hashes:
            return existing
        raise RuntimeError(
            f"refusing to reuse a stale gate result whose probe embeddings changed: {paths.gate_json}"
        )

    gate = pg.evaluate_embedding_gate(
        checkpoint_embeddings, train_labels, val_labels, model_spec=paths.model_spec
    )
    # evaluate_embedding_gate (probe_gate.py, unmodified) already computes
    # best_adapted_checkpoint = argmax(val_macro_f1) over epochs 1..5 ONLY
    # (never epoch0), with ties resolved to the earliest epoch by max()'s
    # left-to-right semantics over our ascending-epoch input. That is
    # exactly the arm this driver must keep/extract/embed -- unlike its own
    # `selected_checkpoint` field, which picks over ALL epochs including 0
    # and is kept below only for disclosure, not for downstream use.
    final_selected = gate["best_adapted_checkpoint"]
    payload = {
        "dataset": paths.dataset,
        "size": paths.size,
        "model_id": paths.model_id,
        "train_csv": str(paths.train_csv),
        "val_csv": str(paths.val_csv),
        "checkpoint_table": gate["checkpoint_table"],
        "frozen_baseline_checkpoint": gate["frozen_baseline_checkpoint"],
        "frozen_baseline_val_macro_f1": gate["frozen_baseline_val_macro_f1"],
        "best_adapted_checkpoint": gate["best_adapted_checkpoint"],
        "best_adapted_val_macro_f1": gate["best_adapted_val_macro_f1"],
        "tolerance": gate["tolerance"],
        "pass_threshold": gate["pass_threshold"],
        "pass": gate["pass"],
        "probe_gate_selected_checkpoint": gate["selected_checkpoint"],
        "selected_checkpoint": final_selected,
        "probe_file_hashes": probe_hashes,
        "metric": "macro_f1",
        "probe": {
            "class": "sklearn.linear_model.LogisticRegression",
            "max_iter": pg.PROBE_MAX_ITER,
            "class_weight": "balanced",
            "random_state": pg.PROBE_RANDOM_STATE,
        },
        "selection_rule": (
            "Fit LogisticRegression(class_weight='balanced', max_iter=2000, "
            "random_state=20260716) on each checkpoint's train embeddings; score "
            "macro-F1 on val embeddings for checkpoint_epoch0 (frozen baseline) "
            "through checkpoint_epoch5 (adapted). 'pass'/'threshold' disclose the "
            "ORIGINAL probe_gate.py rule: best_adapted_val_macro_f1 >= "
            "frozen_baseline_val_macro_f1 - 0.005 (tolerance). 'selected_checkpoint' "
            "is what this driver actually extracts/keeps: argmax(val_macro_f1) over "
            "epochs 1..5 ONLY (ties -> earliest epoch), i.e. always the best ADAPTED "
            "checkpoint -- epoch 0 is never selected here, even if it scored higher "
            "(see frozen_baseline_val_macro_f1/pass above for that disclosure). "
            "'probe_gate_selected_checkpoint' is probe_gate.py's own unmodified "
            "field (argmax over ALL epochs including 0) kept for audit only."
        ),
    }
    paths.gate_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.gate_json.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(paths.gate_json)
    return payload


def write_final_embeddings(paths: ArmPaths, gate: dict[str, Any], device: torch.device) -> None:
    if paths.final_npz.is_file() and paths.final_json.is_file():
        return

    selected = gate["selected_checkpoint"]
    checkpoint_dir = paths.checkpoint_dir / selected
    all_records, _ = _load_records(paths.all_csv, with_labels=False)

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(paths.model_id, local_files_only=True)
    model = AutoModel.from_pretrained(paths.model_id, local_files_only=True).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    provenance = ce.load_checkpoint_export(checkpoint_dir, model, device, results_root=paths.checkpoints_root)
    ids, embeddings = extract_records_streaming(
        records=all_records, model=model, processor=processor, device=device, batch_size=64
    )
    ee.write_embedding_artifacts(
        image_ids=ids,
        embeddings=embeddings,
        output_npz=paths.final_npz,
        metadata_json=paths.final_json,
        metadata={
            "model_id": paths.model_id,
            "encoder_source": "arm_d_checkpoint",
            "checkpoint_provenance": provenance.as_metadata(),
            "selected_checkpoint": selected,
            "dataset": paths.dataset,
            "device": str(device),
            "source_csv": str(paths.all_csv),
        },
    )


def verify_final_embeddings(paths: ArmPaths) -> dict[str, Any]:
    with paths.all_csv.open(newline="", encoding="utf-8-sig") as handle:
        all_ids = {row["image"].strip() for row in csv.DictReader(handle)}
    if not paths.final_npz.is_file():
        raise FileNotFoundError(f"final adapted embeddings missing: {paths.final_npz}")
    with np.load(paths.final_npz) as archive:
        ids = archive["image_ids"]
        embeddings = archive["embeddings"]
        if len(ids) != len(all_ids):
            raise RuntimeError(
                f"final embedding row count {len(ids)} != all.csv row count {len(all_ids)}"
            )
        if set(ids.tolist()) != all_ids:
            raise RuntimeError("final embedding ids do not match all.csv identities")
        if embeddings.shape[1] != paths.model_spec.embedding_dim:
            raise RuntimeError(
                f"final embedding dim {embeddings.shape[1]} != expected {paths.model_spec.embedding_dim}"
            )
        if not np.isfinite(embeddings).all():
            raise RuntimeError("final embeddings contain non-finite values")
    return {"rows": int(len(ids)), "dim": int(embeddings.shape[1]), "verified": True}


def cleanup_checkpoints(paths: ArmPaths, gate: dict[str, Any]) -> dict[str, Any]:
    selected = gate["selected_checkpoint"]
    deleted: list[str] = []
    for path in sorted(paths.checkpoint_dir.glob("checkpoint_epoch*")):
        if not path.is_dir() or path.name == selected:
            continue
        shutil.rmtree(path)
        deleted.append(path.name)
    return {"kept": selected, "deleted": deleted}


def write_receipt(paths: ArmPaths, *, status: str, steps: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "dataset": paths.dataset,
        "size": paths.size,
        "model_id": paths.model_id,
        "status": status,
        "steps": steps,
    }
    paths.receipt_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.receipt_json.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(paths.receipt_json)
    return payload


def run_arm(
    dataset: str,
    size: str,
    *,
    device: str = "cuda",
    adapted_root: Path = ADAPTED_ROOT,
    embeddings_root: Path = EMBEDDINGS_ROOT,
    epochs: int = EPOCHS,
    smoke: bool = False,
    max_images: int | None = None,
    max_steps: int | None = None,
    smoke_stop_after_epoch: int | None = None,
    skip_space_guard: bool = False,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> dict[str, Any]:
    paths = arm_paths(dataset, size, adapted_root=adapted_root, embeddings_root=embeddings_root)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    steps: dict[str, Any] = {}

    # A finished arm keeps only its selected checkpoint, so the per-epoch completeness
    # check below would misread it as partial and relaunch training. Skip the whole arm
    # when its gate result exists and its final embeddings still verify.
    if paths.gate_json.is_file() and paths.final_npz.is_file() and paths.final_json.is_file():
        gate = json.loads(paths.gate_json.read_text(encoding="utf-8"))
        selected = gate.get("selected_checkpoint")
        if selected and (paths.checkpoint_dir / selected).is_dir():
            steps["resume"] = {"status": "SKIPPED_ARM_ALREADY_COMPLETE", "selected_checkpoint": selected}
            steps["final_embed"] = verify_final_embeddings(paths)
            return write_receipt(paths, status="COMPLETE", steps=steps)

    steps["manifest"] = ensure_manifest(dataset)
    steps["space_guard"] = (
        {"skipped": True} if skip_space_guard else check_disk_guard(size)
    )

    adapt_result = run_adaptation(
        paths,
        device=device,
        epochs=epochs,
        smoke=smoke,
        max_images=max_images,
        max_steps=max_steps,
        smoke_stop_after_epoch=smoke_stop_after_epoch,
        num_workers=num_workers,
    )
    adapt_result["num_workers"] = num_workers
    steps["adapt"] = adapt_result
    if adapt_result["status"] == "BLOCKED":
        write_receipt(paths, status="BLOCKED", steps=steps)
        raise RuntimeError(
            f"adaptation for {dataset}/{size} is blocked; see {paths.receipt_json}\n{KNOWN_BLOCKER}"
        )

    last_epoch = smoke_stop_after_epoch or epochs
    try:
        # training_state.npz is kept: checkpoint_encoder.inspect_checkpoint_export refuses
        # adapted checkpoints without it, so the selected checkpoint must stay complete.
        # Non-selected epochs are removed whole by cleanup_checkpoints.
        device_t = torch.device(
            "cuda" if device in ("cuda", "auto") and torch.cuda.is_available() else "cpu"
        )
        probe_results = probe_extract(paths, device=device_t, last_epoch=last_epoch)
        steps["probe_extract"] = {
            name: {"train": str(files["train"]), "val": str(files["val"])}
            for name, files in probe_results.items()
        }

        gate = run_gate(paths, probe_results)
        steps["gate"] = {
            "selected_checkpoint": gate["selected_checkpoint"],
            "frozen_baseline_val_macro_f1": gate["frozen_baseline_val_macro_f1"],
            "best_adapted_val_macro_f1": gate["best_adapted_val_macro_f1"],
            "original_gate_pass": gate["pass"],
        }

        write_final_embeddings(paths, gate, device_t)
        verification = verify_final_embeddings(paths)
        steps["final_embed"] = verification

        cleanup = cleanup_checkpoints(paths, gate)
        steps["cleanup"] = cleanup
    except Exception as error:
        steps["error"] = f"{type(error).__name__}: {error}"
        write_receipt(paths, status="ERROR", steps=steps)
        raise

    return write_receipt(paths, status="COMPLETE", steps=steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--adapted-root", type=Path, default=ADAPTED_ROOT)
    parser.add_argument("--embeddings-root", type=Path, default=EMBEDDINGS_ROOT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke-stop-after-epoch", type=int)
    parser.add_argument(
        "--skip-space-guard",
        action="store_true",
        help="smoke-only escape hatch; official runs must keep the guard",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "adapt_dinov3 DataLoader worker count; see the hardware-profile "
            f"comment block above DEFAULT_NUM_WORKERS in this file (default {DEFAULT_NUM_WORKERS})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_space_guard and not args.smoke:
        raise SystemExit("--skip-space-guard requires --smoke")
    try:
        receipt = run_arm(
            args.dataset,
            args.size,
            device=args.device,
            adapted_root=args.adapted_root,
            embeddings_root=args.embeddings_root,
            epochs=args.epochs,
            smoke=args.smoke,
            max_images=args.max_images,
            max_steps=args.max_steps,
            smoke_stop_after_epoch=args.smoke_stop_after_epoch,
            skip_space_guard=args.skip_space_guard,
            num_workers=args.num_workers,
        )
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
