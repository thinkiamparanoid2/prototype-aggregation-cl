"""Lossless NPZ storage with optional run-local, content-addressed projections.

The storage envelope is separate from the logical learner/prediction schema.
Legacy archives need no root. Compact archives require an explicit bundle root;
copy the whole run directory, not an individual compact checkpoint. Hashes detect
corruption, not authenticity. Only load artifacts from trusted sources.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np


STORAGE_KEY = "_storage_json"
STORAGE_KIND = "shared_projection_npz"
STORAGE_VERSION = 1
BLOB_PATTERN = r"projection_blobs/[0-9a-f]{64}\.npz"
STORAGE_DESCRIPTION = {
    "kind": STORAGE_KIND, "version": STORAGE_VERSION,
    "root": "run_directory", "shared_array": "projection",
    "blob_directory": "projection_blobs",
    "checkpoint_bytes_exclude_shared_blobs": True,
}


def _projection_digest(value: np.ndarray) -> str:
    if value.dtype != np.float32 or value.ndim != 2 or min(value.shape) < 1:
        raise ValueError("shared projection must be a nonempty float32 matrix")
    header = json.dumps([value.dtype.str, list(value.shape)], separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(header + b"\n")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _inside_root(path: Path, root: Path) -> Path:
    def comparable(resolved: Path) -> Path:
        # Windows may return the extended namespace for a concurrently created
        # file but the ordinary spelling for its parent. Compare like spellings
        # only AFTER resolution, retaining symlink/junction escape detection.
        text = str(resolved)
        if os.name == "nt":
            if text.startswith("\\\\?\\UNC\\"):
                return Path("\\\\" + text[8:])
            if text.startswith("\\\\?\\"):
                return Path(text[4:])
        return resolved

    root = comparable(root.resolve())
    resolved = path.resolve()
    if not comparable(resolved).is_relative_to(root):
        raise ValueError(f"checkpoint or blob path escapes shared projection root: {resolved!s} outside {root!s}")
    return resolved


def _publish(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Publish a complete archive atomically, never replacing an existing name.

    A temporary hard link is used only for publication within the same folder.
    The temporary name is removed afterward; no completed artifacts are linked
    together. Unsupported filesystems fail closed (no non-atomic fallback).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def _archive_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("checkpoint array inventory contains duplicate entries")
        return {key: archive[key].copy() for key in archive.files}


def _reference(raw: np.ndarray) -> str:
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError("invalid shared projection storage encoding")
    record = json.loads(raw.tobytes().decode("utf-8"))
    if (not isinstance(record, dict) or set(record) != {"kind", "version", "projection_sha256"}
            or record["kind"] != STORAGE_KIND or type(record["version"]) is not int
            or record["version"] != STORAGE_VERSION
            or not isinstance(record["projection_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", record["projection_sha256"]) is None):
        raise ValueError("invalid or unsupported shared projection storage envelope")
    return record["projection_sha256"]


def _blob_path(path: Path, root: Path | None, digest: str) -> Path:
    if root is None:
        raise ValueError("compact checkpoint requires explicit shared_projection_root (the run directory)")
    root = Path(root)
    _inside_root(path, root)
    return _inside_root(root / "projection_blobs" / f"{digest}.npz", root)


def _read_projection(path: Path, digest: str) -> np.ndarray:
    arrays = _archive_arrays(path)
    if set(arrays) != {"projection"} or _projection_digest(arrays["projection"]) != digest:
        raise ValueError("shared projection blob integrity mismatch")
    return arrays["projection"]


def projection_dependencies(path: Path, *, shared_projection_root: Path) -> set[str]:
    """Read only the envelope, for receipt dependency closure before scoring."""
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("checkpoint array inventory contains duplicate entries")
        if STORAGE_KEY not in archive.files:
            return set()
        if "projection" in archive.files:
            raise ValueError("compact checkpoint also embeds a projection")
        digest = _reference(archive[STORAGE_KEY])
    _blob_path(path, shared_projection_root, digest)
    return {f"projection_blobs/{digest}.npz"}


def save_arrays(path: Path, arrays: dict[str, np.ndarray], *, shared_projection_root: Path | None = None) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    if STORAGE_KEY in arrays:
        raise ValueError("reserved checkpoint storage metadata key")
    payload = dict(arrays)
    if shared_projection_root is not None and "projection" in payload:
        projection = payload.pop("projection")
        digest = _projection_digest(projection)
        blob = _blob_path(path, shared_projection_root, digest)
        if not blob.exists():
            try:
                _publish(blob, {"projection": projection})
            except FileExistsError:
                pass  # Another writer published first; verify before reuse.
        _read_projection(blob, digest)
        record = {"kind": STORAGE_KIND, "version": STORAGE_VERSION, "projection_sha256": digest}
        payload[STORAGE_KEY] = np.frombuffer(json.dumps(record, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    _publish(path, payload)


def load_arrays(path: Path, *, shared_projection_root: Path | None = None) -> dict[str, np.ndarray]:
    """Resolve storage without constructing a learner or changing any RNG."""
    path = Path(path)
    arrays = _archive_arrays(path)
    if STORAGE_KEY in arrays:
        if "projection" in arrays:
            raise ValueError("compact checkpoint also embeds a projection")
        digest = _reference(arrays.pop(STORAGE_KEY))
        arrays["projection"] = _read_projection(_blob_path(path, shared_projection_root, digest), digest)
    return arrays
