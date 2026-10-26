"""Synthetic storage tests: no datasets, evidence runs or completed artifacts."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from ssl_hdc_pilot.scripts import checkpoint_storage as storage
from ssl_hdc_pilot.scripts import learner_checkpoint as checkpoints
from ssl_hdc_pilot.scripts import benchmark_core
from ssl_hdc_pilot.scripts.continual_hdc import HDCPrototypeLearner


def payload():
    return {"projection": np.arange(12, dtype=np.float32).reshape(3, 4),
            "state": np.array([1, 2, 3], dtype=np.int64)}


def assert_exact(left, right):
    assert set(left) == set(right)
    for key in left:
        assert left[key].shape == right[key].shape and left[key].dtype == right[key].dtype
        assert left[key].tobytes() == right[key].tobytes()


def test_legacy_and_shared_are_exact_and_bundle_is_portable(tmp_path):
    root = tmp_path / "run"
    arrays = payload()
    storage.save_arrays(root / "legacy.npz", arrays)
    for filename in ("a/one.npz", "b/snapshots/two.npz"):
        storage.save_arrays(root / filename, arrays, shared_projection_root=root)
        with np.load(root / filename, allow_pickle=False) as archive:
            assert "projection" not in archive.files
            assert storage.STORAGE_KEY in archive.files
        assert_exact(arrays, storage.load_arrays(root / filename, shared_projection_root=root))
    assert len(list((root / "projection_blobs").glob("*.npz"))) == 1
    assert_exact(arrays, storage.load_arrays(root / "legacy.npz"))
    moved = tmp_path / "copied"
    shutil.copytree(root, moved)
    shutil.rmtree(root)  # Only this synthetic test's private directory.
    assert_exact(arrays, storage.load_arrays(moved / "b/snapshots/two.npz", shared_projection_root=moved))
    with pytest.raises(ValueError, match="explicit shared_projection_root"):
        storage.load_arrays(moved / "a/one.npz")
    with pytest.raises(ValueError, match="escapes"):
        storage.load_arrays(moved / "a/one.npz", shared_projection_root=tmp_path / "wrong")


def test_shape_and_content_changes_produce_distinct_blobs(tmp_path):
    for index, projection in enumerate((payload()["projection"], payload()["projection"].reshape(4, 3),
                                        payload()["projection"] + 1)):
        storage.save_arrays(tmp_path / f"{index}.npz", {"projection": projection}, shared_projection_root=tmp_path)
    assert len(list((tmp_path / "projection_blobs").glob("*.npz"))) == 3
    with pytest.raises(ValueError, match="float32"):
        storage.save_arrays(tmp_path / "wrong_dtype.npz", {"projection": np.ones((3, 4), dtype=np.float64)},
                            shared_projection_root=tmp_path)


@pytest.mark.parametrize("problem", ["missing", "corrupt", "changed", "extra_array"])
def test_missing_or_corrupt_blob_fails_before_learner_or_rng_restore(tmp_path, monkeypatch, problem):
    path = tmp_path / "state.npz"
    learner = HDCPrototypeLearner(input_dim=3, hypervector_dim=4, projection_seed=7)
    checkpoints.save_learner_checkpoint(path, learner, shared_projection_root=tmp_path)
    blob = next((tmp_path / "projection_blobs").glob("*.npz"))
    if problem == "missing":
        blob.unlink()
    elif problem == "corrupt":
        blob.write_bytes(b"broken")
    else:
        arrays = storage.load_arrays(blob)
        if problem == "changed":
            arrays["projection"][0, 0] += 1
        else:
            arrays["extra"] = np.ones(1)
        np.savez_compressed(blob, **arrays)
    before = torch.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before learner construction")
    monkeypatch.setattr(checkpoints, "_restore_learner", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        checkpoints.load_learner_checkpoint(path, shared_projection_root=tmp_path, restore_rng=True)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)


@pytest.mark.parametrize("problem", ["version", "traversal", "absolute", "extra_key", "embedded_projection"])
def test_invalid_storage_envelope_is_rejected(tmp_path, problem):
    path = tmp_path / "state.npz"
    storage.save_arrays(path, payload(), shared_projection_root=tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    record = json.loads(arrays[storage.STORAGE_KEY].tobytes())
    if problem == "version":
        record["version"] = 999
    elif problem in {"traversal", "absolute"}:
        record["projection_sha256"] = "../outside" if problem == "traversal" else "C:/outside"
    elif problem == "extra_key":
        record["path"] = "../outside"
    else:
        arrays["projection"] = payload()["projection"]
    arrays[storage.STORAGE_KEY] = np.frombuffer(json.dumps(record).encode(), dtype=np.uint8)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        storage.load_arrays(path, shared_projection_root=tmp_path)
    with pytest.raises(ValueError):
        storage.projection_dependencies(path, shared_projection_root=tmp_path)


def test_existing_corrupt_blob_is_never_replaced(tmp_path):
    storage.save_arrays(tmp_path / "one.npz", payload(), shared_projection_root=tmp_path)
    blob = next((tmp_path / "projection_blobs").glob("*.npz"))
    blob.write_bytes(b"preserve corruption evidence")
    with pytest.raises(ValueError):
        storage.save_arrays(tmp_path / "two.npz", payload(), shared_projection_root=tmp_path)
    assert blob.read_bytes() == b"preserve corruption evidence"
    assert not (tmp_path / "two.npz").exists()


def test_serialization_failure_leaves_no_partial_checkpoint(tmp_path, monkeypatch):
    def fail(handle, **arrays):
        handle.write(b"partial")
        raise OSError("simulated full disk")
    monkeypatch.setattr(storage.np, "savez_compressed", fail)
    with pytest.raises(OSError, match="full disk"):
        storage.save_arrays(tmp_path / "state.npz", payload())
    assert list(tmp_path.iterdir()) == []


def test_atomic_publish_race_refuses_overwrite(tmp_path, monkeypatch):
    path = tmp_path / "state.npz"
    original_link = storage.os.link
    def race(source, destination):
        destination.write_bytes(b"other writer won")
        return original_link(source, destination)
    monkeypatch.setattr(storage.os, "link", race)
    with pytest.raises(FileExistsError):
        storage.save_arrays(path, payload())
    assert path.read_bytes() == b"other writer won"
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_saves_share_one_verified_blob(tmp_path):
    def save(index):
        storage.save_arrays(tmp_path / f"{index}.npz", payload(), shared_projection_root=tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(save, range(8)))
    assert len(list((tmp_path / "projection_blobs").glob("*.npz"))) == 1
    for index in range(8):
        assert_exact(payload(), storage.load_arrays(tmp_path / f"{index}.npz", shared_projection_root=tmp_path))
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.skipif(os.name != "nt", reason="Windows extended namespace spelling")
@pytest.mark.parametrize("escape", [False, True])
def test_extended_windows_namespace_is_normalized_without_waiving_escape(tmp_path, monkeypatch, escape):
    root = tmp_path / "run"
    root.mkdir()
    path = root / "projection_blobs" / ("a" * 64 + ".npz")
    original_resolve = Path.resolve
    def resolve(value, *args, **kwargs):
        if value == path:
            target = tmp_path / "outside.npz" if escape else path
            return Path("\\\\?\\" + str(target))
        return original_resolve(value, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", resolve)
    if escape:
        with pytest.raises(ValueError, match="escapes"):
            storage._inside_root(path, root)
    else:
        assert storage._inside_root(path, root) == Path("\\\\?\\" + str(path))


def test_prediction_export_shares_blob_but_is_not_a_resumable_checkpoint(tmp_path):
    learner = HDCPrototypeLearner(input_dim=3, hypervector_dim=4, projection_seed=7)
    learner.update(torch.ones(3), "a")
    before = benchmark_core.payload_report(learner)
    checkpoints.save_learner_checkpoint(tmp_path / "full.npz", learner, shared_projection_root=tmp_path)
    benchmark_core.save_prediction_payload(tmp_path / "prediction.npz", learner, shared_projection_root=tmp_path)
    assert len(list((tmp_path / "projection_blobs").glob("*.npz"))) == 1
    assert benchmark_core.payload_report(learner) == before
    with pytest.raises(ValueError, match="state_kind"):
        checkpoints.load_learner_checkpoint(tmp_path / "prediction.npz", shared_projection_root=tmp_path)
