"""Checkpoint behavior on synthetic vectors only; no trusted data is loaded."""
from pathlib import Path
import json
import random

import numpy as np
import pytest
import torch

from ssl_hdc_pilot.scripts.continual_hdc import HDCPrototypeLearner
from ssl_hdc_pilot.scripts.continual_baselines import (
    EWCLearner, HeadTrainingConfig, NaiveIncrementalLearner, ReplayLearner,
)
from ssl_hdc_pilot.scripts.continual_prototypes import (
    NCMPrototypeLearner,
    RanPACPrototypeLearner,
)
from ssl_hdc_pilot.scripts.aggregation_prototypes import (
    MedianNCMPrototypeLearner,
    SignNCMPrototypeLearner,
)
from ssl_hdc_pilot.scripts.learner_checkpoint import (
    load_learner_checkpoint,
    save_learner_checkpoint,
)


@pytest.mark.parametrize("shared", [False, True])
def test_hdc_checkpoint_preserves_ties_registry_and_continued_updates(tmp_path: Path, shared) -> None:
    original = HDCPrototypeLearner(input_dim=3, hypervector_dim=4, projection_seed=7)
    original.update_encoded("z", torch.tensor([-1., 1., -1., 1.]))
    original.update_encoded("z", torch.tensor([1., -1., 1., -1.]))
    original.update_encoded("a", torch.ones(4))
    original.register_class("unconfirmed")
    probe = torch.tensor([1., 2., -3.])
    expected = original.predict(probe)
    path = tmp_path / "learner.npz"

    root = tmp_path if shared else None
    save_learner_checkpoint(path, original, context={"data_role": "synthetic"}, shared_projection_root=root)
    checkpoint = load_learner_checkpoint(path, shared_projection_root=root)
    restored = checkpoint.learner

    assert checkpoint.context == {"data_role": "synthetic"}
    assert restored.known_classes == ("z", "a", "unconfirmed")
    assert restored.predict(probe) == expected
    torch.testing.assert_close(restored.get_prototype("z"), torch.tensor([-1., 1., -1., 1.]), rtol=0, atol=0)
    for label, vector in [("z", probe), ("unconfirmed", -probe), ("new", probe / 2)]:
        assert original.update(vector, label) == restored.update(vector, label)
        torch.testing.assert_close(original.get_prototype(label), restored.get_prototype(label), rtol=0, atol=0)
        assert original.predict(probe) == restored.predict(probe)


def _neural(kind: str, head_type: str = "linear"):
    config = HeadTrainingConfig(
        head_type=head_type, hidden_dim=5 if head_type == "mlp" else None,
        learning_rate=0.1, weight_decay=0.01, update_steps=2,
        ewc_lambda=10.0, ewc_decay=0.9, replay_batch_size=2,
    )
    kwargs = dict(input_dim=3, initial_classes=["z", "a"], learner_seed=31,
                  config=config, device="cpu")
    if kind == "replay":
        return ReplayLearner(**kwargs, replay_capacity=3, replay_seed=89)
    return {"naive": NaiveIncrementalLearner, "ewc": EWCLearner}[kind](**kwargs)


@pytest.mark.parametrize("kind", ["naive", "ewc", "replay"])
@pytest.mark.parametrize("head_type", ["linear", "mlp"])
@pytest.mark.parametrize("shared", [False, True])
def test_neural_checkpoint_reproduces_dynamic_classes_and_future_updates(tmp_path: Path, kind: str, head_type: str, shared) -> None:
    original = _neural(kind, head_type)
    vectors = torch.randn(28, 3, generator=torch.Generator().manual_seed(101))
    # Cross an EWC task boundary and overflow the replay reservoir before saving.
    for index, vector in enumerate(vectors[:12]):
        original.update(vector, ("z", "a", "novel")[index % 3])
    path = tmp_path / "neural.npz"
    rng_before = torch.get_rng_state().clone()
    root = tmp_path if shared else None
    save_learner_checkpoint(path, original, shared_projection_root=root)
    restored = load_learner_checkpoint(path, shared_projection_root=root).learner
    assert restored.model.training is original.model.training
    assert restored.config == original.config
    assert restored.known_classes == original.known_classes
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    torch.testing.assert_close(restored.parameter_vector(), original.parameter_vector(), rtol=0, atol=0)
    for index, vector in enumerate(vectors[12:]):
        assert original.predict(vector) == restored.predict(vector)
        label = ("a", "new_after_reload", "novel", "z")[index % 4]
        assert original.update(vector, label) == restored.update(vector, label)
        torch.testing.assert_close(restored.parameter_vector(), original.parameter_vector(), rtol=0, atol=0)
        if kind == "replay":
            assert restored.buffer.labels == original.buffer.labels
        assert original.predict(vector) == restored.predict(vector)


@pytest.mark.parametrize("kind", ["ncm", "ranpac"])
@pytest.mark.parametrize("shared", [False, True])
def test_prototype_checkpoint_preserves_means_counts_and_projection(tmp_path: Path, kind: str, shared) -> None:
    original = (NCMPrototypeLearner(input_dim=3) if kind == "ncm" else
                RanPACPrototypeLearner(input_dim=3, projection_dim=17, projection_seed=71))
    probe = torch.tensor([1., -2., 3.])
    for vector, label in [(probe, "z"), (-probe, "a"), (probe / 2, "z")]:
        original.update(vector, label)
    original.register_class("empty")
    path = tmp_path / "prototype.npz"
    root = tmp_path if shared else None
    save_learner_checkpoint(path, original, shared_projection_root=root)
    restored = load_learner_checkpoint(path, shared_projection_root=root).learner
    assert restored.known_classes == ("z", "a", "empty")
    for vector, label in [(torch.ones(3), "z"), (probe, "empty"), (-probe, "new")]:
        assert original.predict(vector) == restored.predict(vector)
        assert original.update(vector, label) == restored.update(vector, label)
        torch.testing.assert_close(original.get_prototype(label), restored.get_prototype(label), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["sign_ncm", "median_ncm"])
def test_native_aggregation_checkpoint_round_trip_is_projection_free(tmp_path: Path, kind: str) -> None:
    original = (SignNCMPrototypeLearner if kind == "sign_ncm" else MedianNCMPrototypeLearner)(input_dim=3)
    probe = torch.tensor([1., -2., 3.])
    for vector, label in [(probe, "z"), (-probe, "a"), (probe / 2, "z")]:
        original.update(vector, label)
    original.register_class("empty")
    path = tmp_path / "native.npz"
    save_learner_checkpoint(path, original, shared_projection_root=tmp_path)
    assert not (tmp_path / "projection_blobs").exists()
    restored = load_learner_checkpoint(path, shared_projection_root=tmp_path).learner
    assert type(restored) is type(original)
    assert restored.known_classes == original.known_classes
    for vector, label in [(torch.ones(3), "z"), (probe, "empty"), (-probe, "new")]:
        assert original.predict(vector) == restored.predict(vector)
        assert original.update(vector, label) == restored.update(vector, label)
        torch.testing.assert_close(restored.get_prototype(label), original.get_prototype(label), rtol=0, atol=0)


def test_rng_restore_is_explicit_and_checkpoint_is_detached(tmp_path: Path) -> None:
    original = _neural("replay")
    original.update(torch.tensor([1., 0., 0.]), "z")
    path = tmp_path / "rng.npz"
    save_learner_checkpoint(path, original)
    expected = (random.random(), np.random.random(), torch.rand(4))
    original.update(torch.ones(3), "new")
    restored = load_learner_checkpoint(path, restore_rng=True).learner
    assert restored.known_classes == ("z", "a")
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)


def test_checkpoint_refuses_overwrite_and_detects_corruption(tmp_path: Path) -> None:
    path = tmp_path / "immutable.npz"
    learner = NCMPrototypeLearner(input_dim=3)
    learner.update(torch.ones(3), "a")
    save_learner_checkpoint(path, learner)
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        save_learner_checkpoint(path, learner)
    assert path.read_bytes() == before
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["prototype_0"][0] += 1
    np.savez_compressed(tmp_path / "corrupted.npz", **payload)
    with pytest.raises(ValueError, match="integrity"):
        load_learner_checkpoint(tmp_path / "corrupted.npz")


def test_checkpoint_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "schema.npz"
    save_learner_checkpoint(path, NCMPrototypeLearner(input_dim=3))
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(payload["metadata_json"].tobytes())
    metadata["schema_version"] = 999
    payload["metadata_json"] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    np.savez_compressed(tmp_path / "future.npz", **payload)
    with pytest.raises(ValueError, match="schema_version"):
        load_learner_checkpoint(tmp_path / "future.npz")


def test_cpu_reload_does_not_queue_cuda_seeding(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "no_cuda_side_effect.npz"
    save_learner_checkpoint(path, _neural("naive"))

    def refuse_reseed(*args, **kwargs):
        raise AssertionError("CPU reload must not seed CUDA or enqueue deferred CUDA seeding")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", refuse_reseed)
    load_learner_checkpoint(path)


def test_checkpoint_preserves_mixed_modes_frozen_parameters_and_gradients(tmp_path: Path) -> None:
    original = _neural("naive", "mlp")
    original.model.eval()
    original.model.classifier.train()
    parameters = list(original.model.parameters())
    parameters[0].requires_grad_(False)
    parameters[-1].grad = torch.full_like(parameters[-1], 0.25)
    path = tmp_path / "modes.npz"
    save_learner_checkpoint(path, original)
    restored = load_learner_checkpoint(path).learner
    assert [(name, node.training) for name, node in restored.model.named_modules()] == [
        (name, node.training) for name, node in original.model.named_modules()
    ]
    for left, right in zip(original.model.parameters(), restored.model.parameters(), strict=True):
        assert left.requires_grad is right.requires_grad
        if left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
