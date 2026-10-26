from __future__ import annotations

import math

import pytest
import torch

from ssl_hdc_pilot.scripts.continual_baselines import (
    EWCLearner,
    HeadTrainingConfig,
    NaiveIncrementalLearner,
    ReservoirBuffer,
)


def _config(*, ewc_lambda: float = 0.0) -> HeadTrainingConfig:
    return HeadTrainingConfig(
        head_type="linear",
        hidden_dim=None,
        learning_rate=0.2,
        weight_decay=0.0,
        update_steps=8,
        ewc_lambda=ewc_lambda,
        ewc_decay=1.0,
        replay_batch_size=2,
    )


def test_ewc_consolidates_at_a_novel_class_boundary() -> None:
    config = HeadTrainingConfig(
        head_type="linear",
        hidden_dim=None,
        learning_rate=0.2,
        weight_decay=0.0,
        update_steps=1,
        ewc_lambda=10.0,
        ewc_decay=1.0,
        replay_batch_size=2,
    )
    naive = NaiveIncrementalLearner(
        input_dim=2,
        initial_classes=["a", "b"],
        learner_seed=23,
        config=config,
    )
    ewc = EWCLearner(
        input_dim=2,
        initial_classes=["a", "b"],
        learner_seed=23,
        config=config,
    )

    pre_boundary = (
        (torch.tensor([1.0, 0.0]), "a"),
        (torch.tensor([-1.0, 0.0]), "b"),
    )
    for _ in range(6):
        for embedding, label in pre_boundary:
            naive.update(embedding, label)
            ewc.update(embedding, label)

    boundary_embedding = torch.tensor([0.0, 1.0])
    naive.update(boundary_embedding, "novel_c")
    ewc.update(boundary_embedding, "novel_c")

    regularization_losses = []
    for _ in range(8):
        naive.update(boundary_embedding, "novel_c")
        regularization_losses.append(
            ewc.update(boundary_embedding, "novel_c").regularization_loss
        )

    assert any(loss > 0.0 for loss in regularization_losses)
    assert not torch.equal(ewc.parameter_vector(), naive.parameter_vector())

    probe_predictions_differ = False
    for degrees in range(360):
        radians = math.radians(degrees)
        probe = torch.tensor([math.cos(radians), math.sin(radians)])
        if ewc.predict(probe).label != naive.predict(probe).label:
            probe_predictions_differ = True
            break
    assert probe_predictions_differ


def test_ewc_penalty_constrains_a_conflicting_incremental_update() -> None:
    naive = NaiveIncrementalLearner(
        input_dim=2,
        initial_classes=["a", "b"],
        learner_seed=23,
        config=_config(),
    )
    ewc = EWCLearner(
        input_dim=2,
        initial_classes=["a", "b"],
        learner_seed=23,
        config=_config(ewc_lambda=10.0),
    )
    for learner in (naive, ewc):
        learner.update(torch.tensor([1.0, 0.0]), "a")
        learner.update(torch.tensor([-1.0, 0.0]), "b")
        learner.update(torch.tensor([0.0, 1.0]), "novel_c")

    naive_before = naive.parameter_vector()
    ewc_before = ewc.parameter_vector()
    naive.update(torch.tensor([-1.0, 0.0]), "a")
    ewc_result = ewc.update(torch.tensor([-1.0, 0.0]), "a")

    naive_drift = torch.linalg.vector_norm(naive.parameter_vector() - naive_before)
    ewc_drift = torch.linalg.vector_norm(ewc.parameter_vector() - ewc_before)
    assert ewc_result.regularization_loss > 0.0
    assert ewc_drift < naive_drift


def test_online_fisher_decay_bounds_repeated_observations() -> None:
    def learner(decay: float) -> EWCLearner:
        return EWCLearner(
            input_dim=2,
            initial_classes=["a", "b"],
            learner_seed=23,
            config=HeadTrainingConfig(
                head_type="linear",
                hidden_dim=None,
                learning_rate=0.1,
                weight_decay=0.0,
                update_steps=1,
                ewc_lambda=10.0,
                ewc_decay=decay,
                replay_batch_size=2,
            ),
        )

    def fisher_total(learner: EWCLearner) -> float:
        return sum(
            float(fisher.sum().item())
            for fisher in learner._ewc_running_fisher.values()
        )

    cumulative = learner(1.0)
    bounded = learner(0.9)
    embedding = torch.tensor([1.0, 0.0])

    cumulative._after_update(embedding, "a")
    bounded._after_update(embedding, "a")
    one_observation = fisher_total(bounded)
    assert one_observation > 0.0

    steps = 200
    for _ in range(steps - 1):
        cumulative._after_update(embedding, "a")
        bounded._after_update(embedding, "a")

    cumulative_total = fisher_total(cumulative)
    bounded_total = fisher_total(bounded)
    geometric_ceiling = one_observation / (1.0 - 0.9)

    assert cumulative_total > one_observation * (steps - 1)
    assert bounded_total <= geometric_ceiling * 1.000001
    assert cumulative_total > bounded_total * 10.0


def test_non_finite_ewc_loss_names_decay_and_lambda() -> None:
    learner = EWCLearner(
        input_dim=2,
        initial_classes=["a", "b"],
        learner_seed=23,
        config=HeadTrainingConfig(
            head_type="linear",
            hidden_dim=None,
            learning_rate=0.1,
            weight_decay=0.0,
            update_steps=1,
            ewc_lambda=10.0,
            ewc_decay=0.9,
            replay_batch_size=2,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=r"non-finite EWC update loss.*ewc_decay=0.9.*ewc_lambda=10.0",
    ):
        learner.update(torch.tensor([float("nan"), 0.0]), "a")


def test_reservoir_retention_and_sampling_are_seed_deterministic() -> None:
    first = ReservoirBuffer(capacity=3, seed=101)
    second = ReservoirBuffer(capacity=3, seed=101)
    for index in range(10):
        embedding = torch.tensor([float(index), -float(index)])
        first.add(embedding, f"class_{index}")
        second.add(embedding, f"class_{index}")

    assert first.labels == second.labels
    first_sample = first.sample(2)
    second_sample = second.sample(2)
    assert [label for _, label in first_sample] == [label for _, label in second_sample]
    for (left, _), (right, _) in zip(first_sample, second_sample, strict=True):
        torch.testing.assert_close(left, right)


def test_mlp_head_expands_on_a_confirmed_novel_label() -> None:
    learner = NaiveIncrementalLearner(
        input_dim=4,
        initial_classes=["known_a", "known_b"],
        learner_seed=31,
        config=HeadTrainingConfig(
            head_type="mlp",
            hidden_dim=3,
            learning_rate=0.1,
            weight_decay=0.0,
            update_steps=1,
            ewc_lambda=1.0,
            ewc_decay=1.0,
            replay_batch_size=1,
        ),
    )

    update = learner.update(torch.tensor([0.0, 1.0, 0.0, 1.0]), "novel_c")

    assert update.class_registered is True
    assert learner.known_classes == ("known_a", "known_b", "novel_c")
    assert learner.predict(torch.ones(4)).label in learner.known_classes
