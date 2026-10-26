from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from ssl_hdc_pilot.scripts.continual_hdc import HDCPrototypeLearner
from ssl_hdc_pilot.scripts.continual_prototypes import (
    NCMPrototypeLearner,
    RanPACPrototypeLearner,
)


def test_ncm_running_mean_equals_the_batch_mean_of_normalized_embeddings() -> None:
    learner = NCMPrototypeLearner(input_dim=3)
    examples = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )

    for example in examples:
        learner.update(example, "disease_a")

    torch.testing.assert_close(
        learner.get_prototype("disease_a"),
        F.normalize(examples, dim=1).mean(dim=0),
    )


def test_ranpac_projection_is_deterministic_under_its_seed() -> None:
    first = RanPACPrototypeLearner(
        input_dim=3,
        projection_seed=17,
        projection_dim=32,
    )
    second = RanPACPrototypeLearner(
        input_dim=3,
        projection_seed=17,
        projection_dim=32,
    )
    different = RanPACPrototypeLearner(
        input_dim=3,
        projection_seed=18,
        projection_dim=32,
    )
    embedding = torch.tensor([1.0, -2.0, 3.0])

    torch.testing.assert_close(first.project(embedding), second.project(embedding))
    assert not torch.equal(first.project(embedding), different.project(embedding))
    assert torch.all(first.project(embedding) >= 0)


@pytest.mark.parametrize(
    "learner",
    [
        NCMPrototypeLearner(input_dim=2),
        RanPACPrototypeLearner(
            input_dim=2,
            projection_seed=19,
            projection_dim=16,
        ),
    ],
)
def test_prototype_learners_register_on_first_confirmed_novel_label(learner: object) -> None:
    first = learner.update(torch.tensor([1.0, 0.0]), "known_a")
    repeated = learner.update(torch.tensor([2.0, 0.0]), "known_a")
    novel = learner.update(torch.tensor([0.0, 1.0]), "novel_b")

    assert first.class_registered is True
    assert repeated.class_registered is False
    assert novel.class_registered is True
    assert learner.known_classes == ("known_a", "novel_b")


def _assert_cosine_margin(learner: object, query: torch.Tensor) -> None:
    prediction = learner.predict(query)
    labels = learner.known_classes
    if isinstance(learner, HDCPrototypeLearner):
        comparison_query = learner.encode(query)
    elif isinstance(learner, RanPACPrototypeLearner):
        comparison_query = learner.project(query)
    else:
        comparison_query = query
    prototypes = torch.stack([learner.get_prototype(label) for label in labels])
    similarities = F.cosine_similarity(comparison_query.unsqueeze(0), prototypes, dim=1)
    top_two = torch.topk(similarities, k=2).values

    assert prediction.confidence == pytest.approx(float(top_two[0] - top_two[1]))


def test_prototype_confidence_uses_the_same_cosine_margin_semantics_as_hdc() -> None:
    hdc = HDCPrototypeLearner(
        input_dim=2,
        projection_seed=19,
        hypervector_dim=16,
    )
    ncm = NCMPrototypeLearner(input_dim=2)
    ranpac = RanPACPrototypeLearner(
        input_dim=2,
        projection_seed=19,
        projection_dim=16,
    )
    for learner in (hdc, ncm, ranpac):
        learner.update(torch.tensor([2.0, 0.0]), "class_a")
        learner.update(torch.tensor([0.0, 2.0]), "class_b")
        _assert_cosine_margin(learner, torch.tensor([1.5, 0.5]))
