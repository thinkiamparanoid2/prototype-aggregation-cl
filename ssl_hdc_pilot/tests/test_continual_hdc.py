from __future__ import annotations

import torch

from ssl_hdc_pilot.scripts.continual_hdc import HDCPrototypeLearner


def test_confirmed_examples_bundle_then_rebinarize_with_stable_ties() -> None:
    learner = HDCPrototypeLearner(input_dim=2, projection_seed=7)
    positive = torch.ones(10_000)
    half_negative = torch.cat((torch.full((5_000,), -1.0), torch.ones(5_000)))

    learner.update_encoded("known", positive)
    learner.update_encoded("known", half_negative)
    torch.testing.assert_close(learner.get_prototype("known"), positive)

    learner.update_encoded("known", half_negative)
    torch.testing.assert_close(learner.get_prototype("known"), half_negative)


def test_first_confirmed_novel_label_registers_a_new_class() -> None:
    learner = HDCPrototypeLearner(input_dim=384, projection_seed=11)

    update = learner.update(torch.ones(384), "new_disease")

    assert update.class_registered is True
    assert learner.known_classes == ("new_disease",)
    assert set(learner.get_prototype("new_disease").tolist()) == {-1.0, 1.0}
