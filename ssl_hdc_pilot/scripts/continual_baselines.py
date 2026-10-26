"""Continual neural-head baselines over the same frozen embeddings as HDC."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ssl_hdc_pilot.scripts.continual_hdc import Prediction


@dataclass(frozen=True)
class HeadTrainingConfig:
    head_type: str
    hidden_dim: int | None
    learning_rate: float
    weight_decay: float
    update_steps: int
    ewc_lambda: float
    ewc_decay: float
    replay_batch_size: int

    def validate(self) -> None:
        if self.head_type not in {"linear", "mlp"}:
            raise ValueError("head_type must be linear or mlp")
        if self.head_type == "mlp" and (self.hidden_dim is None or self.hidden_dim < 1):
            raise ValueError("MLP heads require a positive hidden_dim")
        if self.head_type == "linear" and self.hidden_dim is not None:
            raise ValueError("linear heads require hidden_dim=null")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.update_steps < 1:
            raise ValueError("invalid incremental optimizer settings")
        if self.ewc_lambda < 0 or not 0 <= self.ewc_decay <= 1:
            raise ValueError("invalid EWC settings")
        if self.replay_batch_size < 1:
            raise ValueError("replay_batch_size must be positive")


@dataclass(frozen=True)
class BaselineUpdateResult:
    class_registered: bool
    base_loss: float
    regularization_loss: float


class _ExpandableHead(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        config: HeadTrainingConfig,
        seed: int | None,
        device: torch.device,
    ) -> None:
        super().__init__()
        feature_dim = input_dim
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            if config.head_type == "mlp":
                assert config.hidden_dim is not None
                self.features = nn.Sequential(
                    nn.Linear(input_dim, config.hidden_dim),
                    nn.ReLU(),
                )
                feature_dim = config.hidden_dim
            else:
                self.features = nn.Identity()
            self.classifier = nn.Linear(feature_dim, output_dim, bias=False)
        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.similarity_scores(inputs)

    def similarity_scores(self, inputs: torch.Tensor) -> torch.Tensor:
        features = F.normalize(self.features(inputs), dim=-1)
        weights = F.normalize(self.classifier.weight, dim=-1)
        return features @ weights.T

    def expand_output(self, seed: int) -> None:
        old = self.classifier
        expanded = nn.Linear(
            old.in_features,
            old.out_features + 1,
            bias=False,
        ).to(old.weight.device)
        with torch.no_grad():
            expanded.weight[:-1].copy_(old.weight)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            bound = 1.0 / math.sqrt(old.in_features)
            new_weight = torch.empty(old.in_features).uniform_(
                -bound,
                bound,
                generator=generator,
            )
            expanded.weight[-1].copy_(new_weight.to(old.weight.device))
        self.classifier = expanded


class NaiveIncrementalLearner:
    """Fine-tune a linear or MLP head only on each confirmed example."""

    def __init__(
        self,
        *,
        input_dim: int,
        initial_classes: Sequence[str],
        learner_seed: int,
        config: HeadTrainingConfig,
        device: torch.device | str = "cpu",
        _checkpoint_restore: bool = False,
    ) -> None:
        config.validate()
        classes = [str(label) for label in initial_classes]
        if not classes or len(classes) != len(set(classes)) or any(not label for label in classes):
            raise ValueError("initial_classes must contain unique non-empty labels")
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.config = config
        self.device = torch.device(device)
        self._seed = int(learner_seed)
        self._classes = classes
        self._class_to_index = {label: index for index, label in enumerate(classes)}
        self.model = _ExpandableHead(
            input_dim=self.input_dim,
            output_dim=len(classes),
            config=config,
            # The persistence adapter immediately replaces every weight. Avoid
            # queuing a CUDA seed during CPU reload; ordinary construction keeps
            # the registered initialization path unchanged.
            seed=None if _checkpoint_restore else self._seed,
            device=self.device,
        )

    @property
    def known_classes(self) -> tuple[str, ...]:
        return tuple(self._classes)

    def register_class(self, label: str) -> bool:
        label = str(label)
        if not label:
            raise ValueError("class labels must be non-empty")
        if label in self._class_to_index:
            return False
        self.model.expand_output(self._seed + len(self._classes))
        self._class_to_index[label] = len(self._classes)
        self._classes.append(label)
        self._after_class_expansion()
        return True

    def _after_class_expansion(self) -> None:
        pass

    def _tensor(self, embedding: torch.Tensor) -> torch.Tensor:
        vector = torch.as_tensor(embedding, dtype=torch.float32, device=self.device)
        if vector.ndim != 1 or vector.numel() != self.input_dim:
            raise ValueError(
                f"expected one {self.input_dim}-d embedding, got {tuple(vector.shape)}"
            )
        return vector

    def predict(self, embedding: torch.Tensor) -> Prediction:
        vector = self._tensor(embedding).unsqueeze(0)
        self.model.eval()
        with torch.inference_mode():
            scores = self.model.similarity_scores(vector).squeeze(0)
            probabilities = torch.softmax(scores, dim=0)
        top_index = int(torch.argmax(scores).item())
        margin = 0.0
        if len(self._classes) >= 2:
            top_two = torch.topk(scores, k=2).values
            margin = float((top_two[0] - top_two[1]).item())
        return Prediction(
            label=self._classes[top_index],
            confidence=margin,
            top_probability=float(probabilities[top_index].item()),
        )

    def _training_batch(
        self, embedding: torch.Tensor, label: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return embedding.unsqueeze(0), torch.tensor(
            [self._class_to_index[label]],
            dtype=torch.long,
            device=self.device,
        )

    def _regularization_penalty(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=self.device)

    def _after_update(self, embedding: torch.Tensor, label: str) -> None:
        del embedding, label

    def _non_finite_loss_message(self) -> str:
        return "non-finite continual-head update loss"

    def update(self, embedding: torch.Tensor, label: str) -> BaselineUpdateResult:
        vector = self._tensor(embedding)
        registered = self.register_class(label)
        inputs, targets = self._training_batch(vector, label)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        base_loss_value = 0.0
        regularization_value = 0.0
        self.model.train()
        for _ in range(self.config.update_steps):
            optimizer.zero_grad(set_to_none=True)
            base_loss = F.cross_entropy(self.model(inputs), targets)
            regularization = self._regularization_penalty()
            loss = base_loss + regularization
            if not torch.isfinite(loss):
                raise RuntimeError(self._non_finite_loss_message())
            loss.backward()
            optimizer.step()
            base_loss_value = float(base_loss.detach().item())
            regularization_value = float(regularization.detach().item())
        self._after_update(vector, label)
        return BaselineUpdateResult(
            class_registered=registered,
            base_loss=base_loss_value,
            regularization_loss=regularization_value,
        )

    def parameter_vector(self) -> torch.Tensor:
        return torch.cat(
            [parameter.detach().reshape(-1).cpu() for parameter in self.model.parameters()]
        )


class EWCLearner(NaiveIncrementalLearner):
    """Online EWC with anchors consolidated at novel-class task boundaries."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._ewc_means: dict[str, torch.Tensor] = {}
        self._ewc_penalty_fisher: dict[str, torch.Tensor] = {}
        self._ewc_running_fisher: dict[str, torch.Tensor] = {}

    def _after_class_expansion(self) -> None:
        if not hasattr(self, "_ewc_running_fisher"):
            return
        for name, parameter in self.model.named_parameters():
            running_fisher = self._ewc_running_fisher.get(name)
            if running_fisher is None:
                running_fisher = torch.zeros_like(parameter)
            elif running_fisher.shape != parameter.shape:
                expanded = torch.zeros_like(parameter)
                slices = tuple(slice(0, size) for size in running_fisher.shape)
                expanded[slices] = running_fisher
                running_fisher = expanded
            self._ewc_running_fisher[name] = running_fisher

        self._ewc_means = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        self._ewc_penalty_fisher = {
            name: fisher.detach().clone()
            for name, fisher in self._ewc_running_fisher.items()
        }

    def _regularization_penalty(self) -> torch.Tensor:
        if not self._ewc_means:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        penalty = torch.zeros((), dtype=torch.float32, device=self.device)
        for name, parameter in self.model.named_parameters():
            penalty = penalty + torch.sum(
                self._ewc_penalty_fisher[name]
                * (parameter - self._ewc_means[name]).pow(2)
            )
        return self.config.ewc_lambda * penalty

    def _non_finite_loss_message(self) -> str:
        return (
            "non-finite EWC update loss "
            f"(ewc_decay={self.config.ewc_decay}, "
            f"ewc_lambda={self.config.ewc_lambda})"
        )

    def _after_update(self, embedding: torch.Tensor, label: str) -> None:
        self.model.zero_grad(set_to_none=True)
        target = torch.tensor(
            [self._class_to_index[label]],
            dtype=torch.long,
            device=self.device,
        )
        loss = F.cross_entropy(self.model(embedding.unsqueeze(0)), target)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            gradient = parameter.grad
            observed = torch.zeros_like(parameter) if gradient is None else gradient.detach().pow(2)
            if name in self._ewc_running_fisher:
                observed = (
                    self.config.ewc_decay * self._ewc_running_fisher[name]
                    + observed
                )
            self._ewc_running_fisher[name] = observed
        self.model.zero_grad(set_to_none=True)


class ReservoirBuffer:
    """Seeded Algorithm-R reservoir over confirmed embedding/label pairs."""

    def __init__(self, *, capacity: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("replay reservoir capacity must be positive")
        self.capacity = int(capacity)
        self._random = random.Random(seed)
        self._items: list[tuple[torch.Tensor, str]] = []
        self._seen = 0

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(label for _, label in self._items)

    def add(self, embedding: torch.Tensor, label: str) -> None:
        item = (torch.as_tensor(embedding, dtype=torch.float32).detach().cpu().clone(), str(label))
        self._seen += 1
        if len(self._items) < self.capacity:
            self._items.append(item)
            return
        replacement = self._random.randrange(self._seen)
        if replacement < self.capacity:
            self._items[replacement] = item

    def sample(self, size: int) -> list[tuple[torch.Tensor, str]]:
        if size < 1:
            raise ValueError("replay sample size must be positive")
        if not self._items:
            return []
        indices = self._random.sample(range(len(self._items)), k=min(size, len(self._items)))
        return [(self._items[index][0].clone(), self._items[index][1]) for index in indices]


class ReplayLearner(NaiveIncrementalLearner):
    """Incremental head updated with the current label plus reservoir replay."""

    def __init__(self, *, replay_capacity: int, replay_seed: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.buffer = ReservoirBuffer(capacity=replay_capacity, seed=replay_seed)

    def _training_batch(
        self, embedding: torch.Tensor, label: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        replay = self.buffer.sample(self.config.replay_batch_size)
        embeddings = [embedding, *[item[0].to(self.device) for item in replay]]
        labels = [label, *[item[1] for item in replay]]
        return torch.stack(embeddings), torch.tensor(
            [self._class_to_index[item] for item in labels],
            dtype=torch.long,
            device=self.device,
        )

    def _after_update(self, embedding: torch.Tensor, label: str) -> None:
        self.buffer.add(embedding, label)
