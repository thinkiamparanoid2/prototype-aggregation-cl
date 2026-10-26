"""Backprop-free continual HDC prototype learning on frozen embeddings."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torchhd


HYPERVECTOR_DIM = 10_000


@dataclass(frozen=True)
class Prediction:
    label: str | None
    confidence: float
    top_probability: float


@dataclass(frozen=True)
class UpdateResult:
    class_registered: bool


class HDCPrototypeLearner:
    """Fixed random-projection HDC with bundled bipolar class prototypes.

    Each confirmed example is encoded once and added to a real-valued class
    accumulator. The classification prototype is re-binarized after every
    addition. When an accumulator component is exactly zero, its previous bit
    is retained; the first update cannot tie because encoded vectors are
    bipolar. This preserves evidence across updates without epochs or
    backpropagation.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        projection_seed: int,
        hypervector_dim: int = HYPERVECTOR_DIM,
        device: torch.device | str = "cpu",
    ) -> None:
        if input_dim < 1 or hypervector_dim < 1:
            raise ValueError("input and hypervector dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hypervector_dim = int(hypervector_dim)
        self.device = torch.device(device)
        generator = torch.Generator(device="cpu").manual_seed(projection_seed)
        projection = torch.randn(
            self.input_dim,
            self.hypervector_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self._projection = projection.to(self.device)
        self._accumulators: dict[str, torch.Tensor] = {}
        self._prototypes: dict[str, torch.Tensor | None] = {}

    @property
    def known_classes(self) -> tuple[str, ...]:
        return tuple(self._prototypes)

    def register_class(self, label: str) -> bool:
        label = str(label)
        if not label:
            raise ValueError("class labels must be non-empty")
        if label in self._prototypes:
            return False
        self._accumulators[label] = torch.zeros(
            self.hypervector_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self._prototypes[label] = None
        return True

    def encode(self, embedding: torch.Tensor) -> torch.Tensor:
        vector = torch.as_tensor(embedding, dtype=torch.float32, device=self.device)
        if vector.ndim != 1 or vector.numel() != self.input_dim:
            raise ValueError(
                f"expected one {self.input_dim}-d embedding, got {tuple(vector.shape)}"
            )
        projected = vector @ self._projection
        encoded = torchhd.functional.normalize(projected)
        return torch.where(encoded == 0, torch.ones_like(encoded), encoded).float()

    def update_encoded(self, label: str, encoded: torch.Tensor) -> UpdateResult:
        hypervector = torch.as_tensor(encoded, dtype=torch.float32, device=self.device)
        if hypervector.ndim != 1 or hypervector.numel() != self.hypervector_dim:
            raise ValueError(
                f"expected one {self.hypervector_dim}-d hypervector, "
                f"got {tuple(hypervector.shape)}"
            )
        if not torch.all((hypervector == -1) | (hypervector == 1)):
            raise ValueError("encoded hypervectors must be bipolar (-1 or +1)")

        registered = self.register_class(label)
        previous = self._prototypes[label]
        accumulator = self._accumulators[label]
        accumulator.add_(hypervector)
        rebinarized = torchhd.functional.normalize(accumulator)
        if previous is not None:
            rebinarized = torch.where(accumulator == 0, previous, rebinarized)
        rebinarized = torch.where(
            rebinarized == 0,
            torch.ones_like(rebinarized),
            rebinarized,
        ).float()
        self._prototypes[label] = rebinarized
        return UpdateResult(class_registered=registered)

    def update(self, embedding: torch.Tensor, label: str) -> UpdateResult:
        return self.update_encoded(label, self.encode(embedding))

    def get_prototype(self, label: str) -> torch.Tensor:
        try:
            prototype = self._prototypes[label]
        except KeyError as error:
            raise KeyError(f"unknown class: {label}") from error
        if prototype is None:
            raise RuntimeError(f"class has no confirmed examples: {label}")
        return prototype.detach().clone()

    def predict(self, embedding: torch.Tensor) -> Prediction:
        active = [
            (label, prototype)
            for label, prototype in self._prototypes.items()
            if prototype is not None
        ]
        if not active:
            return Prediction(label=None, confidence=0.0, top_probability=0.0)

        query = self.encode(embedding)
        labels = [label for label, _ in active]
        prototypes = torch.stack([prototype for _, prototype in active])
        similarities = torchhd.functional.cosine_similarity(query, prototypes).float()
        top_index = int(torch.argmax(similarities).item())
        probabilities = torch.softmax(similarities, dim=0)
        if len(labels) < 2:
            margin = 0.0
        else:
            top_two = torch.topk(similarities, k=2).values
            margin = float((top_two[0] - top_two[1]).item())
        return Prediction(
            label=labels[top_index],
            confidence=margin,
            top_probability=float(probabilities[top_index].item()),
        )
