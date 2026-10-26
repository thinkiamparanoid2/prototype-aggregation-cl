"""Backprop-free online prototype baselines over frozen embeddings."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ssl_hdc_pilot.scripts.continual_hdc import Prediction, UpdateResult


class _OnlineCosinePrototypeLearner:
    def __init__(
        self,
        *,
        input_dim: int,
        prototype_dim: int,
        device: torch.device | str,
    ) -> None:
        if input_dim < 1 or prototype_dim < 1:
            raise ValueError("input and prototype dimensions must be positive")
        self.input_dim = int(input_dim)
        self.prototype_dim = int(prototype_dim)
        self.device = torch.device(device)
        self._counts: dict[str, int] = {}
        self._prototypes: dict[str, torch.Tensor] = {}

    @property
    def known_classes(self) -> tuple[str, ...]:
        return tuple(self._prototypes)

    def register_class(self, label: str) -> bool:
        label = str(label)
        if not label:
            raise ValueError("class labels must be non-empty")
        if label in self._prototypes:
            return False
        self._counts[label] = 0
        self._prototypes[label] = torch.zeros(
            self.prototype_dim,
            dtype=torch.float32,
            device=self.device,
        )
        return True

    def _embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        vector = torch.as_tensor(embedding, dtype=torch.float32, device=self.device)
        if vector.ndim != 1 or vector.numel() != self.input_dim:
            raise ValueError(
                f"expected one {self.input_dim}-d embedding, got {tuple(vector.shape)}"
            )
        return vector

    def _transform(self, embedding: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def update(self, embedding: torch.Tensor, label: str) -> UpdateResult:
        transformed = self._transform(embedding)
        registered = self.register_class(label)
        count = self._counts[label] + 1
        prototype = self._prototypes[label]
        prototype.add_((transformed - prototype) / count)
        self._counts[label] = count
        return UpdateResult(class_registered=registered)

    def get_prototype(self, label: str) -> torch.Tensor:
        try:
            count = self._counts[label]
            prototype = self._prototypes[label]
        except KeyError as error:
            raise KeyError(f"unknown class: {label}") from error
        if count == 0:
            raise RuntimeError(f"class has no confirmed examples: {label}")
        return prototype.detach().clone()

    def predict(self, embedding: torch.Tensor) -> Prediction:
        active_labels = [
            label for label in self._prototypes if self._counts[label] > 0
        ]
        if not active_labels:
            return Prediction(label=None, confidence=0.0, top_probability=0.0)

        query = self._transform(embedding)
        prototypes = torch.stack(
            [self._prototypes[label] for label in active_labels]
        )
        similarities = F.cosine_similarity(query.unsqueeze(0), prototypes, dim=1)
        top_index = int(torch.argmax(similarities).item())
        probabilities = torch.softmax(similarities, dim=0)
        if len(active_labels) < 2:
            margin = 0.0
        else:
            top_two = torch.topk(similarities, k=2).values
            margin = float((top_two[0] - top_two[1]).item())
        return Prediction(
            label=active_labels[top_index],
            confidence=margin,
            top_probability=float(probabilities[top_index].item()),
        )


class NCMPrototypeLearner(_OnlineCosinePrototypeLearner):
    """Online nearest-class-mean over L2-normalized embeddings.

    Each raw frozen embedding is L2-normalized before the streaming-mean
    recurrence. This is the single NCM variant in the locked 36-cell grid.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            prototype_dim=input_dim,
            device=device,
        )

    def _transform(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self._embedding(embedding), dim=0)


class RanPACPrototypeLearner(_OnlineCosinePrototypeLearner):
    """RanPAC-style frozen random-projection prototypes.

    Design source: McDonnell et al., "RanPAC: Random Projections and
    Pre-trained Models for Continual Learning", NeurIPS 2023.

    This controlled ablation implements the paper's fixed Gaussian projection,
    ReLU nonlinearity, and online class-prototype accumulation. It deliberately
    omits Gram-matrix decorrelation and ridge inversion so the comparison with
    HDC isolates continuous ReLU projection/prototype algebra from HDC's
    bipolar encoding and re-binarized bundling.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        projection_seed: int,
        projection_dim: int,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            prototype_dim=projection_dim,
            device=device,
        )
        self.projection_dim = int(projection_dim)
        generator = torch.Generator(device="cpu").manual_seed(projection_seed)
        projection = torch.randn(
            self.input_dim,
            self.projection_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self._projection = projection.to(self.device)

    def project(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.relu(self._embedding(embedding) @ self._projection)

    def _transform(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.project(embedding)
