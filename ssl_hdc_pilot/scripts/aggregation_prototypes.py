"""Projection-free robust aggregation controls for the mechanism ablation.

These learners are deliberately separate from the registered historical grid.
They use native embedding dimensions and cosine classification only.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ssl_hdc_pilot.scripts.continual_hdc import Prediction, UpdateResult


class _NativePrototype:
    def __init__(self, *, input_dim: int, device: torch.device | str = "cpu"):
        self.input_dim, self.device = int(input_dim), torch.device(device)
        self._vectors: dict[str, list[torch.Tensor]] = {}
        self._prototypes: dict[str, torch.Tensor | None] = {}
    @property
    def known_classes(self): return tuple(self._prototypes)
    def _vector(self, x):
        x=torch.as_tensor(x,dtype=torch.float32,device=self.device)
        if x.ndim != 1 or x.numel()!=self.input_dim: raise ValueError("invalid embedding dimension")
        return x
    def register_class(self,label):
        if label in self._prototypes:return False
        self._vectors[label]=[];self._prototypes[label]=None;return True
    def _aggregate(self, values): raise NotImplementedError
    def update(self,embedding,label):
        registered=self.register_class(str(label));self._vectors[label].append(self._vector(embedding).clone())
        self._prototypes[label]=self._aggregate(self._vectors[label]);return UpdateResult(registered)
    def get_prototype(self,label):
        if self._prototypes[label] is None: raise RuntimeError("class has no confirmed examples")
        return self._prototypes[label].clone()

    def get_vectors(self, label):
        """Return a detached copy for checkpointing the exact online state."""
        return [value.clone() for value in self._vectors[label]]
    def predict(self,embedding):
        active=[(k,v) for k,v in self._prototypes.items() if v is not None]
        if not active:return Prediction(None,0.,0.)
        q=self._vector(embedding);labels,p=zip(*active);s=F.cosine_similarity(q.unsqueeze(0),torch.stack(p),dim=1);i=int(s.argmax())
        return Prediction(labels[i],float((torch.topk(s,2).values.diff()[0].abs()) if len(s)>1 else 0.),float(torch.softmax(s,0)[i]))

class SignNCMPrototypeLearner(_NativePrototype):
    """Native-dimension sign-majority prototype; each sample has bounded influence."""
    def _aggregate(self,values):
        summed=torch.stack([torch.where(v>=0,torch.ones_like(v),-torch.ones_like(v)) for v in values]).sum(0)
        return torch.where(summed>=0,torch.ones_like(summed),-torch.ones_like(summed))

class MedianNCMPrototypeLearner(_NativePrototype):
    """Exact running median retained in-memory; appropriate for bounded development grids."""
    def _aggregate(self,values): return torch.median(torch.stack(values),dim=0).values
