"""Preregistered DINOv3 encoder ladder identities."""
from __future__ import annotations

from dataclasses import dataclass


VITS16_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
VITB16_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
VITL16_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
# Frozen-only ceiling rung (2026-07-20 addition). 6.7B params, 4096-d CLS.
# Not adaptable: label-free DINO adaptation of a 7B backbone does not fit the
# 24 GB lab 4090, so this rung is extraction-only (arm_f) and is rejected by
# the adaptation validator.
VIT7B16_MODEL_ID = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
HEADLINE_MODEL_ID = VITB16_MODEL_ID


@dataclass(frozen=True)
class EncoderModelSpec:
    model_id: str
    rung: str
    embedding_dim: int
    arm_d_directory: str
    adaptable: bool = True


LADDER_MODEL_SPECS = {
    VITS16_MODEL_ID: EncoderModelSpec(VITS16_MODEL_ID, "vits16", 384, "arm_d_vits16"),
    VITB16_MODEL_ID: EncoderModelSpec(VITB16_MODEL_ID, "vitb16", 768, "arm_d_vitb16"),
    VITL16_MODEL_ID: EncoderModelSpec(VITL16_MODEL_ID, "vitl16", 1024, "arm_d_vitl16"),
    VIT7B16_MODEL_ID: EncoderModelSpec(
        VIT7B16_MODEL_ID, "vit7b16", 4096, "arm_d_vit7b16", adaptable=False
    ),
}


def get_model_spec(model_id: str) -> EncoderModelSpec:
    try:
        return LADDER_MODEL_SPECS[model_id]
    except KeyError as error:
        raise ValueError(
            "model id is not in the preregistered encoder-size ladder"
        ) from error
