# src/topo_hdc/hdc/encoders.py

from __future__ import annotations

from dataclasses import dataclass

import torch

from topo_hdc.hdc.RandomProjEncoder import RandomProjEncoderTorch
from topo_hdc.utils.Helper_functions import rand_bipolar
from topo_hdc.config import RunConfig
from topo_hdc.features.extractor import FeatureBatch


@dataclass
class Encoders:
    enc_h: RandomProjEncoderTorch
    enc_fmt: RandomProjEncoderTorch
    enc_pi: RandomProjEncoderTorch


@dataclass
class Roles:
    role_hog: torch.Tensor
    role_fmt: torch.Tensor
    role_pi: torch.Tensor

def hole_feature_length(
    hole_k_shape: int,
    ablation_cfg: RobustHoleConfig,
) -> int:
    base_length = hole_k_shape + 6

    pyramid_length = sum(
        level * level
        for level in ablation_cfg.pyramid_levels
    )

    return base_length + pyramid_length
    
def fit_encoders(
    feats: FeatureBatch,
    run_cfg: RunConfig,
    ablation_cfg: RobustHoleConfig,
) -> Encoders:

    enc_h = RandomProjEncoderTorch(
        D=run_cfg.dim,
        seed=2,
        device=run_cfg.device,
    ).fit(feats.Fh)
    
    enc_fmt = RandomProjEncoderTorch(
        D=run_cfg.dim,
        seed=3,
        device=run_cfg.device,
    ).fit(feats.Ffm)
    
    enc_pi = RandomProjEncoderTorch(
        D=run_cfg.dim,
        seed=4,
        device=run_cfg.device,
    ).fit(feats.Fpi)

    return Encoders(
        enc_h=enc_h,
        enc_fmt=enc_fmt,
        enc_pi=enc_pi,
    )


def get_role(run_cfg: RunConfig) -> Roles:
    role_hog = rand_bipolar(
        D=run_cfg.dim,
        seed=1000,
        device=run_cfg.device,
    )
    
    role_fmt = rand_bipolar(
        D=run_cfg.dim,
        seed=1002,
        device=run_cfg.device,
    )
    
    role_pi = rand_bipolar(
        D=run_cfg.dim,
        seed=1003,
        device=run_cfg.device,
    )

    return Roles(
        role_hog=role_hog,
        role_fmt=role_fmt,
        role_pi=role_pi,
    )
