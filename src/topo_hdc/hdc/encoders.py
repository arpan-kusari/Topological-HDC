# src/topo_hdc/hdc/encoders.py

from __future__ import annotations

from dataclasses import dataclass

import torch

from topo_hdc.hdc.RandomProjEncoder import RandomProjEncoderTorch
from topo_hdc.hdc.HoleSetHDC import HoleSetHDCTorch
from topo_hdc.utils.Helper_functions import rand_bipolar
from topo_hdc.config import RunConfig, HoleConfig
from topo_hdc.features.extractor import FeatureBatch


@dataclass
class Encoders:
    enc_z: RandomProjEncoderTorch
    enc_h: RandomProjEncoderTorch
    enc_holes: HoleSetHDCTorch


@dataclass
class Roles:
    role_outer: torch.Tensor
    role_hog: torch.Tensor
    role_holes: torch.Tensor


def fit_encoders(
    feats: FeatureBatch,
    run_cfg: RunConfig,
    hole_cfg: HoleConfig,
) -> Encoders:
    enc_z = RandomProjEncoderTorch(
        D=run_cfg.dim,
        seed=1,
        device=run_cfg.device,
    ).fit(feats.Fz)

    enc_h = RandomProjEncoderTorch(
        D=run_cfg.dim,
        seed=2,
        device=run_cfg.device,
    ).fit(feats.Fh)

    # Use flattened hole features to fit the hole scaler/quantizer.
    # feats.HFEATS should have shape (num_total_holes, hole_cfg.feature_length).
    enc_holes = HoleSetHDCTorch(
        D=run_cfg.dim,
        Q=hole_cfg.Q,
        k=hole_cfg.feature_length,
        seed=123,
        device=run_cfg.device,
    ).fit_scaler(feats.HFEATS)

    return Encoders(
        enc_z=enc_z,
        enc_h=enc_h,
        enc_holes=enc_holes,
    )


def get_role(run_cfg: RunConfig) -> Roles:
    role_outer = rand_bipolar(
        D=run_cfg.dim,
        seed=999,
        device=run_cfg.device,
    )

    role_hog = rand_bipolar(
        D=run_cfg.dim,
        seed=1000,
        device=run_cfg.device,
    )

    role_holes = rand_bipolar(
        D=run_cfg.dim,
        seed=1001,
        device=run_cfg.device,
    )

    return Roles(
        role_outer=role_outer,
        role_hog=role_hog,
        role_holes=role_holes,
    )
