  # config.py
import numpy as np
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class RunConfig:
    seed:int = 0
    dim: int = 10000
    val_split: float = 0.2
    batch_size: int = 512
    epochs: int = 20
    lr: float = 1.0
    alphas=np.linspace(0, 1, 11).tolist()
    betas=np.linspace(0, 1, 11).tolist()
    device: str = "cuda"
    n_jobs: int = None
    
@dataclass(frozen=True)
class DatasetConfig:
    dataset: str = "mnist" # emnist|svhn
    num_classes:int = 10 # 26

@dataclass(frozen=True)
class CorruptionConfig:
    kind: str = "none"   # rotation|gaussian|saltpepper|cutout|zoom|none
    angle_deg: float = 20.0
    sigma: float = 0.1
    p: float = 0.1
    cutout_size: int = 12
    zoom: float = 0.5

@dataclass(frozen=True)
class HoleConfig:
    k_shape: int = 12
    feature_length: int = 16 # k_shape + hole feature vector
    Q: int = 101
    Hmax: int = 5
    
@dataclass(frozen=True)
class ZernikeConfig:
    out_size: int = 64
    degree: int = 6
    radius: float = 20
    grid: tuple =(2,2)
    pad:int =2
    
@dataclass(frozen=True)
class FusionConfig:
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    alpha_step: float = 0.1
    beta_min: float = 0.0
    beta_max: float = 1.0
    beta_step: float = 0.1
    
@dataclass(frozen=True)
class PlotConfig:
    figsize: tuple = (16,12)
    before_title: str = "Classification Matrix for Test Set before Training"
    before_filename: str = 'figures/mnist_before_train.png'
    fontsize:int =20
    after_title: str = "Classification Matrix for Test Set after Training"
    after_filename: str = 'figures/mnist_after_train.png'


@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    data: DatasetConfig
    corrupt: CorruptionConfig
    zernike: ZernikeConfig
    holes: HoleConfig
    fusion: FusionConfig
    plot: PlotConfig
    out_dir: Path


def validate_config(cfg: ExperimentConfig) -> None:
    allowed_datasets = {"mnist", "emnist", "svhn"}
    allowed_corruptions = {"none", "rotation", "gaussian", "saltpepper", "cutout", "zoom"}

    if cfg.data.dataset not in allowed_datasets:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset}")

    if cfg.corrupt.kind not in allowed_corruptions:
        raise ValueError(f"Unknown corruption: {cfg.corrupt.kind}")

    if cfg.run.dim <= 0:
        raise ValueError("run.dim must be positive.")

    if cfg.run.batch_size <= 0:
        raise ValueError("run.batch_size must be positive.")

    if cfg.holes.Hmax <= 0:
        raise ValueError("holes.Hmax must be positive.")

    if not (0.0 < cfg.run.val_split < 1.0):
        raise ValueError("data.val_split must be between 0 and 1.")
