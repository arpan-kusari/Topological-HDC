from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math


# Ordering must match the full PI feature matrix produced by the extractor.
PI_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (1, 1),
    (1, -1),
)


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    dim: int = 10000
    val_split: float = 0.2
    batch_size: int = 512
    epochs: int = 20
    lr: float = 1.0

    alphas: tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0,
    )
    betas: tuple[float, ...] = (
        0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4,
    )

    device: str = "cuda"
    n_jobs: int | None = None
    force_recompute_prototypes: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str = "mnist"
    num_classes: int = 10


@dataclass(frozen=True)
class CorruptionConfig:
    kind: str = "none"
    angle_deg: float = 20.0
    sigma: float = 0.1
    p: float = 0.1
    cutout_size: int = 12
    zoom: float = 0.5


@dataclass(frozen=True)
class AblationConfig:
    """Settings for channel, PI, fusion, and OnlineHD ablations."""

    use_hog: bool = True
    use_fm: bool = True
    use_pi: bool = True

    pi_direction_indices: tuple[int, ...] = (0, 1, 2, 3)
    pi_homology_dims: tuple[int, ...] = (0, 1)

    pi_weighting: str = "persistence"
    use_pi_smoothing: bool = True

    fusion_mode: str = "validation"
    use_onlinehd: bool = True

    def __post_init__(self) -> None:
        if not (self.use_hog or self.use_fm or self.use_pi):
            raise ValueError("At least one descriptor channel is required.")

        _validate_index_subset(
            self.pi_direction_indices,
            allowed=(0, 1, 2, 3),
            name="ablation.pi_direction_indices",
        )
        _validate_index_subset(
            self.pi_homology_dims,
            allowed=(0, 1),
            name="ablation.pi_homology_dims",
        )

        if self.pi_weighting not in {"persistence", "uniform"}:
            raise ValueError(
                "ablation.pi_weighting must be 'persistence' or 'uniform'."
            )

        if self.fusion_mode not in {"validation", "equal"}:
            raise ValueError(
                "ablation.fusion_mode must be 'validation' or 'equal'."
            )

    @property
    def active_channels(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, enabled in (
                ("hog", self.use_hog),
                ("fm", self.use_fm),
                ("pi", self.use_pi),
            )
            if enabled
        )

    @property
    def pi_directions(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            PI_DIRECTIONS[index]
            for index in self.pi_direction_indices
        )

    def pi_feature_length(self, n_bins: int) -> int:
        """PI length after direction and homology selection."""
        _validate_positive_int(n_bins, "n_bins")

        if not self.use_pi:
            return 0

        return (
            len(self.pi_direction_indices)
            * len(self.pi_homology_dims)
            * n_bins
            * n_bins
        )


@dataclass(frozen=True)
class PersistenceImageConfig:
    nbins: int = 20
    sigma: float = 0.05

    @property
    def full_feature_length(self) -> int:
        """Four directions times two homology dimensions."""
        return 8 * self.nbins * self.nbins


@dataclass(frozen=True)
class ImgProcessConfig:
    out_size: int = 64
    degree: int = 6
    radius: float = 20.0
    grid: tuple[int, int] = (2, 2)
    pad: int = 2


@dataclass(frozen=True)
class HoleConfig:
    """Legacy settings retained for hole utility compatibility."""

    k_shape: int = 12
    feature_length: int = 59
    Q: int = 101
    Hmax: int = 5


@dataclass(frozen=True)
class RobustHoleConfig:
    """
    Legacy morphological-hole configuration.

    Use only with legacy hole utilities, not as the active
    ExperimentConfig.ablation configuration.
    """

    proposal_method: str = "multiscale"
    use_spoke_validation: bool = True
    use_multiscale_persistence: bool = True
    use_spatial_pyramid: bool = True
    use_reliability_features: bool = True

    closing_radii: tuple[int, ...] = (0, 1)
    num_spokes: int = 64
    enclosure_threshold: float = 0.95
    max_missing_fraction: float = 0.08

    min_area: int = 4
    max_area_fraction: float = 0.30
    min_scale_count: int = 2
    ray_step: float = 0.5
    merge_distance: float = 3.0

    pyramid_levels: tuple[int, ...] = (1, 2, 4)


@dataclass(frozen=True)
class FusionConfig:
    """
    Optional range-based fusion grid settings.

    RunConfig.alphas/betas retain the explicit grids used by existing
    code. Choose one grid source in the training code rather than
    silently mixing the two.
    """

    alpha_min: float = 0.0
    alpha_max: float = 1.0
    alpha_step: float = 0.1
    beta_min: float = 0.0
    beta_max: float = 1.0
    beta_step: float = 0.1


@dataclass(frozen=True)
class PlotConfig:
    figsize: tuple[float, float] = (16, 12)
    before_title: str = "Classification Matrix Before OnlineHD"
    before_filename: str = "figures/mnist_before_train.png"
    fontsize: int = 20
    after_title: str = "Classification Matrix After OnlineHD"
    after_filename: str = "figures/mnist_after_train.png"


@dataclass(frozen=True)
class FeatureConfig:
    cache_dir: Path = Path("cache/features")
    use_cache: bool = True
    force_recompute: bool = False

    # None preserves Joblib's default worker behavior;
    # use -1 explicitly to request all available CPUs.
    n_jobs: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    data: DatasetConfig
    corrupt: CorruptionConfig
    img_process: ImgProcessConfig
    ablation: AblationConfig
    ablation_name: str
    fusion: FusionConfig
    plot: PlotConfig
    features: FeatureConfig
    persistence: PersistenceImageConfig
    out_dir: Path


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


def _validate_index_subset(
    values: tuple[int, ...],
    *,
    allowed: tuple[int, ...],
    name: str,
) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty.")

    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in allowed
        for value in values
    ):
        raise ValueError(f"{name} must contain indices from {allowed}.")

    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")


def _validate_n_jobs(value: int | None, name: str) -> None:
    if value is None:
        return

    if isinstance(value, bool) or not isinstance(value, int) or value == 0:
        raise ValueError(
            f"{name} must be None or a nonzero integer."
        )


def _validate_weight_grid(
    values: tuple[float, ...],
    name: str,
) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty.")

    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(
            f"{name} must contain finite, nonnegative weights."
        )


def _validate_fusion_range(
    minimum: float,
    maximum: float,
    step: float,
    name: str,
) -> None:
    if not all(math.isfinite(value) for value in (minimum, maximum, step)):
        raise ValueError(f"{name} range values must be finite.")

    if minimum < 0 or maximum < minimum or step <= 0:
        raise ValueError(
            f"{name} requires 0 <= minimum <= maximum and step > 0."
        )


def validate_config(cfg: ExperimentConfig) -> None:
    allowed_datasets = {"mnist", "emnist", "svhn", "medmnist"}
    allowed_corruptions = {
        "none", "rotation", "gaussian", "saltpepper", "cutout", "zoom"
    }

    if cfg.data.dataset not in allowed_datasets:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset}")

    if cfg.corrupt.kind not in allowed_corruptions:
        raise ValueError(f"Unknown corruption: {cfg.corrupt.kind}")

    if not isinstance(cfg.ablation, AblationConfig):
        raise TypeError(
            "ExperimentConfig.ablation must be an AblationConfig, "
            "not a RobustHoleConfig."
        )

    _validate_positive_int(cfg.data.num_classes, "data.num_classes")
    _validate_positive_int(cfg.run.dim, "run.dim")
    _validate_positive_int(cfg.run.batch_size, "run.batch_size")
    _validate_nonnegative_int(cfg.run.epochs, "run.epochs")

    if not math.isfinite(cfg.run.lr) or cfg.run.lr <= 0:
        raise ValueError("run.lr must be finite and positive.")

    if not 0.0 < cfg.run.val_split < 1.0:
        raise ValueError("run.val_split must be between 0 and 1.")

    _validate_n_jobs(cfg.run.n_jobs, "run.n_jobs")
    _validate_n_jobs(cfg.features.n_jobs, "features.n_jobs")
    _validate_weight_grid(cfg.run.alphas, "run.alphas")
    _validate_weight_grid(cfg.run.betas, "run.betas")

    _validate_positive_int(cfg.img_process.out_size, "img_process.out_size")
    _validate_nonnegative_int(cfg.img_process.pad, "img_process.pad")

    # FM radial sampling needs a nontrivial normalized canvas.
    if cfg.ablation.use_fm and cfg.img_process.out_size < 3:
        raise ValueError(
            "img_process.out_size must be at least 3 for the FM descriptor."
        )

    _validate_positive_int(cfg.persistence.nbins, "persistence.nbins")

    if (
        not math.isfinite(cfg.persistence.sigma)
        or cfg.persistence.sigma < 0
    ):
        raise ValueError(
            "persistence.sigma must be finite and nonnegative."
        )

    _validate_fusion_range(
        cfg.fusion.alpha_min,
        cfg.fusion.alpha_max,
        cfg.fusion.alpha_step,
        "fusion.alpha",
    )
    _validate_fusion_range(
        cfg.fusion.beta_min,
        cfg.fusion.beta_max,
        cfg.fusion.beta_step,
        "fusion.beta",
    )

    if not math.isfinite(cfg.corrupt.angle_deg):
        raise ValueError("corrupt.angle_deg must be finite.")

    if not math.isfinite(cfg.corrupt.sigma) or cfg.corrupt.sigma < 0:
        raise ValueError("corrupt.sigma must be finite and nonnegative.")

    if not 0.0 <= cfg.corrupt.p <= 1.0:
        raise ValueError("corrupt.p must be in [0, 1].")

    _validate_positive_int(cfg.corrupt.cutout_size, "corrupt.cutout_size")

    if not math.isfinite(cfg.corrupt.zoom) or cfg.corrupt.zoom <= 0:
        raise ValueError("corrupt.zoom must be finite and positive.")

    if not cfg.ablation_name.strip():
        raise ValueError("ablation_name must not be empty.")
