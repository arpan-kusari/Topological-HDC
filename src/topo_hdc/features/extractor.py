# src/topo_hdc/features/extractor.py

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

import cripser
import numpy as np
from joblib import Parallel, delayed, dump, load
from scipy.ndimage import gaussian_filter
from gtda.images import HeightFiltration

from topo_hdc.config import (
    ImgProcessConfig,
    PersistenceImageConfig,
    AblationConfig,
)
from topo_hdc.features.Image_processing import ProcessImage


def identity_persistence_weight(x):
    """Linear persistence weighting."""
    return x


@dataclass
class Features:
    hog: np.ndarray
    fourier_mellin: np.ndarray
    persistent_image: np.ndarray


@dataclass
class FeatureBatch:
    Fh: np.ndarray   # (N, HOG feature length)
    Ffm: np.ndarray  # (N, Fourier--Mellin feature length)
    Fpi: np.ndarray  # (N, 8 * n_bins * n_bins)


def _format_persistence_diagram(raw_ph: np.ndarray) -> np.ndarray:
    """
    Convert cripser output to [birth, persistence, dimension].

    Excludes nonfinite pairs, essential pairs represented by the
    largest float64 value, nonpositive lifetimes, and dimensions
    other than H0 and H1.
    """
    raw_ph = np.asarray(raw_ph, dtype=np.float64)

    if raw_ph.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    if raw_ph.ndim != 2 or raw_ph.shape[1] < 3:
        raise ValueError(
            f"Unexpected cripser output shape: {raw_ph.shape}."
        )

    dimensions = raw_ph[:, 0]
    births = raw_ph[:, 1]
    deaths = raw_ph[:, 2]

    # cripser can represent essential classes with DBL_MAX.
    valid = (
        np.isfinite(births)
        & np.isfinite(deaths)
        & (deaths < np.finfo(np.float64).max)
        & (deaths > births)
        & np.isin(dimensions, [0, 1])
    )

    births = births[valid]
    deaths = deaths[valid]
    dimensions = dimensions[valid]

    with np.errstate(over="ignore", invalid="ignore"):
        persistences = deaths - births

    finite_lifetimes = (
        np.isfinite(persistences)
        & (persistences > 0)
    )

    return np.column_stack(
        (
            births[finite_lifetimes],
            persistences[finite_lifetimes],
            dimensions[finite_lifetimes],
        )
    )


def _valid_diagram_rows(diagram: np.ndarray) -> np.ndarray:
    """Validate and filter [birth, persistence, dimension] rows."""
    diagram = np.asarray(diagram, dtype=np.float64)

    if diagram.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    if diagram.ndim != 2 or diagram.shape[1] != 3:
        raise ValueError(
            "A persistence diagram must have shape (n_points, 3); "
            f"got {diagram.shape}."
        )

    valid = (
        np.isfinite(diagram).all(axis=1)
        & (diagram[:, 1] > 0)
        & np.isin(diagram[:, 2], [0, 1])
    )

    return diagram[valid]


def _validate_bounds(bounds: dict) -> dict[str, float]:
    """Check that histogram bounds are finite and nondegenerate."""
    required = ("b_min", "b_max", "p_min", "p_max")

    if not isinstance(bounds, dict):
        raise ValueError("Persistence-image bounds must be a dictionary.")

    missing = set(required) - set(bounds)

    if missing:
        raise ValueError(
            f"Persistence-image bounds are missing: {sorted(missing)}."
        )

    result = {key: float(bounds[key]) for key in required}

    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError(
            "Persistence-image bounds must contain only finite values."
        )

    if result["b_max"] <= result["b_min"]:
        raise ValueError("b_max must be greater than b_min.")

    if result["p_max"] <= result["p_min"]:
        raise ValueError("p_max must be greater than p_min.")

    return result


def _nondegenerate_range(
    minimum: float,
    maximum: float,
) -> tuple[float, float]:
    """Expand a constant range without changing ordinary ranges."""
    if maximum <= minimum:
        delta = max(1e-4, abs(minimum) * 1e-6)
        maximum = minimum + delta

    return minimum, maximum


def _fit_global_bounds(
    nested_diagrams: Sequence[Sequence[np.ndarray]],
) -> dict[str, float]:
    """Fit shared bounds using only valid, finite persistence pairs."""
    b_min = np.inf
    b_max = -np.inf
    p_min = np.inf
    p_max = -np.inf

    for sample_diagrams in nested_diagrams:
        for diagram in sample_diagrams:
            valid = _valid_diagram_rows(diagram)

            if len(valid) == 0:
                continue

            b_min = min(b_min, float(valid[:, 0].min()))
            b_max = max(b_max, float(valid[:, 0].max()))
            p_min = min(p_min, float(valid[:, 1].min()))
            p_max = max(p_max, float(valid[:, 1].max()))

    if not np.isfinite(b_min):
        return {
            "b_min": 0.0,
            "b_max": 1.0,
            "p_min": 0.0,
            "p_max": 1.0,
        }

    b_min, b_max = _nondegenerate_range(b_min, b_max)
    p_min, p_max = _nondegenerate_range(p_min, p_max)

    return _validate_bounds(
        {
            "b_min": b_min,
            "b_max": b_max,
            "p_min": p_min,
            "p_max": p_max,
        }
    )


class FeatureExtractor:
    def __init__(
        self,
        img_cfg: ImgProcessConfig,
        ablation_cfg: AblationConfig,
        persistence_cfg: PersistenceImageConfig,
        n_jobs: int | None = None,
    ) -> None:
        self.img_cfg = img_cfg
        self.ablation_cfg = ablation_cfg
        self.persistence_cfg = persistence_cfg
        self.n_jobs = (
            n_jobs if n_jobs is not None else (os.cpu_count() or 1)
        )

    def extract_one(
        self,
        img: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        img_process = ProcessImage(img)

        norm = img_process.normalize_glyph_gray(
            out_size=self.img_cfg.out_size,
            pad=self.img_cfg.pad,
        )

        fh = img_process.hog_descriptor(norm)

        # Preserves the original Fourier--Mellin implementation,
        # including its separate normalization with default padding.
        ffm = img_process.fourier_mellin_descriptor(
            out_size=self.img_cfg.out_size,
        )

        directions = (
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
            np.array([1, -1]),
        )

        single_image = norm[np.newaxis, ...]
        diagrams_for_this_image = []

        for direction in directions:
            filtration = HeightFiltration(direction=direction)
            filtered_digit = filtration.fit_transform(single_image)

            # Remove the leading sample axis: (1, H, W) -> (H, W).
            grid_2d = np.ascontiguousarray(
                filtered_digit[0],
                dtype=np.float64,
            )

            if grid_2d.ndim != 2:
                raise ValueError(
                    "Expected a 2D filtration grid; "
                    f"got {grid_2d.shape}."
                )

            raw_ph = cripser.compute_ph(
                grid_2d,
                filtration="V",
                maxdim=1,
            )

            diagrams_for_this_image.append(
                _format_persistence_diagram(raw_ph)
            )

        return (
            np.asarray(fh, dtype=np.float32),
            np.asarray(ffm, dtype=np.float32),
            diagrams_for_this_image,
        )

    @staticmethod
    def generate_fast_persistence_image(
        diagram: np.ndarray,
        n_bins: int,
        sigma: float,
        bounds: dict,
    ) -> np.ndarray:
        """
        Generate concatenated H0 and H1 persistence images.

        Histogram points are weighted by persistence and smoothed
        with a Gaussian filter. Sigma is expressed in coordinate
        units and converted separately for each histogram axis.

        Points outside the fitted bounds are omitted by histogram2d.
        """
        if not isinstance(n_bins, (int, np.integer)) or n_bins <= 0:
            raise ValueError("n_bins must be a positive integer.")

        sigma = float(sigma)

        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative.")

        bounds = _validate_bounds(bounds)
        diagram = _valid_diagram_rows(diagram)

        if len(diagram) == 0:
            return np.zeros(
                2 * n_bins * n_bins,
                dtype=np.float32,
            )

        birth_range = (bounds["b_min"], bounds["b_max"])
        persistence_range = (bounds["p_min"], bounds["p_max"])

        birth_step = (
            bounds["b_max"] - bounds["b_min"]
        ) / n_bins
        persistence_step = (
            bounds["p_max"] - bounds["p_min"]
        ) / n_bins

        sigma_indices = (
            sigma / birth_step,
            sigma / persistence_step,
        )

        pi_channels = []

        for target_dim in (0, 1):
            sub_diagram = diagram[
                diagram[:, 2] == target_dim
            ]

            if len(sub_diagram) == 0:
                pi_channels.append(
                    np.zeros(
                        (n_bins, n_bins),
                        dtype=np.float64,
                    )
                )
                continue

            hist, _, _ = np.histogram2d(
                sub_diagram[:, 0],
                sub_diagram[:, 1],
                bins=n_bins,
                range=[birth_range, persistence_range],
                weights=sub_diagram[:, 1],
            )

            if sigma > 0:
                hist = gaussian_filter(
                    hist,
                    sigma=sigma_indices,
                    mode="constant",
                    cval=0.0,
                )

            pi_channels.append(hist)

        return np.stack(pi_channels).reshape(-1).astype(
            np.float32
        )


def _vectorize_sample_worker(
    sample_channels: Sequence[np.ndarray],
    n_bins: int,
    sigma: float,
    bounds: dict,
) -> np.ndarray:
    """Module-level worker for parallel PI vectorization."""
    return np.concatenate(
        [
            FeatureExtractor.generate_fast_persistence_image(
                diagram=diagram,
                n_bins=n_bins,
                sigma=sigma,
                bounds=bounds,
            )
            for diagram in sample_channels
        ]
    ).astype(np.float32, copy=False)

def compute_features(
    X: np.ndarray,
    img_cfg: ImgProcessConfig,
    ablation_cfg: AblationConfig,
    persistence_cfg: PersistenceImageConfig,
    n_jobs: int | None = None,
    fit: bool = True,
    cache_dir: str | Path | None = None,
) -> FeatureBatch:
    """
    Compute HOG, Fourier--Mellin, and directional PI features.

    Use fit=True for training to save global PI bounds.
    Use fit=False for validation/test to reuse those bounds.
    """
    if len(X) == 0:
        raise ValueError("X must contain at least one image.")

    n_bins = persistence_cfg.nbins
    sigma = float(persistence_cfg.sigma)

    if not isinstance(n_bins, (int, np.integer)) or n_bins <= 0:
        raise ValueError(
            "persistence_cfg.nbins must be a positive integer."
        )

    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(
            "persistence_cfg.sigma must be finite and nonnegative."
        )

    feature_extractor = FeatureExtractor(
        img_cfg=img_cfg,
        ablation_cfg=ablation_cfg,
        persistence_cfg=persistence_cfg,
        n_jobs=n_jobs,
    )

    bounds_root = (
        Path(cache_dir) if cache_dir is not None else Path(".")
    )
    bounds_path = bounds_root / "tda_global_bounds.pkl"

    # Fail early if validation/test bounds are unavailable.
    bounds = None

    if not fit:
        if not bounds_path.is_file():
            raise FileNotFoundError(
                f"Training PI bounds not found: {bounds_path}. "
                "Run the training split with fit=True first."
            )

        print(f"Loading training PI bounds: {bounds_path}")
        bounds = _validate_bounds(load(bounds_path))

    print("----------Extracting features and persistence diagrams----------")

    # loky supports serialization of workers without the nested-worker
    # limitations of the multiprocessing backend.
    with Parallel(
        n_jobs=feature_extractor.n_jobs,
        backend="loky",
        max_nbytes="1M",
    ) as parallel:
        raw_results = parallel(
            delayed(feature_extractor.extract_one)(img)
            for img in X
        )

        hog_list, fm_list, nested_diagrams = zip(*raw_results)

        if fit:
            print("Fitting global PI coordinate bounds...")
            bounds = _fit_global_bounds(nested_diagrams)

            bounds_root.mkdir(parents=True, exist_ok=True)
            dump(bounds, bounds_path)

            print(f"Saved training PI bounds: {bounds_path}")

        print("----------Vectorizing persistence images----------")

        pi_features = parallel(
            delayed(_vectorize_sample_worker)(
                sample_channels=sample_channels,
                n_bins=n_bins,
                sigma=sigma,
                bounds=bounds,
            )
            for sample_channels in nested_diagrams
        )

    Fh = np.vstack(hog_list).astype(np.float32, copy=False)
    Ffm = np.vstack(fm_list).astype(np.float32, copy=False)
    Fpi = np.vstack(pi_features).astype(np.float32, copy=False)

    expected_pi_length = 8 * n_bins * n_bins

    if Fpi.shape != (len(X), expected_pi_length):
        raise ValueError(
            f"Expected Fpi shape {(len(X), expected_pi_length)}; "
            f"got {Fpi.shape}."
        )

    return FeatureBatch(
        Fh=Fh,
        Ffm=Ffm,
        Fpi=Fpi,
    )
