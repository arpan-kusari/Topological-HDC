# src/topo_hdc/ablations.py

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from topo_hdc.config import AblationConfig, PI_DIRECTIONS

def _validate_n_bins(n_bins: int) -> None:
    if (
        isinstance(n_bins, bool)
        or not isinstance(n_bins, (int, np.integer))
        or n_bins <= 0
    ):
        raise ValueError("n_bins must be a positive integer.")


# ---------------------------------------------------------------------
# Channel ablations
# ---------------------------------------------------------------------

def full_model() -> AblationConfig:
    """Full HOG + FM + PI pipeline."""
    return AblationConfig()


def hog_only() -> AblationConfig:
    return replace(full_model(), use_fm=False, use_pi=False)


def fm_only() -> AblationConfig:
    return replace(full_model(), use_hog=False, use_pi=False)


def pi_only() -> AblationConfig:
    return replace(full_model(), use_hog=False, use_fm=False)


def hog_fm() -> AblationConfig:
    """Appearance baseline without topology."""
    return replace(full_model(), use_pi=False)


def hog_pi() -> AblationConfig:
    return replace(full_model(), use_fm=False)


def fm_pi() -> AblationConfig:
    return replace(full_model(), use_hog=False)


# ---------------------------------------------------------------------
# PI structure ablations: keep HOG and FM active
# ---------------------------------------------------------------------

def pi_h0_only() -> AblationConfig:
    """Connected-component persistence only."""
    return replace(full_model(), pi_homology_dims=(0,))


def pi_h1_only() -> AblationConfig:
    """Loop persistence only."""
    return replace(full_model(), pi_homology_dims=(1,))


def pi_direction_0() -> AblationConfig:
    return replace(full_model(), pi_direction_indices=(0,))


def pi_direction_1() -> AblationConfig:
    return replace(full_model(), pi_direction_indices=(1,))


def pi_direction_2() -> AblationConfig:
    return replace(full_model(), pi_direction_indices=(2,))


def pi_direction_3() -> AblationConfig:
    return replace(full_model(), pi_direction_indices=(3,))


def pi_two_directions() -> AblationConfig:
    """Horizontal and vertical sweep directions."""
    return replace(full_model(), pi_direction_indices=(0, 1))


def pi_uniform_weighting() -> AblationConfig:
    return replace(full_model(), pi_weighting="uniform")


def pi_no_smoothing() -> AblationConfig:
    return replace(full_model(), use_pi_smoothing=False)


# ---------------------------------------------------------------------
# Classifier ablations
# ---------------------------------------------------------------------

def no_onlinehd() -> AblationConfig:
    """Use initially bundled prototypes without refinement."""
    return replace(full_model(), use_onlinehd=False)


def equal_fusion() -> AblationConfig:
    """Use unit weights for all active channels."""
    return replace(full_model(), fusion_mode="equal")


ABLATION_PRESETS: dict[str, Callable[[], AblationConfig]] = {
    "A0_full": full_model,
    "A1_hog": hog_only,
    "A2_fm": fm_only,
    "A3_pi": pi_only,
    "A4_hog_fm": hog_fm,
    "A5_hog_pi": hog_pi,
    "A6_fm_pi": fm_pi,
    "T0_h0": pi_h0_only,
    "T1_h1": pi_h1_only,
    "T2_direction_0": pi_direction_0,
    "T3_direction_1": pi_direction_1,
    "T4_direction_2": pi_direction_2,
    "T5_direction_3": pi_direction_3,
    "T6_two_directions": pi_two_directions,
    "T7_uniform_weighting": pi_uniform_weighting,
    "T8_no_smoothing": pi_no_smoothing,
    "C0_no_onlinehd": no_onlinehd,
    "C1_equal_fusion": equal_fusion,
}


def get_ablation_config(name: str) -> AblationConfig:
    try:
        factory = ABLATION_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(ABLATION_PRESETS)
        raise ValueError(
            f"Unknown ablation preset {name!r}. "
            f"Available presets: {choices}"
        ) from exc

    return factory()


def select_pi_features(
    Fpi: np.ndarray,
    *,
    n_bins: int,
    config: AblationConfig,
) -> np.ndarray:
    """
    Select PI channels from a FULL cached PI matrix.

    Expected input ordering:
        direction 0: H0, H1
        direction 1: H0, H1
        direction 2: H0, H1
        direction 3: H0, H1

    Input shape:
        (N, 8 * n_bins * n_bins)

    This function only selects directions and homology dimensions.
    It cannot change weighting or smoothing in already computed PIs.
    """
    _validate_n_bins(n_bins)
    Fpi = np.asarray(Fpi)

    expected_length = 8 * n_bins * n_bins

    if Fpi.ndim != 2 or Fpi.shape[1] != expected_length:
        raise ValueError(
            f"Expected full PI shape (N, {expected_length}); "
            f"got {Fpi.shape}."
        )

    if not config.use_pi:
        return np.empty((Fpi.shape[0], 0), dtype=Fpi.dtype)

    channels = Fpi.reshape(
        Fpi.shape[0], 4, 2, n_bins, n_bins
    )

    selected = np.take(
        channels,
        config.pi_direction_indices,
        axis=1,
    )
    selected = np.take(
        selected,
        config.pi_homology_dims,
        axis=2,
    )

    return selected.reshape(
        Fpi.shape[0],
        config.pi_feature_length(n_bins),
    )
