# src/topo_hdc/ablations.py

from collections.abc import Callable

from topo_hdc.config import RobustHoleConfig


def original_components() -> RobustHoleConfig:
    """A0: Original exact bounded-component extraction."""
    return RobustHoleConfig(
        proposal_method="exact",
        use_spoke_validation=False,
        use_multiscale_persistence=False,
        use_spatial_pyramid=False,
        use_reliability_features=False,
        closing_radii=(0,),
        min_scale_count=1,
    )


def multiscale_only() -> RobustHoleConfig:
    """A1: Multiscale proposals and persistence, without spoke validation."""
    return RobustHoleConfig(
        proposal_method="multiscale",
        use_spoke_validation=False,
        use_multiscale_persistence=True,
        use_spatial_pyramid=False,
        use_reliability_features=False,
        closing_radii=(0, 1, 2),
        min_scale_count=1,
    )


def multiscale_spoke() -> RobustHoleConfig:
    """A2: Multiscale proposals with spoke-wheel validation."""
    return RobustHoleConfig(
        proposal_method="multiscale",
        use_spoke_validation=True,
        use_multiscale_persistence=True,
        use_spatial_pyramid=False,
        use_reliability_features=True,
        closing_radii=(0, 1, 2),
        min_scale_count=1,
    )


def multiscale_spoke_pyramid() -> RobustHoleConfig:
    """A3: Full proposed method."""
    return RobustHoleConfig(
        proposal_method="multiscale",
        use_spoke_validation=True,
        use_multiscale_persistence=True,
        use_spatial_pyramid=True,
        use_reliability_features=True,
        closing_radii=(0, 1, 2),
        min_scale_count=1,
        pyramid_levels=(1, 2, 4),
    )


def exact_spoke() -> RobustHoleConfig:
    """A4: Exact proposals with spoke validation."""
    return RobustHoleConfig(
        proposal_method="exact",
        use_spoke_validation=True,
        use_multiscale_persistence=False,
        use_spatial_pyramid=False,
        use_reliability_features=True,
        closing_radii=(0,),
        min_scale_count=1,
    )


ABLATION_PRESETS: dict[str, Callable[[], RobustHoleConfig]] = {
    "A0_original": original_components,
    "A1_multiscale": multiscale_only,
    "A2_spoke": multiscale_spoke,
    "A3_spoke_pyramid": multiscale_spoke_pyramid,
    "A4_exact_spoke": exact_spoke,
}


def get_ablation_config(name: str) -> RobustHoleConfig:
    try:
        factory = ABLATION_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(ABLATION_PRESETS)
        raise ValueError(
            f"Unknown ablation preset {name!r}. Available presets: {choices}"
        ) from exc

    return factory()
