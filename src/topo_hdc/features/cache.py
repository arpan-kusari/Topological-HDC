# src/topo_hdc/features/cache.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import hashlib
import json
from dataclasses import asdict, is_dataclass
import numpy as np
from topo_hdc.features.extractor import FeatureBatch


FEATURE_CACHE_VERSION = 1

def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def save_feature_batch(
    path: str | Path,
    features: FeatureBatch,
    metadata: dict[str, Any] | None = None,
    compressed: bool = True,
) -> Path:
    """
    Save a FeatureBatch without using Python pickle.

    Parameters
    ----------
    path:
        Destination `.npz` file.

    features:
        Extracted feature arrays.

    metadata:
        Cache identity information such as dataset, split, corruption,
        ablation, and feature settings.

    compressed:
        If True, use np.savez_compressed. Compression saves disk space but
        is somewhat slower to save/load.
    """
    path = Path(path)

    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "Fz": np.asarray(features.Fz, dtype=np.float32),
        "Fh": np.asarray(features.Fh, dtype=np.float32),
        "HF": np.asarray(features.HF, dtype=np.float32),
        "M": np.asarray(features.M, dtype=np.float32),
        "HFEATS": np.asarray(features.HFEATS, dtype=np.float32),
        "hole_counts": np.asarray(
            features.hole_counts,
            dtype=np.int64,
        ),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "cache_version": FEATURE_CACHE_VERSION,
                    **(metadata or {}),
                },
                sort_keys=True,
                default=str,
            )
        ),
    }

    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)

    return path


def load_feature_batch(
    path: str | Path,
) -> tuple[FeatureBatch, dict[str, Any]]:
    """Load a saved FeatureBatch and its metadata."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "Fz",
            "Fh",
            "HF",
            "M",
            "HFEATS",
            "hole_counts",
            "metadata_json",
        }

        missing = required - set(archive.files)

        if missing:
            raise ValueError(
                f"Invalid feature cache {path}; missing arrays: "
                f"{sorted(missing)}"
            )

        features = FeatureBatch(
            Fz=archive["Fz"].astype(np.float32, copy=False),
            Fh=archive["Fh"].astype(np.float32, copy=False),
            HF=archive["HF"].astype(np.float32, copy=False),
            M=archive["M"].astype(np.float32, copy=False),
            HFEATS=archive["HFEATS"].astype(
                np.float32,
                copy=False,
            ),
            hole_counts=archive["hole_counts"].astype(
                np.int64,
                copy=False,
            ),
        )

        metadata = json.loads(str(archive["metadata_json"].item()))

    validate_feature_batch(features)

    return features, metadata


def validate_feature_batch(features: FeatureBatch) -> None:
    """Check that all sample-level arrays have consistent dimensions."""
    num_samples = features.Fz.shape[0]

    sample_arrays = {
        "Fh": features.Fh,
        "HF": features.HF,
        "M": features.M,
        "hole_counts": features.hole_counts,
    }

    for name, array in sample_arrays.items():
        if array.shape[0] != num_samples:
            raise ValueError(
                f"Feature array {name!r} has {array.shape[0]} samples; "
                f"expected {num_samples}."
            )

    if features.Fz.ndim != 2:
        raise ValueError(
            f"Fz must have shape (N, F); got {features.Fz.shape}."
        )

    if features.Fh.ndim != 2:
        raise ValueError(
            f"Fh must have shape (N, F); got {features.Fh.shape}."
        )

    if features.HF.ndim != 3:
        raise ValueError(
            f"HF must have shape (N, Hmax, k); got {features.HF.shape}."
        )

    if features.M.shape != features.HF.shape[:2]:
        raise ValueError(
            f"Mask shape {features.M.shape} does not match "
            f"HF shape {features.HF.shape[:2]}."
        )

    if features.HFEATS.ndim != 2:
        raise ValueError(
            "HFEATS must have shape (number_of_holes, feature_length); "
            f"got {features.HFEATS.shape}."
        )

    if (
        features.HFEATS.shape[0] > 0
        and features.HFEATS.shape[1] != features.HF.shape[2]
    ):
        raise ValueError(
            f"HFEATS length {features.HFEATS.shape[1]} does not match "
            f"HF length {features.HF.shape[2]}."
        )

def make_feature_cache_key(
    *,
    dataset_name: str,
    split_name: str,
    split_seed: int,
    zernike_cfg,
    hole_cfg,
    ablation_cfg,
    corruption_cfg=None,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "cache_version": FEATURE_CACHE_VERSION,
        "dataset": dataset_name,
        "split": split_name,
        "split_seed": split_seed,
        "zernike": _json_safe(zernike_cfg),
        "holes": _json_safe(hole_cfg),
        "ablation": _json_safe(ablation_cfg),
        "corruption": _json_safe(corruption_cfg),
    }

    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256(encoded).hexdigest()[:16]

    return digest, identity
    
def load_or_compute_features(
    *,
    X: np.ndarray,
    cache_root: str | Path,
    dataset_name: str,
    split_name: str,
    split_seed: int,
    zernike_cfg,
    hole_cfg,
    ablation_cfg,
    corruption_cfg,
    compute_fn: Callable[[], FeatureBatch],
    use_cache: bool = True,
    force_recompute: bool = False,
) -> FeatureBatch:
    cache_key, metadata = make_feature_cache_key(
        dataset_name=dataset_name,
        split_name=split_name,
        split_seed=split_seed,
        zernike_cfg=zernike_cfg,
        hole_cfg=hole_cfg,
        ablation_cfg=ablation_cfg,
        corruption_cfg=corruption_cfg,
    )

    cache_path = (
        Path(cache_root)
        / dataset_name
        / split_name
        / f"features_{cache_key}.npz"
    )

    if use_cache and cache_path.is_file() and not force_recompute:
        print(f"Loading cached {split_name} features: {cache_path}")

        features, saved_metadata = load_feature_batch(cache_path)

        if saved_metadata != metadata:
            raise ValueError(
                f"Feature-cache metadata mismatch for {cache_path}. "
                "Delete the cache or force recomputation."
            )

        if features.Fz.shape[0] != len(X):
            raise ValueError(
                f"Cached {split_name} features contain "
                f"{features.Fz.shape[0]} samples, but the current split "
                f"contains {len(X)}."
            )

        return features

    print(f"Computing {split_name} features.")

    features = compute_fn()
    validate_feature_batch(features)

    if features.Fz.shape[0] != len(X):
        raise ValueError(
            f"Computed {features.Fz.shape[0]} feature rows for "
            f"{len(X)} input images."
        )

    if use_cache:
        save_feature_batch(
            path=cache_path,
            features=features,
            metadata=metadata,
        )
        print(f"Saved {split_name} features: {cache_path}")

    return features



