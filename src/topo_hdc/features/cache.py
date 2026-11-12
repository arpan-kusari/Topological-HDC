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


FEATURE_CACHE_VERSION = 2

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
        "Fh": np.asarray(features.Fh, dtype=np.float32),
        "Ffm": np.asarray(features.Ffm, dtype=np.float32),
        "Fpi": np.asarray(features.Fpi, dtype=np.float32),
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
            "Fh",
            "Ffm",
            "Fpi",
            "metadata_json",
        }

        missing = required - set(archive.files)

        if missing:
            raise ValueError(
                f"Invalid feature cache {path}; missing arrays: "
                f"{sorted(missing)}"
            )

        features = FeatureBatch(
            Fh=archive["Fh"].astype(np.float32, copy=False),
            Ffm=archive["Ffm"].astype(np.float32, copy=False),
            Fpi=archive["Fpi"].astype(np.float32, copy=False),
        )

        metadata = json.loads(str(archive["metadata_json"].item()))

    validate_feature_batch(features)

    return features, metadata


def validate_feature_batch(features: FeatureBatch) -> None:
    """Check that all sample-level arrays have consistent dimensions."""
    num_samples = features.Fh.shape[0]

    sample_arrays = {
        "Fh": features.Fh,
        "Ffm": features.Ffm,
        "Fpi": features.Fpi,
    }

    for name, array in sample_arrays.items():
        if array.shape[0] != num_samples:
            raise ValueError(
                f"Feature array {name!r} has {array.shape[0]} samples; "
                f"expected {num_samples}."
            )


    if features.Fh.ndim != 2:
        raise ValueError(
            f"Fh must have shape (N, F); got {features.Fh.shape}."
        )
        
    if features.Ffm.ndim != 2:
        raise ValueError(
            f"Ffm must have shape (N, F); got {features.Ffm.shape}."
        )

    if features.Fpi.ndim != 2:
        raise ValueError(
            f"Fpi must have shape (N, F); got {features.Fpi.shape}."
        )

def make_feature_cache_key(
    *,
    dataset_name: str,
    split_name: str,
    split_seed: int,
    img_cfg,
    ablation_cfg,
    persistence_cfg,
    corruption_cfg=None,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "cache_version": FEATURE_CACHE_VERSION,
        "dataset": dataset_name,
        "split": split_name,
        "split_seed": split_seed,
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
    img_cfg,
    ablation_cfg,
    persistence_cfg,
    corruption_cfg,
    compute_fn: Callable[[], FeatureBatch],
    use_cache: bool = True,
    force_recompute: bool = False,
) -> FeatureBatch:
    cache_key, metadata = make_feature_cache_key(
        dataset_name=dataset_name,
        split_name=split_name,
        split_seed=split_seed,
        img_cfg=img_cfg,
        ablation_cfg=ablation_cfg,
        corruption_cfg=corruption_cfg,
        persistence_cfg=persistence_cfg,
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

        

        return features

    print(f"Computing {split_name} features.")

    features = compute_fn()
    validate_feature_batch(features)

    if use_cache:
        save_feature_batch(
            path=cache_path,
            features=features,
            metadata=metadata,
        )
        print(f"Saved {split_name} features: {cache_path}")

    return features



