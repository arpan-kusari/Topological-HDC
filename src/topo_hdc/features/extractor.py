# src/topo_hdc/features/extractor.py

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed
from topo_hdc.features.Image_processing import ProcessImage


from topo_hdc.config import HoleConfig, ZernikeConfig


@dataclass
class Features:
    zernike: np.ndarray          # shape: (Fz,)
    hog: np.ndarray              # shape: (Fh,)
    holes: list[np.ndarray]      # list of hole feature vectors, each shape: (k,)


@dataclass
class FeatureBatch:
    Fz: np.ndarray               # shape: (N, Fz)
    Fh: np.ndarray               # shape: (N, Fh)
    HF: np.ndarray               # shape: (N, Hmax, k)
    M: np.ndarray                # shape: (N, Hmax)
    HFEATS: np.ndarray | None = None  # shape: (num_total_holes, k), used to fit hole scaler


class FeatureExtractor:
    def __init__(
        self,
        z_cfg: ZernikeConfig,
        hole_cfg: HoleConfig,
        n_jobs: int | None = None,
    ) -> None:
        self.z_cfg = z_cfg
        self.hole_cfg = hole_cfg
        self.n_jobs = n_jobs if n_jobs is not None else os.cpu_count()

    def extract_one(self, img: np.ndarray) -> Features:
        img_process = ProcessImage(img)

        norm = img_process.normalize_glyph_gray(
            out_size=self.z_cfg.out_size,
            pad=self.z_cfg.pad,
        )

        fz = img_process.spatial_pyramid_zernike_gray(
            norm,
            out_size=self.z_cfg.out_size,
            degree=self.z_cfg.degree,
            radius=self.z_cfg.radius,
            grid=self.z_cfg.grid,
            pad=self.z_cfg.pad,
        )

        fh = img_process.hog_descriptor(norm)

        holes = img_process.compute_hole_feats_for_image(
            hole_k_shape=self.hole_cfg.k_shape,
        )

        return Features(
            zernike=np.asarray(fz, dtype=np.float32),
            hog=np.asarray(fh, dtype=np.float32),
            holes=[np.asarray(h, dtype=np.float32) for h in holes],
        )

    def parallel_run(self, X: np.ndarray) -> list[Features]:
        print("----------Get features----------")

        results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self.extract_one)(img) for img in X
        )

        return results


def pad_holes(
    hole_feat_lists: Sequence[list[np.ndarray]],
    hole_cfg: HoleConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts ragged hole feature lists into padded tensor format.

    Returns
    -------
    HF:
        Shape (N, Hmax, k), padded hole features.
    M:
        Shape (N, Hmax), mask where 1 indicates a valid hole.
    """
    N = len(hole_feat_lists)
    Hmax = hole_cfg.Hmax
    k = hole_cfg.feature_length

    HF = np.zeros((N, Hmax, k), dtype=np.float32)
    M = np.zeros((N, Hmax), dtype=np.float32)

    for i, holes in enumerate(hole_feat_lists):
        m = min(len(holes), Hmax)

        if m > 0:
            arr = np.asarray(holes[:m], dtype=np.float32)

            if arr.shape[-1] != k:
                raise ValueError(
                    f"Hole feature length mismatch at sample {i}: "
                    f"expected {k}, got {arr.shape[-1]}"
                )

            HF[i, :m, :] = arr
            M[i, :m] = 1.0

    return HF, M


def flatten_holes(
    hole_feat_lists: Sequence[list[np.ndarray]],
    hole_cfg: HoleConfig,
) -> np.ndarray:
    """
    Flattens all per-image hole feature lists into one matrix for fitting
    the hole feature scaler/quantizer.

    Returns shape (num_total_holes, k).
    """
    k = hole_cfg.feature_length
    all_hole_feats = [f for holes in hole_feat_lists for f in holes]

    if len(all_hole_feats) == 0:
        return np.zeros((0, k), dtype=np.float32)

    HFEATS = np.vstack(all_hole_feats).astype(np.float32)

    if HFEATS.shape[1] != k:
        raise ValueError(
            f"Flattened hole feature length mismatch: expected {k}, got {HFEATS.shape[1]}"
        )

    return HFEATS


def compute_features(
    X: np.ndarray,
    z_cfg: ZernikeConfig,
    hole_cfg: HoleConfig,
    n_jobs: int | None = None,
) -> FeatureBatch:
    """
    Extracts Zernike, HOG, and hole features for a dataset split.

    Parameters
    ----------
    X:
        Images, shape (N, H, W).
    z_cfg:
        Zernike/spatial-pyramid configuration.
    hole_cfg:
        Hole feature configuration.
    n_jobs:
        Number of CPU workers for parallel extraction.

    Returns
    -------
    FeatureBatch
    """
    feature_extractor = FeatureExtractor(
        z_cfg=z_cfg,
        hole_cfg=hole_cfg,
        n_jobs=n_jobs,
    )

    results = feature_extractor.parallel_run(X=X)

    Fz = np.vstack([r.zernike for r in results]).astype(np.float32)
    Fh = np.vstack([r.hog for r in results]).astype(np.float32)
    holes = [r.holes for r in results]

    HFEATS = flatten_holes(
        hole_feat_lists=holes,
        hole_cfg=hole_cfg,
    )

    HF, mask = pad_holes(
        hole_feat_lists=holes,
        hole_cfg=hole_cfg,
    )

    return FeatureBatch(
        Fz=Fz,
        Fh=Fh,
        HF=HF,
        M=mask,
        HFEATS=HFEATS,
    )
