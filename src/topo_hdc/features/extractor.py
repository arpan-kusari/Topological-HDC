# src/topo_hdc/features/extractor.py

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed, dump, load
from topo_hdc.features.Image_processing import ProcessImage


from topo_hdc.config import HoleConfig, ZernikeConfig, PersistenceImageConfig
from gtda.homology import CubicalPersistence
from gtda.images import HeightFiltration
from gtda.diagrams import PersistenceImage

# Define this at the module level (top-level) so it is pickleable!
def identity_persistence_weight(x):
    return x


@dataclass
class Features:
    hog: np.ndarray              # shape: (Fh,)
    fourier_mellin: np.ndarray   # shape: 
    persistent_image: np.ndarray # shape:


@dataclass
class FeatureBatch:
    Fh: np.ndarray               # shape: (N, Fh)
    Ffm: np.ndarray                  # shape: (N, Ffm)
    Fpi: np.ndarray                  # shape: (N, Fpi)


class FeatureExtractor:
    def __init__(
        self,
        z_cfg: ZernikeConfig,
        hole_cfg: HoleConfig,
        ablation_cfg: RobustHoleConfig, 
        persistence_cfg: PersistenceImageConfig,
        n_jobs: int | None = None,
    ) -> None:
        self.z_cfg = z_cfg
        self.hole_cfg = hole_cfg
        self.ablation_cfg = ablation_cfg
        self.persistence_cfg = persistence_cfg
        self.n_jobs = n_jobs if n_jobs is not None else os.cpu_count()

    def extract_one(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        img_process = ProcessImage(img)

        norm = img_process.normalize_glyph_gray(
            out_size=self.z_cfg.out_size,
            pad=self.z_cfg.pad,
        )

        fh = img_process.hog_descriptor(norm)
        ffm = img_process.fourier_mellin_descriptor(out_size=self.z_cfg.out_size)
        
        # Define directions explicitly as 1D numpy arrays
        direction_list = [
            np.array([1, 0]), 
            np.array([0, 1]), 
            np.array([1, 1]), 
            np.array([1, -1])
        ]

        # Dynamically extract the resolution (e.g., 64, 64)
        img_shape = norm.shape  
        single_image = norm.reshape((1,) + img_shape)
        
        filtered_channels = []
        
        for d in direction_list:
            hf = HeightFiltration(direction=d)
            filtered_channels.append(hf.fit_transform(single_image))
            
        return (
            np.asarray(fh, dtype=np.float32), 
            np.asarray(ffm, dtype=np.float32), 
            np.array(filtered_channels, dtype=np.float32)
        )
        
        
    def parallel_run(self, X: np.ndarray, fit: bool=True, cache_dir: str=None) -> list[Features]:
        print("----------Get features----------")

        # 1. Parallel collection of HOG, FM, and raw 4-channel filtered images
        raw_results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self.extract_one)(img) for img in X
        )

        # Unpack columns
        hog_list, fm_list, filtered_images_list = zip(*raw_results)

        # 2. Collapse the sample and direction axes into a single 3D batch array
        # If len(X) is 48000, this creates an array of shape (192000, 28, 28)
        X_filtered_flat = np.vstack(filtered_images_list)

        print("----------Computing Global Persistent Homology Batch----------")
        
        # 3. Running globally forces giotto-tda to pad varying rows automatically,
        # ensuring a perfectly uniform dense 3D numpy array output.
        cp = CubicalPersistence(homology_dimensions=[0, 1], n_jobs=-1)
        global_diagrams = cp.fit_transform(X_filtered_flat)

        # Path where the training grid metadata will be stored
        model_path = os.path.join(cache_dir, "tda_pi_vectorizer.pkl") if cache_dir else "tda_pi_vectorizer.pkl"

        if fit:
            print("Fitting and saving global TDA coordinate grid...")
            pi_vectorizer = PersistenceImage(
                n_bins=self.persistence_cfg.nbins,  
                sigma=self.persistence_cfg.sigma,  
                weight_function=identity_persistence_weight,
            )
            global_pi_features = pi_vectorizer.fit_transform(global_diagrams)
            
            # Save the trained state to disk so later splits can read it
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            dump(pi_vectorizer, model_path)
        else:
            print(f"Loading established training coordinate grid from: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Could not find a fitted TDA vectorizer at {model_path}. "
                    "Make sure the 'train' split runs and saves its state before validation/test splits run."
                )
            pi_vectorizer = load(model_path)
            global_pi_features = pi_vectorizer.transform(global_diagrams)
        # Reshape back to a flat feature row per sample
        n_samples = len(X)
        flat_tda_features = global_pi_features.reshape(n_samples, -1)
        
        # Assemble final clean list of Features objects
        final_features = []
        for i in range(n_samples):
            final_features.append(
                Features(
                    hog=hog_list[i],
                    fourier_mellin=fm_list[i],
                    persistent_image=np.asarray(flat_tda_features[i], dtype=np.float32),
                )
            )

        return final_features

        
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


def pad_holes(
    hole_feat_lists: list[list[np.ndarray]],
    hole_cfg: HoleConfig,
    ablation_cfg: RobustHoleConfig,
) -> tuple[np.ndarray, np.ndarray]:
    feature_length = hole_feature_length(
        hole_k_shape=hole_cfg.k_shape,
        ablation_cfg=ablation_cfg,
    )

    N = len(hole_feat_lists)

    HF = np.zeros(
        (N, hole_cfg.Hmax, feature_length),
        dtype=np.float32,
    )
    M = np.zeros(
        (N, hole_cfg.Hmax),
        dtype=np.float32,
    )

    for i, holes in enumerate(hole_feat_lists):
        m = min(len(holes), hole_cfg.Hmax)

        if m > 0:
            values = np.asarray(
                holes[:m],
                dtype=np.float32,
            )

            if values.shape[1] != feature_length:
                raise ValueError(
                    f"Sample {i}: expected hole feature length "
                    f"{feature_length}, got {values.shape[1]}."
                )

            HF[i, :m] = values
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
    ablation_cfg: RobustHoleConfig,
    persistence_cfg: PersistenceImageConfig,
    n_jobs: int | None = None,
    fit=True,
    cache_dir=None,
) -> FeatureBatch:
    """
    Extracts Zernike, HOG, Fourier-Mellin, Persistent Image, and hole features for a dataset split.

    Parameters
    ----------
    X:
        Images, shape (N, H, W).
    z_cfg:
        Zernike/spatial-pyramid configuration.
    hole_cfg:
        Hole feature configuration.
    ablation_cfg:
        Robust topology and scale parameters.
    n_jobs:
        Number of CPU workers for parallel extraction.

    Returns
    -------
    FeatureBatch
    """
    feature_extractor = FeatureExtractor(
        z_cfg=z_cfg,
        hole_cfg=hole_cfg,
        ablation_cfg=ablation_cfg,
        persistence_cfg=persistence_cfg,
        n_jobs=n_jobs,
    )

    results = feature_extractor.parallel_run(X=X, fit=fit, cache_dir=cache_dir)

    Fh = np.vstack([r.hog for r in results]).astype(np.float32)
    Ffm = np.vstack([r.fourier_mellin for r in results]).astype(np.float32)
    Fpi = np.vstack([r.persistent_image for r in results]).astype(np.float32)

   
    return FeatureBatch(
        Fh=Fh,
        Ffm=Ffm,
        Fpi=Fpi,
    )
