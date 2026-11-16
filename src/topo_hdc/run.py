# src/topo_hdc/run.py

from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from topo_hdc.config import validate_config
from topo_hdc.ablations import select_pi_features
from topo_hdc.data import (
    load_mnist,
    load_emnist,
    load_medmnist,
    stratified_train_val_split,
    batch_rgb_to_grayscale,
)
from topo_hdc.corruptions import apply_corruption
from topo_hdc.features.extractor import FeatureBatch, compute_features
from topo_hdc.features.cache import load_or_compute_features
from topo_hdc.hdc.encoders import fit_encoders, get_role
from topo_hdc.utils.batching import make_hv_generators
from topo_hdc.hdc.prototypes import (
    load_or_train_prototypes_before,
    onlinehd_train_stream_all,
)
from topo_hdc.hdc.fusion import collect_channel_scores
from topo_hdc.plot import plot_classification_heatmap
from topo_hdc.utils.io import save_run_outputs


CHANNEL_FLAGS = {
    "hog": "use_hog",
    "fourier_mellin": "use_fm",
    "persistent_image": "use_pi",
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _digest(value: dict) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()[:16]


def _active_channels(cfg) -> tuple[str, ...]:
    return tuple(
        channel
        for channel, flag in CHANNEL_FLAGS.items()
        if getattr(cfg.ablation, flag)
    )


def _load_data(cfg):
    if cfg.data.dataset == "mnist":
        X_train, y_train, X_test, y_test = load_mnist()

        X_train, y_train, X_val, y_val = stratified_train_val_split(
            X_train,
            y_train,
            val_frac=cfg.run.val_split,
        )

    elif cfg.data.dataset == "emnist":
        X_train, y_train, X_test, y_test = load_emnist()

        X_train, y_train, X_val, y_val = stratified_train_val_split(
            X_train,
            y_train,
            val_frac=cfg.run.val_split,
        )

    elif cfg.data.dataset == "medmnist":
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        ) = load_medmnist()

        X_train = batch_rgb_to_grayscale(X_train)
        X_val = batch_rgb_to_grayscale(X_val)
        X_test = batch_rgb_to_grayscale(X_test)

    elif cfg.data.dataset == "svhn":
        raise NotImplementedError(
            "The supplied run module does not define/import load_svhn "
            "or svhn_normalize_for_hog. Add your SVHN loader and create "
            "a training/validation split before enabling this branch."
        )

    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")

    y_train = np.asarray(y_train).reshape(-1)
    y_val = np.asarray(y_val).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    for split_name, X, y in (
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ):
        if len(X) != len(y):
            raise ValueError(
                f"{split_name}: image and label counts differ."
            )

        if not np.isin(y, np.arange(cfg.data.num_classes)).all():
            raise ValueError(
                f"{split_name}: labels must be in "
                f"0..{cfg.data.num_classes - 1}. "
                "Check dataset-specific label remapping."
            )

    return X_train, y_train, X_val, y_val, X_test, y_test


def _select_features(features: FeatureBatch, cfg) -> FeatureBatch:
    """
    Select PI subsets before encoder fitting.

    Disabled channels retain their arrays for compatibility with existing
    FeatureBatch-based encoder code, but their generators are removed
    before prototype training and scoring.
    """
    if cfg.ablation.use_pi:
        Fpi = select_pi_features(
            features.Fpi,
            n_bins=cfg.persistence.nbins,
            config=cfg.ablation,
        )
    else:
        Fpi = features.Fpi

    return FeatureBatch(
        Fh=features.Fh,
        Ffm=features.Ffm,
        Fpi=Fpi,
    )


def _make_active_generators(features, encoders, roles, cfg):
    generators = make_hv_generators(
        features=features,
        encoders=encoders,
        roles=roles,
        batch_size=cfg.run.batch_size,
    )

    active = _active_channels(cfg)
    missing = set(active) - set(generators)

    if missing:
        raise ValueError(
            f"Hypervector generators are missing channels: {sorted(missing)}"
        )

    return {channel: generators[channel] for channel in active}


def _scores_to_numpy(channel_scores) -> dict[str, np.ndarray]:
    result = {}

    for channel, scores in channel_scores.items():
        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()

        scores = np.asarray(scores)

        if scores.ndim != 2:
            raise ValueError(
                f"{channel} scores must have shape (N, C); "
                f"got {scores.shape}."
            )

        if not np.isfinite(scores).all():
            raise ValueError(f"{channel} scores contain nonfinite values.")

        result[channel] = scores

    if not result:
        raise ValueError("No channel scores were produced.")

    shapes = {scores.shape for scores in result.values()}

    if len(shapes) != 1:
        raise ValueError("Channel score arrays have inconsistent shapes.")

    return result


def _predict_weighted(channel_scores, weights) -> np.ndarray:
    fused = None

    for channel, scores in channel_scores.items():
        contribution = float(weights[channel]) * scores
        fused = contribution if fused is None else fused + contribution

    if fused is None:
        raise ValueError("No active channels to fuse.")

    return np.argmax(fused, axis=1)


def _select_fusion_weights(channel_scores, y_val, cfg):
    """
    Fix one active channel's weight to 1 and tune the others.

    Anchor priority: HOG, Fourier--Mellin, PI.
    Thus, FM + PI fixes FM=1 and tunes PI over run.betas.
    """
    channels = tuple(
        channel
        for channel in CHANNEL_FLAGS
        if channel in channel_scores
    )

    if cfg.ablation.fusion_mode == "equal" or len(channels) == 1:
        weights = {channel: 1.0 for channel in channels}
        predictions = _predict_weighted(channel_scores, weights)

        best = {
            "weights": weights,
            "alpha": weights.get("fourier_mellin", 0.0),
            "beta": weights.get("persistent_image", 0.0),
            "val_accuracy": float(np.mean(predictions == y_val)),
        }
        return best, [best]

    anchor = channels[0]
    tunable = channels[1:]

    grids = {
        "fourier_mellin": cfg.run.alphas,
        "persistent_image": cfg.run.betas,
    }

    best = None
    records = []

    for values in itertools.product(*(grids[ch] for ch in tunable)):
        weights = {anchor: 1.0}
        weights.update(zip(tunable, map(float, values)))

        predictions = _predict_weighted(channel_scores, weights)
        val_accuracy = float(np.mean(predictions == y_val))

        record = {
            "weights": weights,
            "alpha": weights.get("fourier_mellin", 0.0),
            "beta": weights.get("persistent_image", 0.0),
            "val_accuracy": val_accuracy,
        }
        records.append(record)

        # Deterministic tie handling: retain the first grid candidate.
        if best is None or val_accuracy > best["val_accuracy"]:
            best = record

    return best, records


def _single_channel_metrics(channel_scores, y_true):
    return {
        channel: {
            "accuracy": float(
                np.mean(np.argmax(scores, axis=1) == y_true)
            ),
        }
        for channel, scores in channel_scores.items()
    }


def _evaluate_stage(
    *,
    protos,
    val_generators,
    test_generators,
    y_val,
    y_test,
    cfg,
):
    val_scores = _scores_to_numpy(
        collect_channel_scores(
            generators=val_generators,
            protos=protos,
            num_samples=len(y_val),
        )
    )

    best, fusion_grid = _select_fusion_weights(
        val_scores, y_val, cfg
    )

    test_scores = _scores_to_numpy(
        collect_channel_scores(
            generators=test_generators,
            protos=protos,
            num_samples=len(y_test),
        )
    )

    predictions = _predict_weighted(
        test_scores,
        best["weights"],
    )

    return {
        "accuracy": float(np.mean(predictions == y_test)),
        "predictions": predictions,
        "best": best,
        "fusion_grid": fusion_grid,
        "single_channels": _single_channel_metrics(
            test_scores, y_test
        ),
    }


def _plot_stage(y_true, predictions, cfg, out_dir, *, after):
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=np.arange(cfg.data.num_classes),
    )

    filename = (
        cfg.plot.after_filename
        if after
        else cfg.plot.before_filename
    )
    destination = out_dir / "figures" / Path(filename).name
    destination.parent.mkdir(parents=True, exist_ok=True)

    plot_classification_heatmap(
        matrix,
        class_labels=range(cfg.data.num_classes),
        figsize=cfg.plot.figsize,
        title=(
            cfg.plot.after_title
            if after
            else cfg.plot.before_title
        ),
        fig_filename=str(destination),
        fontsize=cfg.plot.fontsize,
    )


def run_experiment(cfg) -> dict:
    validate_config(cfg)
    _seed_everything(cfg.run.seed)

    active_channels = _active_channels(cfg)

    print(f"Ablation: {cfg.ablation_name}")
    print(f"Active channels: {active_channels}")

    # Isolate outputs/prototypes by configuration.
    run_identity = asdict(cfg)
    run_identity.pop("out_dir", None)
    run_key = _digest(run_identity)
    run_identity["features"].pop("force_recompute", None)
    run_identity["features"].pop("force_recompute_test", None)
    run_identity["run"].pop("force_recompute_prototypes", None)

    out_dir = Path(cfg.out_dir) / f"run_{run_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = _load_data(cfg)

    # Corruption is applied only to test images in this protocol.
    X_test = apply_corruption(X=X_test, cfg=cfg.corrupt)

    # Isolate learned PI bounds by training/extraction identity.
    # Include an implementation version to invalidate incompatible bounds.
    bounds_identity = {
        "implementation_version": 1,
        "dataset": asdict(cfg.data),
        "seed": cfg.run.seed,
        "val_split": cfg.run.val_split,
        "normalization": asdict(cfg.img_process),
        "persistence": asdict(cfg.persistence),
        "ablation": asdict(cfg.ablation),
        "full_pi_directions": (0, 1, 2, 3),
        "full_pi_homology": (0, 1),
    }
    bounds_key = _digest(bounds_identity)

    # Separate full-feature caches from older/reduced-feature caches.
    feature_cache_root = (
        Path(cfg.features.cache_dir)
        / "full_pi_v1"
        / bounds_key
    )
    bounds_dir = feature_cache_root / "pi_bounds"
    bounds_path = bounds_dir / "tda_global_bounds.pkl"

    # If training features are cached but fitted bounds are missing,
    # recompute training features so validation/test can transform safely.
    force_train = (
        cfg.features.force_recompute
        or not bounds_path.is_file()
    )

    n_jobs = (
        cfg.features.n_jobs
        if cfg.features.n_jobs is not None
        else cfg.run.n_jobs
    )

    def extract_split(X, split_name, *, fit, corruption_cfg):
        return load_or_compute_features(
            X=X,
            cache_root=feature_cache_root,
            dataset_name=cfg.data.dataset,
            split_name=split_name,
            split_seed=cfg.run.seed,
            img_cfg=cfg.img_process,
            ablation_cfg=cfg.ablation,
            corruption_cfg=corruption_cfg,
            persistence_cfg=cfg.persistence,
            compute_fn=lambda: compute_features(
                X=X,
                img_cfg=cfg.img_process,
                ablation_cfg=cfg.ablation,
                persistence_cfg=cfg.persistence,
                n_jobs=n_jobs,
                fit=fit,
                cache_dir=bounds_dir,
            ),
            use_cache=cfg.features.use_cache,
            force_recompute=(
                cfg.features.force_recompute
                or force_train
                or (
                    split_name == "test"
                    and cfg.features.force_recompute_test
                )
            ),
        )

    train_full = extract_split(
        X_train,
        "train",
        fit=True,
        corruption_cfg=None,
    )
    val_full = extract_split(
        X_val,
        "validation",
        fit=False,
        corruption_cfg=None,
    )
    test_full = extract_split(
        X_test,
        "test",
        fit=False,
        corruption_cfg=cfg.corrupt,
    )

    train_feats = _select_features(train_full, cfg)
    val_feats = _select_features(val_full, cfg)
    test_feats = _select_features(test_full, cfg)

    # IMPORTANT: fit once on training features.
    # Validation/test must use identical scalers and projection matrices.
    encoders = fit_encoders(
        feats=train_feats,
        run_cfg=cfg.run,
        ablation_cfg=cfg.ablation,
    )
    roles = get_role(run_cfg=cfg.run)

    train_generators = _make_active_generators(
        train_feats, encoders, roles, cfg
    )
    val_generators = _make_active_generators(
        val_feats, encoders, roles, cfg
    )
    test_generators = _make_active_generators(
        test_feats, encoders, roles, cfg
    )

    protos_before, before_metadata = load_or_train_prototypes_before(
        out_dir=out_dir,
        train_generators=train_generators,
        y_train=y_train,
        run_cfg=cfg.run,
        data_cfg=cfg.data,
        force_recompute=cfg.run.force_recompute_prototypes,
    )

    before = _evaluate_stage(
        protos=protos_before,
        val_generators=val_generators,
        test_generators=test_generators,
        y_val=y_val,
        y_test=y_test,
        cfg=cfg,
    )

    print(f"Test accuracy before OnlineHD: {before['accuracy']:.4f}")
    print(f"Fusion weights before: {before['best']['weights']}")

    _plot_stage(
        y_test,
        before["predictions"],
        cfg,
        out_dir,
        after=False,
    )

    if cfg.ablation.use_onlinehd:
        # Protect the original prototypes if OnlineHD updates in place.
        protos_after = onlinehd_train_stream_all(
            generators=train_generators,
            y_np=y_train,
            run_cfg=cfg.run,
            data_cfg=cfg.data,
            init_protos=copy.deepcopy(protos_before),
        )

        after = _evaluate_stage(
            protos=protos_after,
            val_generators=val_generators,
            test_generators=test_generators,
            y_val=y_val,
            y_test=y_test,
            cfg=cfg,
        )

        print(f"Test accuracy after OnlineHD: {after['accuracy']:.4f}")
        print(f"Fusion weights after: {after['best']['weights']}")

        _plot_stage(
            y_test,
            after["predictions"],
            cfg,
            out_dir,
            after=True,
        )
    else:
        print("OnlineHD disabled by ablation configuration.")

        # Preserve the existing output interface without rerunning scoring.
        protos_after = copy.deepcopy(protos_before)
        after = before

    metrics = {
        "ablation_name": cfg.ablation_name,
        "active_channels": list(active_channels),
        "onlinehd_applied": cfg.ablation.use_onlinehd,
        "fusion_mode": cfg.ablation.fusion_mode,
        "acc_before": before["accuracy"],
        # When disabled, this equals acc_before; onlinehd_applied is False.
        "acc_after": after["accuracy"],
        "alpha_before": before["best"]["alpha"],
        "beta_before": before["best"]["beta"],
        "alpha_after": after["best"]["alpha"],
        "beta_after": after["best"]["beta"],
        "weights_before": before["best"]["weights"],
        "weights_after": after["best"]["weights"],
        "val_acc_before": before["best"]["val_accuracy"],
        "val_acc_after": after["best"]["val_accuracy"],
        "num_train": len(y_train),
        "num_val": len(y_val),
        "num_test": len(y_test),
        "out_dir": str(out_dir),
    }

    metric_prefixes = {
        "hog": "hog",
        "fourier_mellin": "fourier_mellin",
        "persistent_image": "pi",
    }

    for channel, prefix in metric_prefixes.items():
        before_result = before["single_channels"].get(channel)
        after_result = after["single_channels"].get(channel)

        metrics[f"{prefix}_acc_before"] = (
            before_result["accuracy"]
            if before_result is not None
            else None
        )
        metrics[f"{prefix}_acc_after"] = (
            after_result["accuracy"]
            if after_result is not None
            else None
        )

    save_run_outputs(
        out_dir,
        cfg,
        metrics,
        before["best"]["alpha"],
        before["best"]["beta"],
        after["best"]["alpha"],
        after["best"]["beta"],
        encoders,
        protos_before,
        protos_after,
    )

    with (out_dir / "fusion_grids.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "before": before["fusion_grid"],
                "after": (
                    after["fusion_grid"]
                    if cfg.ablation.use_onlinehd
                    else None
                ),
            },
            handle,
            indent=2,
        )

    return metrics
