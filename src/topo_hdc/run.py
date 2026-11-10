import torch
import numpy as np
from topo_hdc.data import load_mnist, load_emnist, load_medmnist, stratified_train_val_split, batch_rgb_to_grayscale
from topo_hdc.corruptions import apply_corruption
from topo_hdc.features.extractor import FeatureBatch, compute_features
from topo_hdc.hdc.encoders import Encoders, Roles, fit_encoders, get_role
from topo_hdc.utils.batching import make_hv_generators
from topo_hdc.hdc.prototypes import train_prototypes_stream_all, onlinehd_train_stream_all, load_or_train_prototypes_before
from topo_hdc.hdc.fusion import pick_alpha_beta_on_val, predict_single_channel_batched, predict_late_fusion_batched, collect_channel_scores, tune_cached_fusion, predict_from_cached_scores, tune_alpha_cached, make_early_fusion_generator
from topo_hdc.hdc.topology_compatibility import fit_topology_compatibility, topology_compatibility_scores
from topo_hdc.plot import plot_classification_heatmap
from topo_hdc.utils.io import save_run_outputs
from topo_hdc.features.cache import load_or_compute_features
from topo_hdc.metrics import accuracy
from sklearn.metrics import confusion_matrix

def run_experiment(cfg) -> dict:
    if cfg.data.dataset == "mnist":
        X_train, y_train, X_test, y_test = load_mnist()
        X_train, y_train, X_val, y_val = stratified_train_val_split(X_train, y_train, val_frac=cfg.run.val_split)
    elif cfg.data.dataset == "emnist":
        X_train, y_train, X_test, y_test = load_emnist()
        X_train, y_train, X_val, y_val = stratified_train_val_split(X_train, y_train, val_frac=cfg.run.val_split)
    elif cfg.data.dataset == "svhn":
        X_train, y_train, X_test, y_test = load_svhn()
        X_train = batch_rgb_to_grayscale(X_train)
        X_test = batch_rgb_to_grayscale(X_test)
        X_train = svhn_normalize_for_hog(X_train)
        X_test = svhn_normalize_for_hog(X_test)
    elif cfg.data.dataset == "medmnist":
        X_train, y_train, X_val, y_val, X_test, y_test = load_medmnist()
        X_train = batch_rgb_to_grayscale(X_train)
        X_val = batch_rgb_to_grayscale(X_val)
        X_test = batch_rgb_to_grayscale(X_test)
    X_test = apply_corruption(X=X_test, cfg=cfg.corrupt)
    
    train_feats = load_or_compute_features(
        X=X_train,
        cache_root=cfg.features.cache_dir,
        dataset_name=cfg.data.dataset,
        split_name="train",
        split_seed=cfg.run.seed,
        zernike_cfg=cfg.zernike,
        hole_cfg=cfg.holes,
        ablation_cfg=cfg.ablation,
        corruption_cfg=None,
        persistence_cfg=cfg.persistence,
        compute_fn=lambda: compute_features(
            X=X_train,
            z_cfg=cfg.zernike,
            hole_cfg=cfg.holes,
            ablation_cfg=cfg.ablation,
            persistence_cfg=cfg.persistence,
            n_jobs=cfg.run.n_jobs,
            fit=True,
            cache_dir=cfg.features.cache_dir,
        ),
        use_cache=cfg.features.use_cache,
        force_recompute=cfg.features.force_recompute,
    )
    

    val_feats = load_or_compute_features(
        X=X_val,
        cache_root=cfg.features.cache_dir,
        dataset_name=cfg.data.dataset,
        split_name="validation",
        split_seed=cfg.run.seed,
        zernike_cfg=cfg.zernike,
        hole_cfg=cfg.holes,
        ablation_cfg=cfg.ablation,
        corruption_cfg=None,
        persistence_cfg=cfg.persistence,
        compute_fn=lambda: compute_features(
            X=X_val,
            z_cfg=cfg.zernike,
            hole_cfg=cfg.holes,
            ablation_cfg=cfg.ablation,
            persistence_cfg=cfg.persistence,
            n_jobs=cfg.run.n_jobs,
            fit=False,
            cache_dir=cfg.features.cache_dir,
        ),
        use_cache=cfg.features.use_cache,
        force_recompute=cfg.features.force_recompute,
    )
    
    test_feats = load_or_compute_features(
        X=X_test,
        cache_root=cfg.features.cache_dir,
        dataset_name=cfg.data.dataset,
        split_name="test",
        split_seed=cfg.run.seed,
        zernike_cfg=cfg.zernike,
        hole_cfg=cfg.holes,
        ablation_cfg=cfg.ablation,
        corruption_cfg=cfg.corrupt,
        persistence_cfg=cfg.persistence,
        compute_fn=lambda: compute_features(
            X=X_test,
            z_cfg=cfg.zernike,
            hole_cfg=cfg.holes,
            ablation_cfg=cfg.ablation,
            persistence_cfg=cfg.persistence,
            n_jobs=cfg.run.n_jobs,
            fit=False,
            cache_dir=cfg.features.cache_dir,
        ),
        use_cache=cfg.features.use_cache,
        force_recompute=cfg.features.force_recompute,
    )
    
    train_encoders = fit_encoders(feats=train_feats, run_cfg=cfg.run, hole_cfg=cfg.holes, ablation_cfg=cfg.ablation)
    val_encoders = fit_encoders(feats=val_feats, run_cfg=cfg.run, hole_cfg=cfg.holes, ablation_cfg=cfg.ablation)
    test_encoders = fit_encoders(feats=test_feats, run_cfg=cfg.run, hole_cfg=cfg.holes, ablation_cfg=cfg.ablation)
    roles = get_role(run_cfg=cfg.run)
    train_generators = make_hv_generators(features=train_feats, encoders=train_encoders, roles=roles, batch_size=cfg.run.batch_size)
    val_generators = make_hv_generators(features=val_feats, encoders=val_encoders, roles=roles, batch_size=cfg.run.batch_size)
    test_generators = make_hv_generators(features=test_feats, encoders=test_encoders, roles=roles, batch_size=cfg.run.batch_size)
    
#    # Isolate your target channels into an early-fused joint hypervector stream
#    fusion_channels = ["fourier_mellin", "persistent_image"]
#    train_joint_generator = make_early_fusion_generator(train_generators, fusion_channels)
#    val_joint_generator = make_early_fusion_generator(val_generators, fusion_channels)
#    test_joint_generator = make_early_fusion_generator(test_generators, fusion_channels)
#    
#    # Wrap the single stream into a container format expected by your prototype modules
#    train_generators = {"early_fusion": train_joint_generator}
#    test_generators = {"early_fusion": test_joint_generator}
    
    protos_before, before_metadata = load_or_train_prototypes_before(
        out_dir=cfg.out_dir,
        train_generators=train_generators,
        y_train=y_train,
        run_cfg=cfg.run,
        data_cfg=cfg.data,
        force_recompute=cfg.run.force_recompute_prototypes,
    )
    val_channel_scores_before = collect_channel_scores(
        generators=val_generators,
        protos=protos_before,
        num_samples=len(y_val),
    )

    best_before, fusion_grid_before = tune_cached_fusion(
        channel_scores=val_channel_scores_before,
        y_val=y_val,
        run_cfg=cfg.run,
    )
    
    test_scores_before = collect_channel_scores(
        generators=test_generators,
        protos=protos_before,
        num_samples=len(y_test),
    )

    y_pred_before = predict_from_cached_scores(
        channel_scores=test_scores_before,
        alpha=best_before["alpha"],
        beta=best_before["beta"],
    )
    single_channel_before = {}

    for channel in ("hog", "fourier_mellin", "persistent_image"):
        channel_pred = predict_single_channel_batched(
            generator_fn=test_generators[channel],
            prototypes=protos_before[channel],
            num_samples=len(y_test),
        )

        single_channel_before[channel] = {
            "accuracy": float((channel_pred == y_test).mean()),
            "predictions": channel_pred,
        }

        print(
            f"{channel} accuracy before OnlineHD: "
            f"{single_channel_before[channel]['accuracy']:.4f}"
        )
    
    y_pred = predict_late_fusion_batched(
        generators=test_generators, 
        protos=protos_before, 
        alpha=best_before["alpha"], 
        beta=best_before["beta"], 
        N=len(y_test),
    )

    acc_before = accuracy(y_test, y_pred_before)
    print("Test accuracy:", acc_before)
    conf_matrix = confusion_matrix(y_test, y_pred_before)
    plot_classification_heatmap(conf_matrix, class_labels=range(0,cfg.data.num_classes), figsize=cfg.plot.figsize, title=cfg.plot.before_title, fig_filename=cfg.plot.before_filename, fontsize=cfg.plot.fontsize)
    
    # After OnlineHD
   
    protos_after = onlinehd_train_stream_all(
            generators=train_generators,
            y_np=y_train,
            run_cfg=cfg.run,
            data_cfg=cfg.data,
            init_protos=protos_before,
        )
    
    val_scores_after = collect_channel_scores(
        generators=val_generators,
        protos=protos_after,
        num_samples=len(y_val),
    )

    best_after, fusion_grid_after = tune_cached_fusion(
        channel_scores=val_scores_after,
        y_val=y_val,
        run_cfg=cfg.run,
    )

    test_scores_after = collect_channel_scores(
        generators=test_generators,
        protos=protos_after,
        num_samples=len(y_test),
    )

    y_pred_after = predict_from_cached_scores(
        channel_scores=test_scores_after,
        alpha=best_after["alpha"],
        beta=best_after["beta"],
    )

    single_channel_after = {}

    for channel in ("hog", "fourier_mellin", "persistent_image"):
        channel_pred = predict_single_channel_batched(
            generator_fn=test_generators[channel],
            prototypes=protos_after[channel],
            num_samples=len(y_test),
        )

        single_channel_after[channel] = {
            "accuracy": float((channel_pred == y_test).mean()),
            "predictions": channel_pred,
        }

        print(
            f"{channel} accuracy after OnlineHD: "
            f"{single_channel_after[channel]['accuracy']:.4f}"
        )
    
    y_pred = predict_late_fusion_batched(
        generators=test_generators, 
        protos=protos_after, 
        alpha=best_after["alpha"], 
        beta=best_after["beta"], 
        N=len(y_test),
    )
    acc_after = accuracy(y_test, y_pred_after)
    print("Test accuracy:", acc_after)
    conf_matrix = confusion_matrix(y_test, y_pred_after)
    plot_classification_heatmap(conf_matrix, class_labels=range(0, cfg.data.num_classes), figsize=cfg.plot.figsize, title=cfg.plot.after_title, fig_filename=cfg.plot.after_filename, fontsize=cfg.plot.fontsize)
    metrics = {
        "acc_before": acc_before,
        "hog_acc_before": single_channel_before["hog"]['accuracy'],
        "fourier_mellin_acc_before": single_channel_before["fourier_mellin"]['accuracy'],
        "pi_acc_before": single_channel_before["persistent_image"]['accuracy'],
        "hog_acc_after": single_channel_after["hog"]['accuracy'],
        "fourier_mellin_acc_after": single_channel_after["fourier_mellin"]['accuracy'],
        "pi_acc_after": single_channel_after["persistent_image"]['accuracy'],
        "acc_after": acc_after,
        "alpha_before": best_before["alpha"],
        "beta_before": best_before["beta"],
        "alpha_after": best_after["alpha"],
        "beta_after": best_after["beta"],
        "num_train": X_train.shape[0],
        "num_val": X_val.shape[0],
        "num_test": X_test.shape[0]
    }
    save_run_outputs(
        cfg.out_dir,
        cfg,
        metrics,
        best_before["alpha"],
        best_before["beta"],
        best_after["alpha"],
        best_after["beta"],
        train_encoders,
        protos_before,
        protos_after,
    )
    
    return metrics
          
    
    
