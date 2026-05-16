import torch
import numpy as np
from topo_hdc.data import load_mnist, load_emnist, stratified_train_val_split
from topo_hdc.corruptions import apply_corruption
from topo_hdc.features.extractor import FeatureBatch, compute_features
from topo_hdc.hdc.encoders import Encoders, Roles, fit_encoders, get_role
from topo_hdc.utils.batching import make_hv_generators
from topo_hdc.hdc.prototypes import train_prototypes_stream_all, onlinehd_train_stream_all
from topo_hdc.hdc.fusion import pick_alpha_beta_on_val, predict_late_fusion_batched
from topo_hdc.plot import plot_classification_heatmap
from topo_hdc.utils.io import save_run_outputs
from topo_hdc.metrics import accuracy
from sklearn.metrics import confusion_matrix

def run_experiment(cfg) -> dict:
    if cfg.data.dataset == "mnist":
        X_train, y_train, X_test, y_test = load_mnist()
    elif cfg.data.dataset == "emnist":
        X_train, y_train, X_test, y_test = load_emnist()
    X_train, y_train, X_val, y_val = stratified_train_val_split(X_train, y_train, val_frac=cfg.run.val_split)
    X_test = apply_corruption(X=X_test, cfg=cfg.corrupt)
    train_feats = compute_features(
        X=X_train,
        z_cfg=cfg.zernike,
        hole_cfg=cfg.holes,
        n_jobs=cfg.run.n_jobs,
    )

    val_feats = compute_features(
        X=X_val,
        z_cfg=cfg.zernike,
        hole_cfg=cfg.holes,
        n_jobs=cfg.run.n_jobs,
    )

    test_feats = compute_features(
        X=X_test,
        z_cfg=cfg.zernike,
        hole_cfg=cfg.holes,
        n_jobs=cfg.run.n_jobs,
    )
    
    train_encoders = fit_encoders(feats=train_feats, run_cfg=cfg.run, hole_cfg=cfg.holes)
    val_encoders = fit_encoders(feats=val_feats, run_cfg=cfg.run, hole_cfg=cfg.holes)
    test_encoders = fit_encoders(feats=test_feats, run_cfg=cfg.run, hole_cfg=cfg.holes)
    roles = get_role(run_cfg=cfg.run)
    train_generators = make_hv_generators(features=train_feats, encoders=train_encoders, roles=roles, batch_size=cfg.run.batch_size)
    val_generators = make_hv_generators(features=val_feats, encoders=val_encoders, roles=roles, batch_size=cfg.run.batch_size)
    test_generators = make_hv_generators(features=test_feats, encoders=test_encoders, roles=roles, batch_size=cfg.run.batch_size)
    # Before OnlineHD
    protos_before = train_prototypes_stream_all(
        generators=train_generators,
        y_np=y_train,
        run_cfg=cfg.run,
        data_cfg=cfg.data,
    )
    alpha_before, beta_before = pick_alpha_beta_on_val(generators=val_generators, protos=protos_before, y_val=y_val, run_cfg=cfg.run)
    
    y_pred = predict_late_fusion_batched(generators=test_generators, protos=protos_before, alpha=alpha_before, beta=beta_before, N=len(y_test))
    acc_before = accuracy(y_test, y_pred)
    print("Test accuracy:", acc_before)
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_classification_heatmap(conf_matrix, class_labels=range(0,cfg.data.num_classes), figsize=cfg.plot.figsize, title=cfg.plot.before_title, fig_filename=cfg.plot.before_filename, fontsize=cfg.plot.fontsize)
    
    # After OnlineHD
    protos_after = onlinehd_train_stream_all(
        generators=train_generators,
        y_np=y_train,
        run_cfg=cfg.run,
        data_cfg=cfg.data,
        init_protos=protos_before,
    )
    alpha_after, beta_after = pick_alpha_beta_on_val(generators=val_generators, protos=protos_after, y_val=y_val, run_cfg=cfg.run)
    y_pred = predict_late_fusion_batched(generators=test_generators, protos=protos_after, alpha=alpha_after, beta=beta_after, N=len(y_test))
    acc_after = accuracy(y_test, y_pred)
    print("Test accuracy:", acc_after)
    conf_matrix = confusion_matrix(y_test, y_pred)
    plot_classification_heatmap(conf_matrix, class_labels=range(0, cfg.data.num_classes), figsize=cfg.plot.figsize, title=cfg.plot.after_title, fig_filename=cfg.plot.after_filename, fontsize=cfg.plot.fontsize)
    metrics = {
        "acc_before": acc_before,
        "acc_after": acc_after,
        "alpha_before": alpha_before,
        "beta_before": beta_before,
        "alpha_after": alpha_after,
        "beta_after": beta_after,
        "num_train": X_train.shape[0],
        "num_val": X_val.shape[0],
        "num_test": X_test.shape[0]
    }
    save_run_outputs(
        cfg.out_dir,
        cfg,
        metrics,
        alpha_before,
        beta_before,
        alpha_after,
        beta_after,
        train_encoders,
        protos_before,
        protos_after,
    )
    return metrics
          
    
    
