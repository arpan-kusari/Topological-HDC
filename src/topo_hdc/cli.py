# src/topo_hdc/cli.py

import argparse
from pathlib import Path

from topo_hdc.config import (
    RunConfig,
    DatasetConfig,
    CorruptionConfig,
    ZernikeConfig,
    HoleConfig,
    FusionConfig,
    PlotConfig,
    FeatureConfig,
    ExperimentConfig,
    validate_config,
)
from topo_hdc.run import run_experiment
from topo_hdc.ablations import ABLATION_PRESETS, get_ablation_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Topology-guided HDC offline experiment runner"
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--num-classes", type=int, default=10)
    

    # Run settings
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dim", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument(
        "--force-recompute-prototypes",
        action="store_true",
        help="Ignore saved prototypes and retrain them.",
    )

    # Corruption
    parser.add_argument(
        "--corrupt",
        type=str,
        default="none",
        choices=["none", "rotation", "gaussian", "saltpepper", "cutout", "zoom"],
    )
    parser.add_argument("--angle-deg", type=float, default=20.0)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--saltpepper-p", type=float, default=0.1)
    parser.add_argument("--cutout-size", type=int, default=12)
    parser.add_argument("--zoom", type=float, default=0.75)

    # Hole / topology settings
    parser.add_argument("--k-shape", type=int, default=12)
    parser.add_argument("--hole-feature-length", type=int, default=39)
    parser.add_argument("--hole-q", type=int, default=101)
    parser.add_argument("--hmax", type=int, default=4)
    
    # Zernike settings
    parser.add_argument("--out-size", type=int, default=64)
    parser.add_argument("--degree", type=int, default=6)
    parser.add_argument("--radius", type=float, default=20)
    parser.add_argument("--grid", type=tuple, default=(2,2))
    parser.add_argument("--pad", type=int, default=2)
    
    # Fusion settings
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--beta-step", type=float, default=0.1)

    # Output/cache
    parser.add_argument("--out-dir", type=str, default="runs/mnist")
    parser.add_argument("--cache-dir", type=str, default="cache/features")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    
    # Plot settings
    parser.add_argument("--plot-figsize", type=tuple, default=(16,12))
    parser.add_argument("--plot-fontsize", type=int, default=20)
    parser.add_argument("--plot-before-title", type=str, default="Classification Matrix for Test Set before Training")
    parser.add_argument("--plot-after-title", type=str, default="Classification Matrix for Test Set after Training")
    parser.add_argument("--plot-before-filename", type=str, default='figures/mnist_before_train.png')
    parser.add_argument("--plot-after-filename", type=str, default='figures/mnist_after_train.png')
    
    #Ablation 
    parser.add_argument(
        "--ablation",
        type=str,
        default="A3_spoke_pyramid",
        choices=sorted(ABLATION_PRESETS),
        help="Hole-extraction ablation preset.",
    )
    

    parser.add_argument(
        "--force-recompute-features",
        action="store_true",
    )
    

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_cfg = RunConfig(
        seed=args.seed,
        val_split=args.val_split,
        device=args.device,
        dim=args.dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        force_recompute_prototypes=args.force_recompute_prototypes,
    )

    data_cfg = DatasetConfig(
        dataset=args.dataset,
        num_classes=args.num_classes,
    )

    corrupt_cfg = CorruptionConfig(
        kind=args.corrupt,
        angle_deg=args.angle_deg,
        sigma=args.sigma,
        p=args.saltpepper_p,
        cutout_size=args.cutout_size,
        zoom=args.zoom,
    )

    zernike_cfg = ZernikeConfig(
        out_size=args.out_size,
        degree=args.degree,
        radius=args.radius,
        grid=args.grid,
        pad=args.pad,
    )

    hole_cfg = HoleConfig(
        k_shape=args.k_shape,
        feature_length=args.hole_feature_length,
        Q=args.hole_q,
        Hmax=args.hmax,
    )
    
    fusion_cfg = FusionConfig(
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max, 
        alpha_step=args.alpha_step,
        beta_min=args.beta_min, 
        beta_max=args.beta_max,
        beta_step=args.beta_step,
    )
    
    plot_cfg = PlotConfig(
        figsize=args.plot_figsize,
        fontsize=args.plot_fontsize,
        before_title=args.plot_before_title,
        after_title=args.plot_after_title,
        before_filename=args.plot_before_filename,
        after_filename=args.plot_after_filename,
    )
    
    ablation_cfg = get_ablation_config(args.ablation)
    
    feature_cfg = FeatureConfig(
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
        force_recompute=args.force_recompute_features,
        n_jobs=args.n_jobs,
    )

    cfg = ExperimentConfig(
        run=run_cfg,
        data=data_cfg,
        corrupt=corrupt_cfg,
        zernike=zernike_cfg,
        holes=hole_cfg,
        fusion=fusion_cfg,
        plot=plot_cfg,
        features=feature_cfg,
        ablation_name=args.ablation,
        ablation=ablation_cfg,
        out_dir=Path(args.out_dir),
    )

    validate_config(cfg)

    metrics = run_experiment(cfg)

    print("\nFinished experiment.")
    print(metrics)


if __name__ == "__main__":
    main()
