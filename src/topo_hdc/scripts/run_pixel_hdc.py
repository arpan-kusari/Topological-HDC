#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.datasets import mnist

from topo_hdc.config import CorruptionConfig
from topo_hdc.corruptions import apply_corruption


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pixel-value HDC baseline with optional Gabor filtering."
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Random seeds to run, for example: --seeds 0 1 2",
    )
    parser.add_argument("--dim", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.2)

    parser.add_argument(
        "--encoding",
        choices=("random", "similar"),
        default="similar",
    )
    parser.add_argument("--percent-change", type=float, default=10.0)

    parser.add_argument(
        "--corruptions",
        nargs="+",
        choices=(
            "none",
            "rotation",
            "gaussian",
            "saltpepper",
            "cutout",
            "zoom",
        ),
        default=[
            "none",
            "rotation",
            "gaussian",
            "saltpepper",
            "cutout",
            "zoom",
        ],
        help="Corruptions to evaluate. Defaults to all six conditions.",
    )default="none",
        )
    parser.add_argument("--angle", type=float, default=20.0)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--cutout", type=int, default=12)
    parser.add_argument("--zoom", type=float, default=0.5)

    parser.add_argument("--no-gabor", action="store_true")
    parser.add_argument("--gabor-size", type=int, default=3)
    parser.add_argument("--gabor-sigma", type=float, default=4.0)
    parser.add_argument("--gabor-theta", type=float, default=45.0)
    parser.add_argument("--gabor-wavelength", type=float, default=10.0)
    parser.add_argument("--gabor-gamma", type=float, default=0.5)

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/pixel_hdc"),
    )

    return parser.parse_args()


def validate_args(args):
    if args.dim <= 0:
        raise ValueError("--dim must be positive.")

    if args.epochs < 0:
        raise ValueError("--epochs must be nonnegative.")

    if not 0 < args.val_split < 1:
        raise ValueError("--val-split must lie between 0 and 1.")

    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive.")

    if not 0 <= args.percent_change <= 100:
        raise ValueError("--percent-change must lie between 0 and 100.")

    if args.gabor_size <= 0 or args.gabor_size % 2 == 0:
        raise ValueError("--gabor-size must be a positive odd integer.")


def split_train_val(X, y, val_fraction, seed):
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=seed,
    )
    train_indices, val_indices = next(splitter.split(X, y))

    return (
        X[train_indices],
        y[train_indices],
        X[val_indices],
        y[val_indices],
    )


def make_item_memory(
    num_vectors,
    dim,
    rng,
    encoding,
    percent_change,
):
    """Create independent or sequentially correlated bipolar vectors."""
    if encoding == "random":
        return (
            rng.integers(
                0, 2,
                size=(num_vectors, dim),
                dtype=np.int8,
            ) * 2 - 1
        )

    memory = np.empty((num_vectors, dim), dtype=np.int8)
    memory[0] = (
        rng.integers(0, 2, size=dim, dtype=np.int8) * 2 - 1
    )

    num_flips = int(dim * percent_change / 100.0)

    for index in range(1, num_vectors):
        memory[index] = memory[index - 1]

        if num_flips:
            positions = rng.choice(
                dim, size=num_flips, replace=False
            )
            memory[index, positions] *= -1

    return memory


def make_gabor_kernel(args):
    if args.no_gabor:
        return None

    return cv2.getGaborKernel(
        (args.gabor_size, args.gabor_size),
        args.gabor_sigma,
        np.deg2rad(args.gabor_theta),
        args.gabor_wavelength,
        args.gabor_gamma,
        0.0,
        ktype=cv2.CV_32F,
    )


def prepare_image(image, kernel):
    """
    Quantize an image expressed in 0..255 intensity units.

    Corruption output must preserve those units. Values outside the
    valid intensity range are clipped.
    """
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a grayscale image; got {image.shape}."
        )

    if not np.isfinite(image).all():
        raise ValueError("Image contains nonfinite values.")

    image = np.clip(np.rint(image), 0, 255).astype(np.uint8)

    if kernel is not None:
        # Explicitly preserve uint8 output for this baseline.
        # Negative responses are clipped by OpenCV.
        image = cv2.filter2D(
            image,
            ddepth=-1,
            kernel=kernel,
        )

    return image


def encode_image(image, position_hvs, value_hvs, dim):
    """Bind pixel positions to intensity levels, then bundle."""
    values = image.reshape(-1).astype(np.int64)

    if len(values) != len(position_hvs):
        raise ValueError("Image size and position-memory size differ.")

    accumulator = np.zeros(dim, dtype=np.int32)

    for index in np.flatnonzero(values):
        accumulator += (
            position_hvs[index] * value_hvs[values[index]]
        )

    # Fixed tie rule: zero -> +1.
    return np.where(accumulator >= 0, 1, -1).astype(np.int8)


def encode_dataset(
    X,
    position_hvs,
    value_hvs,
    kernel,
    dim,
    name,
):
    encoded = np.empty((len(X), dim), dtype=np.int8)

    for index, image in enumerate(X):
        prepared = prepare_image(image, kernel)

        encoded[index] = encode_image(
            prepared,
            position_hvs,
            value_hvs,
            dim,
        )

        if (index + 1) % 1000 == 0 or index + 1 == len(X):
            print(f"{name}: encoded {index + 1}/{len(X)}")

    return encoded


def initialize_prototypes(encoded, labels, num_classes):
    """Bundle each training hypervector once into its class."""
    prototypes = np.zeros(
        (num_classes, encoded.shape[1]),
        dtype=np.float32,
    )

    for class_index in range(num_classes):
        prototypes[class_index] = encoded[
            labels == class_index
        ].sum(axis=0, dtype=np.int64)

    return prototypes


def predict_batched(encoded, prototypes, batch_size=512):
    normalized_prototypes = prototypes / np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True),
        1e-8,
    )

    predictions = np.empty(len(encoded), dtype=np.int64)

    for start in range(0, len(encoded), batch_size):
        stop = min(start + batch_size, len(encoded))
        batch = encoded[start:stop].astype(np.float32)

        # Query norms are common across classes for each sample,
        # so omitting them does not change argmax.
        scores = batch @ normalized_prototypes.T
        predictions[start:stop] = scores.argmax(axis=1)

    return predictions


def refine_one_epoch(encoded, labels, prototypes, rng, lr):
    """Mistake-driven add/subtract updates with cosine classification."""
    mistakes = 0
    norms = np.maximum(
        np.linalg.norm(prototypes, axis=1), 1e-8
    )

    for index in rng.permutation(len(encoded)):
        query = encoded[index].astype(np.float32)
        label = int(labels[index])

        scores = (prototypes @ query) / norms
        prediction = int(scores.argmax())

        if prediction != label:
            mistakes += 1

            prototypes[label] += lr * query
            prototypes[prediction] -= lr * query

            norms[label] = max(
                float(np.linalg.norm(prototypes[label])), 1e-8
            )
            norms[prediction] = max(
                float(np.linalg.norm(prototypes[prediction])), 1e-8
            )

    return mistakes


def evaluate(encoded, labels, prototypes):
    predictions = predict_batched(encoded, prototypes)
    return float(accuracy_score(labels, predictions)), predictions


def run_one_seed(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Independent random streams prevent item-memory construction from
    # changing the training shuffle sequence.
    seed_sequence = np.random.SeedSequence(args.seed)
    position_seed, value_seed, training_seed = seed_sequence.spawn(3)

    position_rng = np.random.default_rng(position_seed)
    value_rng = np.random.default_rng(value_seed)
    training_rng = np.random.default_rng(training_seed)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train, y_train, X_val, y_val = split_train_val(
        X_train,
        y_train,
        args.val_split,
        args.seed,
    )

    position_hvs = make_item_memory(
        28 * 28,
        args.dim,
        position_rng,
        args.encoding,
        args.percent_change,
    )
    value_hvs = make_item_memory(
        256,
        args.dim,
        value_rng,
        args.encoding,
        args.percent_change,
    )

    kernel = make_gabor_kernel(args)

    train_encoded = encode_dataset(
        X_train, position_hvs, value_hvs,
        kernel, args.dim, "Training",
    )
    val_encoded = encode_dataset(
        X_val, position_hvs, value_hvs,
        kernel, args.dim, "Validation",
    )

    prototypes = initialize_prototypes(
        train_encoded, y_train, num_classes=10
    )
    prototypes_before = prototypes.copy()

    initial_train_acc, _ = evaluate(
        train_encoded, y_train, prototypes
    )
    initial_val_acc, _ = evaluate(
        val_encoded, y_val, prototypes
    )

    history = [{
        "epoch": 0,
        "mistakes": None,
        "train_accuracy": initial_train_acc,
        "val_accuracy": initial_val_acc,
    }]

    best_prototypes = prototypes.copy()
    best_val_accuracy = initial_val_acc
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        mistakes = refine_one_epoch(
            train_encoded,
            y_train,
            prototypes,
            training_rng,
            args.lr,
        )

        train_acc, _ = evaluate(
            train_encoded, y_train, prototypes
        )
        val_acc, _ = evaluate(
            val_encoded, y_val, prototypes
        )

        history.append({
            "epoch": epoch,
            "mistakes": mistakes,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        })

        print(
            f"Epoch {epoch}: mistakes={mistakes}/{len(y_train)}, "
            f"train={train_acc:.4f}, validation={val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            best_prototypes = prototypes.copy()

    corruption_cfg = CorruptionConfig(
        kind=args.corrupt,
        angle_deg=args.angle,
        sigma=args.sigma,
        p=args.p,
        cutout_size=args.cutout,
        zoom=args.zoom,
    )

    # Seed immediately before corruption so item-memory generation and
    # training do not consume its NumPy/Python random stream.
    np.random.seed(args.seed)
    random.seed(args.seed)

    X_corrupted = apply_corruption(
        X=X_test.copy(),
        cfg=corruption_cfg,
    )
    X_corrupted = np.asarray(X_corrupted)

    if X_corrupted.shape != X_test.shape:
        raise ValueError(
            "Corruption must preserve the test dataset shape."
        )

    clean_encoded = encode_dataset(
        X_test, position_hvs, value_hvs,
        kernel, args.dim, "Clean test",
    )

    if args.corrupt == "none":
        corrupted_encoded = clean_encoded
    else:
        corrupted_encoded = encode_dataset(
            X_corrupted, position_hvs, value_hvs,
            kernel, args.dim, "Corrupted test",
        )

    clean_before, clean_pred_before = evaluate(
        clean_encoded, y_test, prototypes_before
    )
    clean_after, clean_pred_after = evaluate(
        clean_encoded, y_test, best_prototypes
    )

    acc_before, pred_before = evaluate(
        corrupted_encoded, y_test, prototypes_before
    )
    acc_after, pred_after = evaluate(
        corrupted_encoded, y_test, best_prototypes
    )

    metrics = {
        "model": "pixel_value_hdc",
        "dataset": "mnist",
        "seed": args.seed,
        "dim": args.dim,
        "encoding": args.encoding,
        "gabor_enabled": not args.no_gabor,
        "corruption": asdict(corruption_cfg),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "clean_acc_before": clean_before,
        "clean_acc_after": clean_after,
        "acc_before": acc_before,
        "acc_after": acc_after,
        "clean_to_corrupted_drop_pp": (
            100.0 * (clean_after - acc_after)
        ),
        "num_train": len(y_train),
        "num_val": len(y_val),
        "num_test": len(y_test),
    }

    with (args.out_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2)

    settings = vars(args).copy()
    settings["out_dir"] = str(settings["out_dir"])

    with (args.out_dir / "config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(settings, handle, indent=2)

    with (args.out_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(history[0]),
        )
        writer.writeheader()
        writer.writerows(history)

    np.savez_compressed(
        args.out_dir / "model.npz",
        position_hvs=position_hvs,
        value_hvs=value_hvs,
        prototypes_before=prototypes_before,
        prototypes_best=best_prototypes,
    )

    np.savez_compressed(
        args.out_dir / "predictions.npz",
        labels=y_test,
        clean_before=clean_pred_before,
        clean_after=clean_pred_after,
        corrupted_before=pred_before,
        corrupted_after=pred_after,
    )

    for name, predictions in (
        ("clean", clean_pred_after),
        ("corrupted", pred_after),
    ):
        matrix = confusion_matrix(
            y_test, predictions, labels=np.arange(10)
        )
        np.savetxt(
            args.out_dir / f"confusion_{name}.csv",
            matrix,
            delimiter=",",
            fmt="%d",
        )

    print("\nFinished experiment.")
    print(metrics)

    return metrics
    
def save_suite_results(root, records):
    """Save completed runs after each experiment."""
    with (root / "results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(records, handle, indent=2)

    csv_records = []

    for record in records:
        row = record.copy()
        row["corruption"] = json.dumps(
            row["corruption"],
            sort_keys=True,
        )
        csv_records.append(row)

    if csv_records:
        with (root / "results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(csv_records[0]),
            )
            writer.writeheader()
            writer.writerows(csv_records)

def summarize_suite(records):
    """Aggregate results across seeds separately for each corruption."""
    metric_names = (
        "clean_acc_before",
        "clean_acc_after",
        "acc_before",
        "acc_after",
        "best_val_accuracy",
        "clean_to_corrupted_drop_pp",
    )

    summary = {}

    corruption_names = list(dict.fromkeys(
        record["corruption"]["kind"]
        for record in records
    ))

    for corruption in corruption_names:
        selected = [
            record
            for record in records
            if record["corruption"]["kind"] == corruption
        ]

        corruption_summary = {
            "seeds": [record["seed"] for record in selected],
            "num_seeds": len(selected),
            "metrics": {},
        }

        for name in metric_names:
            values = np.asarray(
                [record[name] for record in selected],
                dtype=np.float64,
            )

            corruption_summary["metrics"][name] = {
                "mean": float(values.mean()),
                "std": (
                    float(values.std(ddof=1))
                    if len(values) > 1
                    else None
                ),
                "min": float(values.min()),
                "max": float(values.max()),
            }

        summary[corruption] = corruption_summary

    return summary


def main():
    args = parse_args()
    validate_args(args)

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates.")

    if any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be nonnegative.")

    if len(set(args.corruptions)) != len(args.corruptions):
        raise ValueError("--corruptions must not contain duplicates.")

    root = args.out_dir
    root.mkdir(parents=True, exist_ok=True)

    records = []
    total = len(args.seeds) * len(args.corruptions)
    counter = 0

    # Complete all corruptions within a seed before changing seeds.
    for seed in args.seeds:
        for corruption in args.corruptions:
            counter += 1

            print(
                f"\n{'=' * 60}\n"
                f"Run {counter}/{total}: "
                f"seed={seed}, corruption={corruption}\n"
                f"{'=' * 60}"
            )

            run_args = argparse.Namespace(**vars(args))
            run_args.seed = seed
            run_args.corrupt = corruption
            run_args.out_dir = (
                root / f"seed_{seed}" / corruption
            )

            metrics = run_one_seed(run_args)
            records.append(metrics)

            save_suite_results(root, records)

            summary = summarize_suite(records)
            with (root / "summary.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(summary, handle, indent=2)

    # Save a compact percentage-based summary for papers/tables.
    summary_rows = []

    for corruption, entry in summary.items():
        before = entry["metrics"]["acc_before"]
        after = entry["metrics"]["acc_after"]

        summary_rows.append({
            "corruption": corruption,
            "num_seeds": entry["num_seeds"],
            "accuracy_before_mean_percent": 100.0 * before["mean"],
            "accuracy_before_std_pp": (
                100.0 * before["std"]
                if before["std"] is not None
                else None
            ),
            "accuracy_after_mean_percent": 100.0 * after["mean"],
            "accuracy_after_std_pp": (
                100.0 * after["std"]
                if after["std"] is not None
                else None
            ),
        })

    with (root / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary_rows[0]),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nCompleted all seeds and corruptions.")

    for row in summary_rows:
        mean = row["accuracy_after_mean_percent"]
        std = row["accuracy_after_std_pp"]

        if std is None:
            print(f"{row['corruption']}: {mean:.2f}%")
        else:
            print(
                f"{row['corruption']}: "
                f"{mean:.2f}% ± {std:.2f} percentage points"
            )

    print(f"\nDetailed results: {root / 'results.csv'}")
    print(f"Aggregated results: {root / 'summary.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
