#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv2D,
    Dropout,
    MaxPooling2D,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from topo_hdc.config import CorruptionConfig
from topo_hdc.corruptions import apply_corruption


CORRUPTIONS = (
    "none",
    "rotation",
    "gaussian",
    "saltpepper",
    "cutout",
    "zoom",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CNN per seed and evaluate all corruptions."
    )

    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--corruptions",
        nargs="+",
        choices=CORRUPTIONS,
        default=list(CORRUPTIONS),
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-split", type=float, default=0.2)

    parser.add_argument("--angle", type=float, default=20.0)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--cutout", type=int, default=12)
    parser.add_argument("--zoom", type=float, default=0.5)

    parser.add_argument(
        "--corruption-output-scale",
        choices=("uint8", "unit"),
        default="uint8",
        help=(
            "Intensity units returned by apply_corruption: "
            "'uint8' means 0..255; 'unit' means 0..1."
        ),
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/cnn_suite"),
    )

    return parser.parse_args()


def validate_args(args):
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must not contain duplicates.")

    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be nonnegative.")

    if len(set(args.corruptions)) != len(args.corruptions):
        raise ValueError("Corruptions must not contain duplicates.")

    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("Epochs and batch size must be positive.")

    if args.patience < 0:
        raise ValueError("Patience must be nonnegative.")

    if not 0 < args.val_split < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")

    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be finite and positive.")

    if not np.isfinite(args.sigma) or args.sigma < 0:
        raise ValueError("Noise sigma must be finite and nonnegative.")

    if not 0 <= args.p <= 1:
        raise ValueError("Salt-and-pepper probability must be in [0, 1].")

    if not 1 <= args.cutout <= 28:
        raise ValueError("Cutout size must be between 1 and 28.")

    if not np.isfinite(args.angle):
        raise ValueError("Rotation angle must be finite.")

    if not np.isfinite(args.zoom) or args.zoom <= 0:
        raise ValueError("Zoom must be finite and positive.")


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


def cnn_inputs(X, scale):
    """Convert grayscale images to float32 (N, 28, 28, 1)."""
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3 or X.shape[1:] != (28, 28):
        raise ValueError(
            f"Expected image shape (N, 28, 28); got {X.shape}."
        )

    if not np.isfinite(X).all():
        raise ValueError("Images contain nonfinite values.")

    if scale == "uint8":
        X = X / 255.0

    return np.clip(X, 0.0, 1.0)[..., np.newaxis]


def build_model(lr):
    """Preserve the architecture from the supplied CNN."""
    model = Sequential([
        Input(shape=(28, 28, 1)),

        Conv2D(
            16,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        ),
        BatchNormalization(),

        Conv2D(32, kernel_size=(3, 3), activation="relu"),
        BatchNormalization(),

        Conv2D(
            64,
            kernel_size=5,
            strides=2,
            padding="same",
            activation="relu",
        ),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Dropout(0.4),

        Conv2D(
            64,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        ),
        MaxPooling2D((2, 2)),
        BatchNormalization(),

        Flatten(),
        Dense(32, activation="relu"),
        Dropout(0.4),
        Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def save_history(history, destination):
    values = history.history
    fields = ["epoch"] + list(values)

    with (destination / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for index in range(len(values["loss"])):
            writer.writerow({
                "epoch": index + 1,
                **{
                    name: float(series[index])
                    for name, series in values.items()
                },
            })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(values["accuracy"], label="Training")
    axes[0].plot(values["val_accuracy"], label="Validation")
    axes[0].set_xlabel("Epoch index")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(values["loss"], label="Training")
    axes[1].plot(values["val_loss"], label="Validation")
    axes[1].set_xlabel("Epoch index")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(destination / "learning_curves.png", dpi=150)
    plt.close(fig)


def save_confusion(y_true, predictions, destination, title):
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=np.arange(10),
    )

    np.savetxt(
        destination / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.arange(10),
        yticklabels=np.arange(10),
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(destination / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_suite_results(root, records):
    save_json(root / "results.json", records)

    if not records:
        return

    rows = []
    for record in records:
        row = record.copy()
        row["corruption_config"] = json.dumps(
            row["corruption_config"],
            sort_keys=True,
        )
        rows.append(row)

    with (root / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = []

    for corruption in dict.fromkeys(
        record["corruption"] for record in records
    ):
        selected = [
            record for record in records
            if record["corruption"] == corruption
        ]
        accuracies = np.asarray([
            record["test_accuracy"] for record in selected
        ])

        summary.append({
            "corruption": corruption,
            "num_seeds": len(selected),
            "mean_accuracy_percent": float(100 * accuracies.mean()),
            "std_accuracy_pp": (
                float(100 * accuracies.std(ddof=1))
                if len(selected) > 1
                else None
            ),
        })

    save_json(root / "summary.json", summary)

    with (root / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary[0]),
        )
        writer.writeheader()
        writer.writerows(summary)


def main():
    args = parse_args()
    validate_args(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    settings = vars(args).copy()
    settings["out_dir"] = str(settings["out_dir"])
    save_json(args.out_dir / "config.json", settings)

    if args.deterministic:
        tf.config.experimental.enable_op_determinism()

    (X_full, y_full), (X_test_raw, y_test) = mnist.load_data()

    records = []

    # Train once per seed, then finish all corruptions for that seed.
    for seed in args.seeds:
        print(f"\n{'=' * 60}\nTraining seed {seed}\n{'=' * 60}")

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)

        seed_dir = args.out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        X_train_raw, y_train, X_val_raw, y_val = split_train_val(
            X_full, y_full, args.val_split, seed
        )

        X_train = cnn_inputs(X_train_raw, "uint8")
        X_val = cnn_inputs(X_val_raw, "uint8")
        X_clean = cnn_inputs(X_test_raw, "uint8")

        model = build_model(args.lr)

        if seed == args.seeds[0]:
            model.summary()

        checkpoint = seed_dir / "best_model.weights.h5"

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(checkpoint),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose=1,
        )

        # Explicitly load this seed's best validation-loss checkpoint.
        model.load_weights(checkpoint)

        save_history(history, seed_dir)

        train_metrics = model.evaluate(
            X_train, y_train,
            batch_size=args.batch_size,
            verbose=0,
            return_dict=True,
        )
        val_metrics = model.evaluate(
            X_val, y_val,
            batch_size=args.batch_size,
            verbose=0,
            return_dict=True,
        )

        clean_probabilities = model.predict(
            X_clean,
            batch_size=args.batch_size,
            verbose=0,
        )
        clean_predictions = clean_probabilities.argmax(axis=1)
        clean_accuracy = float(
            accuracy_score(y_test, clean_predictions)
        )

        best_epoch = int(
            np.argmin(history.history["val_loss"]) + 1
        )

        save_json(
            seed_dir / "training_metrics.json",
            {
                "seed": seed,
                "epochs_completed": len(history.history["loss"]),
                "best_epoch": best_epoch,
                "train_accuracy": float(train_metrics["accuracy"]),
                "train_loss": float(train_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_loss": float(val_metrics["loss"]),
                "clean_test_accuracy": clean_accuracy,
                "num_parameters": int(model.count_params()),
            },
        )

        for corruption in args.corruptions:
            print(f"\nSeed {seed}: evaluating {corruption}")

            destination = seed_dir / corruption
            destination.mkdir(parents=True, exist_ok=True)

            corruption_cfg = CorruptionConfig(
                kind=corruption,
                angle_deg=args.angle,
                sigma=args.sigma,
                p=args.p,
                cutout_size=args.cutout,
                zoom=args.zoom,
            )

            if corruption == "none":
                predictions = clean_predictions
            else:
                # Reset corruption randomness independently of training.
                np.random.seed(seed)
                random.seed(seed)

                corrupted = np.asarray(
                    apply_corruption(
                        X=X_test_raw.copy(),
                        cfg=corruption_cfg,
                    )
                )

                if corrupted.shape != X_test_raw.shape:
                    raise ValueError(
                        f"{corruption}: corruption changed image shape "
                        f"from {X_test_raw.shape} to {corrupted.shape}."
                    )

                inputs = cnn_inputs(
                    corrupted,
                    args.corruption_output_scale,
                )

                probabilities = model.predict(
                    inputs,
                    batch_size=args.batch_size,
                    verbose=0,
                )
                predictions = probabilities.argmax(axis=1)

            test_accuracy = float(
                accuracy_score(y_test, predictions)
            )

            record = {
                "model": "cnn",
                "dataset": "mnist",
                "seed": seed,
                "corruption": corruption,
                "corruption_config": asdict(corruption_cfg),
                "best_epoch": best_epoch,
                "epochs_completed": len(history.history["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "clean_test_accuracy": clean_accuracy,
                "test_accuracy": test_accuracy,
                "test_accuracy_percent": 100 * test_accuracy,
                "clean_to_corrupted_drop_pp": (
                    100 * (clean_accuracy - test_accuracy)
                ),
                "num_train": len(y_train),
                "num_val": len(y_val),
                "num_test": len(y_test),
                "out_dir": str(destination),
            }

            save_json(destination / "metrics.json", record)

            np.savez_compressed(
                destination / "predictions.npz",
                labels=y_test,
                predictions=predictions,
            )

            save_confusion(
                y_test,
                predictions,
                destination,
                title=f"CNN: {corruption}, seed {seed}",
            )

            records.append(record)
            save_suite_results(args.out_dir, records)

            print(f"Test accuracy: {100 * test_accuracy:.2f}%")

        del model

    print(f"\nFinished. Results saved to {args.out_dir}")
    print(f"Detailed results: {args.out_dir / 'results.csv'}")
    print(f"Mean and standard deviation: {args.out_dir / 'summary.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
