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
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

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
        description="PyTorch CNN: train per seed and evaluate all corruptions."
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
        help="Output units of apply_corruption: 0..255 or 0..1.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("runs/cnn_torch_suite")
    )
    return parser.parse_args()


def validate_args(args):
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must not contain duplicates.")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be nonnegative.")
    if len(set(args.corruptions)) != len(args.corruptions):
        raise ValueError("Corruptions must not contain duplicates.")
    if args.epochs <= 0 or args.batch_size < 2:
        raise ValueError("Epochs must be positive and batch size >= 2.")
    if args.patience < 0 or args.num_workers < 0:
        raise ValueError("Patience and worker count must be nonnegative.")
    if not 0 < args.val_split < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    if not np.isfinite(args.sigma) or args.sigma < 0:
        raise ValueError("Sigma must be finite and nonnegative.")
    if not 0 <= args.p <= 1:
        raise ValueError("Salt-and-pepper probability must be in [0, 1].")
    if not 1 <= args.cutout <= 28:
        raise ValueError("Cutout size must be between 1 and 28.")
    if not np.isfinite(args.angle):
        raise ValueError("Rotation angle must be finite.")
    if not np.isfinite(args.zoom) or args.zoom <= 0:
        raise ValueError("Zoom must be finite and positive.")


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


class CharacterCNN(nn.Module):
    """
    Matches the supplied Keras layer layout.

    Outputs logits: CrossEntropyLoss applies log-softmax internally.
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),

            # For 26x26 input, asymmetric padding matches Keras "same".
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.Dropout(0.4),

            # For 6x6 input, asymmetric padding matches Keras "same".
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 10),
        )

        # Explicit initialization; cross-framework results need not match.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.kaiming_normal_(
            self.features[0].weight,
            mode="fan_in",
            nonlinearity="relu",
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_loader(
    X, y, args, device, *, scale="uint8", shuffle=False, seed=0
):
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3 or X.shape[1:] != (28, 28):
        raise ValueError(f"Expected (N, 28, 28); received {X.shape}.")
    if not np.isfinite(X).all():
        raise ValueError("Images contain nonfinite values.")

    if scale == "uint8":
        X = X / 255.0

    X = np.clip(X, 0, 1)
    inputs = torch.from_numpy(np.ascontiguousarray(X[:, None]))
    labels = torch.as_tensor(np.asarray(y), dtype=torch.long)

    if len(inputs) != len(labels):
        raise ValueError("Image and label counts differ.")

    generator = torch.Generator().manual_seed(seed)

    # BatchNorm at the final 1x1 map cannot train on a singleton batch.
    drop_last = shuffle and len(inputs) % args.batch_size == 1

    return DataLoader(
        TensorDataset(inputs, labels),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        drop_last=drop_last,
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        count += len(labels)
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()

    return total_loss / count, correct / count


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    predictions = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, labels)
        predicted = logits.argmax(1)

        count += len(labels)
        total_loss += loss.item() * len(labels)
        correct += (predicted == labels).sum().item()
        predictions.append(predicted.cpu().numpy())

    return (
        total_loss / count,
        correct / count,
        np.concatenate(predictions),
    )


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_history(history, destination):
    write_csv(destination / "history.csv", history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [row["epoch"] for row in history]

    for key, label in (
        ("train_accuracy", "Training"),
        ("val_accuracy", "Validation"),
    ):
        axes[0].plot(epochs, [r[key] for r in history], label=label)

    for key, label in (
        ("train_loss", "Training"),
        ("val_loss", "Validation"),
    ):
        axes[1].plot(epochs, [r[key] for r in history], label=label)

    axes[0].set_ylabel("Accuracy")
    axes[1].set_ylabel("Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()

    fig.tight_layout()
    fig.savefig(destination / "learning_curves.png", dpi=150)
    plt.close(fig)


def save_confusion(labels, predictions, destination, title):
    matrix = confusion_matrix(
        labels, predictions, labels=np.arange(10)
    )
    np.savetxt(
        destination / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(destination / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_suite_results(root, records):
    save_json(root / "results.json", records)

    csv_rows = [
        {
            **record,
            "corruption_config": json.dumps(
                record["corruption_config"], sort_keys=True
            ),
        }
        for record in records
    ]
    write_csv(root / "results.csv", csv_rows)

    summary = []
    for corruption in dict.fromkeys(r["corruption"] for r in records):
        selected = [
            r for r in records if r["corruption"] == corruption
        ]
        values = np.asarray([r["test_accuracy"] for r in selected])
        summary.append({
            "corruption": corruption,
            "num_seeds": len(selected),
            "mean_accuracy_percent": float(100 * values.mean()),
            "std_accuracy_pp": (
                float(100 * values.std(ddof=1))
                if len(values) > 1 else None
            ),
        })

    save_json(root / "summary.json", summary)
    write_csv(root / "summary.csv", summary)


def main():
    args = parse_args()
    validate_args(args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check PyTorch installation/GPU allocation, "
            "or run with --device cpu."
        )

    # Required by some CUDA operations in strict deterministic mode.
    if args.deterministic:
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    settings["torch_version"] = str(torch.__version__)
    settings["torch_cuda_version"] = torch.version.cuda
    save_json(args.out_dir / "config.json", settings)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    training_data = MNIST(
        root=str(args.data_dir), train=True, download=True
    )
    testing_data = MNIST(
        root=str(args.data_dir), train=False, download=True
    )

    X_full = training_data.data.numpy()
    y_full = training_data.targets.numpy()
    X_test = testing_data.data.numpy()
    y_test = testing_data.targets.numpy()

    records = []

    # Seed -> all corruptions -> next seed.
    for seed in args.seeds:
        seed_everything(seed, args.deterministic)

        seed_dir = args.out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=args.val_split,
            random_state=seed,
        )
        train_idx, val_idx = next(splitter.split(X_full, y_full))

        train_loader = make_loader(
            X_full[train_idx], y_full[train_idx],
            args, device, shuffle=True, seed=seed,
        )
        val_loader = make_loader(
            X_full[val_idx], y_full[val_idx], args, device
        )
        clean_loader = make_loader(X_test, y_test, args, device)

        model = CharacterCNN().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, eps=1e-7
        )
        criterion = nn.CrossEntropyLoss()

        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        wait = 0
        history = []

        print(f"\nTraining seed {seed}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_accuracy, _ = evaluate(
                model, val_loader, criterion, device
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            })
            print(
                f"Epoch {epoch}: "
                f"train loss={train_loss:.4f}, acc={train_accuracy:.4f}; "
                f"val loss={val_loss:.4f}, acc={val_accuracy:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                wait = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "epoch": best_epoch,
                        "val_loss": best_loss,
                        "seed": seed,
                        "config": settings,
                    },
                    seed_dir / "best_model.pt",
                )
            else:
                wait += 1
                if wait >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        save_history(history, seed_dir)

        # Evaluate training in inference mode for a clean generalization check.
        train_eval_loader = make_loader(
            X_full[train_idx], y_full[train_idx], args, device
        )
        train_loss, train_accuracy, _ = evaluate(
            model, train_eval_loader, criterion, device
        )
        val_loss, val_accuracy, _ = evaluate(
            model, val_loader, criterion, device
        )
        clean_loss, clean_accuracy, clean_predictions = evaluate(
            model, clean_loader, criterion, device
        )

        save_json(seed_dir / "training_metrics.json", {
            "seed": seed,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "clean_test_accuracy": clean_accuracy,
        })

        for corruption in args.corruptions:
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
                loss = clean_loss
                accuracy = clean_accuracy
                predictions = clean_predictions
            else:
                # Independent of randomness consumed by model training.
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                corrupted = np.asarray(apply_corruption(
                    X=X_test.copy(),
                    cfg=corruption_cfg,
                ))
                if corrupted.shape != X_test.shape:
                    raise ValueError(
                        f"{corruption}: corruption changed image shape."
                    )

                loader = make_loader(
                    corrupted,
                    y_test,
                    args,
                    device,
                    scale=args.corruption_output_scale,
                )
                loss, accuracy, predictions = evaluate(
                    model, loader, criterion, device
                )

            record = {
                "model": "pytorch_cnn",
                "dataset": "mnist",
                "seed": seed,
                "corruption": corruption,
                "corruption_config": asdict(corruption_cfg),
                "best_epoch": best_epoch,
                "epochs_completed": len(history),
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "clean_test_accuracy": clean_accuracy,
                "test_loss": loss,
                "test_accuracy": accuracy,
                "test_accuracy_percent": 100 * accuracy,
                "clean_to_corrupted_drop_pp": (
                    100 * (clean_accuracy - accuracy)
                ),
                "num_train": len(train_idx),
                "num_val": len(val_idx),
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
                f"PyTorch CNN: {corruption}, seed {seed}",
            )

            records.append(record)
            save_suite_results(args.out_dir, records)

            print(
                f"Seed {seed}, {corruption}: "
                f"{100 * accuracy:.2f}%"
            )

        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nFinished. Results: {args.out_dir / 'results.csv'}")
    print(f"Summary: {args.out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
