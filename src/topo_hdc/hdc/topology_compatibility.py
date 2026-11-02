from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TopologyCompatibilityModel:
    """
    Class-conditional distribution over topology categories.

    Categories:
        0: no detected holes
        1: one detected hole
        2: two or more detected holes
    """

    log_prob: np.ndarray       # shape (num_classes, num_bins)
    counts: np.ndarray         # shape (num_classes, num_bins)
    num_classes: int
    num_bins: int = 3
    smoothing: float = 1.0


def topology_bin(hole_counts: np.ndarray) -> np.ndarray:
    """
    Map raw hole counts to {0, 1, 2+}.
    """
    counts = np.asarray(hole_counts, dtype=np.int64)

    if np.any(counts < 0):
        raise ValueError("Hole counts must be nonnegative.")

    return np.clip(counts, 0, 2)


def fit_topology_compatibility(
    hole_counts: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    smoothing: float = 1.0,
) -> TopologyCompatibilityModel:
    """
    Estimate P(topology bin | class) using clean training data only.
    """
    labels = np.asarray(labels, dtype=np.int64)
    bins = topology_bin(hole_counts)

    if len(labels) != len(bins):
        raise ValueError(
            "The number of labels and hole counts must match."
        )

    num_bins = 3
    count_table = np.zeros(
        (num_classes, num_bins),
        dtype=np.float64,
    )

    for class_id in range(num_classes):
        class_bins = bins[labels == class_id]

        for topology_id in range(num_bins):
            count_table[class_id, topology_id] = np.sum(
                class_bins == topology_id
            )

    probabilities = (
        count_table + smoothing
    ) / (
        count_table.sum(axis=1, keepdims=True)
        + smoothing * num_bins
    )

    return TopologyCompatibilityModel(
        log_prob=np.log(probabilities),
        counts=count_table,
        num_classes=num_classes,
        num_bins=num_bins,
        smoothing=smoothing,
    )


def topology_compatibility_scores(
    hole_counts: np.ndarray,
    model: TopologyCompatibilityModel,
) -> np.ndarray:
    """
    Return topology compatibility scores with shape (N, C).
    """
    bins = topology_bin(hole_counts)

    # model.log_prob has shape (C, K).
    # Selecting columns by bins gives (C, N), so transpose to (N, C).
    scores = model.log_prob[:, bins].T

    return scores.astype(np.float32)
