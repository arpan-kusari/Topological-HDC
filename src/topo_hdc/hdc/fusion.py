import torch
from topo_hdc.config import RunConfig
import numpy as np
from collections.abc import Callable, Mapping
from topo_hdc.hdc.topology_compatibility import TopologyCompatibilityModel, topology_compatibility_scores
 
@torch.no_grad()
def cosine_scores(Hb_int8, protos_i32):
    Hf = Hb_int8.to(torch.float32)
    Pf = protos_i32.to(torch.float32)
    Hn = torch.norm(Hf, dim=1, keepdim=True) + 1e-8
    Pn = torch.norm(Pf, dim=1, keepdim=True).T + 1e-8
    return (Hf @ Pf.T) / (Hn * Pn)  # (B,C)
    
@torch.no_grad()
def make_early_fusion_generator(generators: dict, channels: list[str]):
    """
    Wraps individual channel batch generators, forwards configuration parameters 
    like shuffle, and concatenates their hypervector outputs along the feature dimension.
    """
    def combined_generator(*args, **kwargs):
        # Instantiate the underlying generator iterators, passing shuffle/etc. down
        gen_iters = {ch: generators[ch](*args, **kwargs) for ch in channels}
        
        while True:
            try:
                hx_batches = {ch: next(gen_iters[ch]) for ch in channels}
                # Unpack the batch signature (idxb, Hb) from each channel
                # All channels share identical sample indices (idxb), so grab from the first one
                idxb = hx_batches[channels[0]][0]
                
                # Extract the hypervector tensors (index 1) and concatenate them
                Hb_concat = torch.cat([hx_batches[ch][1] for ch in channels], dim=-1)
                
                yield idxb, Hb_concat
            except StopIteration:
                break
    return combined_generator
    
@torch.no_grad()
def predict_late_fusion_batched(
    generators, 
    protos, 
    alpha:float = 0.0, 
    beta:float = 0.0, 
    N:int | None=None,
) -> np.ndarray:
    """
    Hhog_batches: yields (key, Hh) int8
    Hfm_batches:   yields (key, Hfm) int8  (same keys, same order)
    alpha=0 disables FMT.
    Returns numpy preds length N (if N provided and keys are indices),
    else concatenates in yielded order.
    """
    if N is None:
        raise ValueError(
            "N must be provided when using indexed batch generators"
        )
        
    preds = []
    out = None
    
    Hhog_batches = generators["hog"](shuffle=False)
    Hfm_batches = generators["fourier_mellin"](shuffle=False)
    Hpi_batches = generators["persistent_image"](shuffle=False)
    
    protos_h = protos["hog"]
    protos_fm = protos["fourier_mellin"]
    protos_pi = protos["persistent_image"]


    for (key_h, Hh), (key_fm, Hfm), (key_pi, Hpi) in zip(
        Hhog_batches,
        Hfm_batches,
        Hpi_batches,
    ):
        # sanity: same batch alignment
        if isinstance(key_h, (int, np.integer)) and isinstance(key_fm, (int, np.integer)) and isinstance(key_pi, (int, np.integer)):
            assert int(key_h) == int(key_fm) == int(key_pi)
        # compute scores
        s = cosine_scores(Hh, protos_h)
        if alpha != 0.0:
            s_fm = cosine_scores(Hfm, protos_fm)
            s = s + float(alpha) * s_fm
        if beta != 0.0:
            s_pi = cosine_scores(Hpi, protos_pi)
            s = s + float(beta) * s_pi
        pb = torch.argmax(s, dim=1).cpu().numpy()

        if N is None:
            preds.append(pb)
        else:
            if out is None:
                out = np.empty((N,), dtype=np.int64)
            if isinstance(key_h, (int, np.integer)):
                start = int(key_h)
                out[start:start+len(pb)] = pb
            else:
                idxb = key_h.cpu().numpy() if torch.is_tensor(key_h) else np.asarray(key_h)
                out[idxb] = pb

    return np.concatenate(preds) if N is None else out
    
@torch.no_grad()
def predict_single_channel_batched(
    generator_fn: Callable,
    prototypes: torch.Tensor,
    num_samples: int,
) -> np.ndarray:
    """
    Predict classes using one HDC feature channel.

    Parameters
    ----------
    generator_fn:
        Callable batch generator, such as generators["holes"].
        Calling generator_fn(shuffle=False, seed=0) must yield
        `(indices, hypervectors)`.

    prototypes:
        Class prototypes with shape (num_classes, D).

    num_samples:
        Total number of samples in the dataset split.

    Returns
    -------
    np.ndarray
        Predicted labels with shape (num_samples,).
    """
    predictions = np.empty(num_samples, dtype=np.int64)
    visited = np.zeros(num_samples, dtype=bool)

    for indices, hypervectors in generator_fn(
        shuffle=False,
    ):
        scores = cosine_scores(
            Hb_int8=hypervectors,
            protos_i32=prototypes,
        )

        batch_predictions = (
            scores.argmax(dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        if torch.is_tensor(indices):
            indices_np = indices.detach().cpu().numpy()
        else:
            indices_np = np.asarray(indices)

        indices_np = indices_np.astype(np.int64, copy=False)

        if len(indices_np) != len(batch_predictions):
            raise ValueError(
                "Batch index and prediction lengths differ: "
                f"{len(indices_np)} versus {len(batch_predictions)}."
            )

        predictions[indices_np] = batch_predictions
        visited[indices_np] = True

    if not visited.all():
        missing = np.flatnonzero(~visited)
        raise RuntimeError(
            f"The generator did not produce predictions for "
            f"{len(missing)} samples. First missing indices: "
            f"{missing[:10].tolist()}."
        )

    return predictions
    
def pick_alpha_beta_on_val(
    generators,
    protos,
    y_val,
    run_cfg,
    tolerance=1e-12,
):
    results = []
    best_acc = -np.inf

    for alpha in run_cfg.alphas:
        for beta in run_cfg.betas:
            pred = predict_late_fusion_batched(
                generators=generators,
                protos=protos,
                alpha=alpha,
                beta=beta,
                N=len(y_val),
            )
            acc = float(np.mean(pred == y_val))
            results.append((acc, float(alpha), float(beta)))
            best_acc = max(best_acc, acc)

    maximizers = [
        row for row in results
        if abs(row[0] - best_acc) <= tolerance
    ]

    print("Best accuracy:", best_acc)
    print("All maximizing weights:", maximizers)

    # Conservative tie-break: smallest total auxiliary weight.
    selected = min(maximizers, key=lambda row: row[1] + row[2])
    _, alpha, beta = selected
    return alpha, beta, best_acc, results
    
def _to_numpy_indices(
    indices: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Convert batch indices to a one-dimensional NumPy array."""
    if torch.is_tensor(indices):
        indices = indices.detach().cpu().numpy()

    return np.asarray(indices, dtype=np.int64).reshape(-1)
    
@torch.no_grad()
def collect_channel_scores(
    generators: Mapping[str, Callable],
    protos: Mapping[str, torch.Tensor],
    num_samples: int,
) -> dict[str, np.ndarray]:
    """
    Collect HOG, Fourier-Mellin, Persistent-Image class-score matrices.

    Required generator interface
    ----------------------------
    generator(shuffle=False) yields:
        (sample_indices, hypervectors)

    Returns
    -------
    dict[str, np.ndarray]
        {
            "hog":      array of shape (N, C),
            "fourier_mellin":  array of shape (N, C),
        }
    """
    required_channels = ("hog", "fourier_mellin", "persistent_image")

    missing_generators = set(required_channels) - set(generators)
    missing_prototypes = set(required_channels) - set(protos)

    if missing_generators:
        raise KeyError(
            f"Missing generators: {sorted(missing_generators)}"
        )

    if missing_prototypes:
        raise KeyError(
            f"Missing prototypes: {sorted(missing_prototypes)}"
        )

    num_classes = int(protos["hog"].shape[0])
    outputs: dict[str, np.ndarray] = {}

    for channel in required_channels:
        channel_prototypes = protos[channel]

        if channel_prototypes.ndim != 2:
            raise ValueError(
                f"Prototypes for {channel!r} must be two-dimensional; "
                f"received shape {tuple(channel_prototypes.shape)}."
            )

        if channel_prototypes.shape[0] != num_classes:
            raise ValueError(
                f"Channel {channel!r} has "
                f"{channel_prototypes.shape[0]} classes; "
                f"expected {num_classes}."
            )

        scores_out = np.empty(
            (num_samples, num_classes),
            dtype=np.float32,
        )

        visited = np.zeros(
            num_samples,
            dtype=bool,
        )

        for indices, hypervectors in generators[channel](
            shuffle=False
        ):
            indices_np = _to_numpy_indices(indices)

            if np.any(indices_np < 0) or np.any(
                indices_np >= num_samples
            ):
                raise IndexError(
                    f"Generator for {channel!r} returned "
                    "out-of-range sample indices."
                )

            batch_scores = cosine_scores(
                Hb_int8=hypervectors,
                protos_i32=channel_prototypes,
            ).detach().cpu().numpy()

            if len(indices_np) != len(batch_scores):
                raise ValueError(
                    f"Channel {channel!r} produced "
                    f"{len(indices_np)} indices but "
                    f"{len(batch_scores)} score rows."
                )

            scores_out[indices_np] = batch_scores
            visited[indices_np] = True

        if not visited.all():
            missing = np.flatnonzero(~visited)

            raise RuntimeError(
                f"Channel {channel!r} is missing scores for "
                f"{len(missing)} samples. First missing indices: "
                f"{missing[:10].tolist()}."
            )

        outputs[channel] = scores_out

    return outputs


def tune_cached_fusion(
    channel_scores: Mapping[str, np.ndarray],
    y_val: np.ndarray,
    run_cfg: RunConfig,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """
    Tune alpha and beta using cached validation scores.

    Fusion rule
    -----------
    fused_scores =
        HOG scores
        + alpha * Zernike scores
        + beta * topology-compatibility scores

    Here beta weights topology compatibility, not a hole-HDC channel.
    """
    y_val = np.asarray(
        y_val,
        dtype=np.int64,
    ).reshape(-1)

    hog_scores = np.asarray(
        channel_scores["hog"],
        dtype=np.float32,
    )

    fmt_scores = np.asarray(
        channel_scores["fourier_mellin"],
        dtype=np.float32,
    )

    pi_scores = np.asarray(
        channel_scores["persistent_image"],
        dtype=np.float32,
    )

    expected_shape = hog_scores.shape

    if hog_scores.ndim != 2:
        raise ValueError(
            f"HOG scores must have shape (N, C); "
            f"received {hog_scores.shape}."
        )

    for name, scores in (
        ("fourier_mellin", fmt_scores),
        ("persistent_image", pi_scores),
    ):
        if scores.shape != expected_shape:
            raise ValueError(
                f"{name} scores have shape {scores.shape}; "
                f"expected {expected_shape}."
            )

    if len(y_val) != expected_shape[0]:
        raise ValueError(
            f"Received {len(y_val)} validation labels but "
            f"{expected_shape[0]} score rows."
        )

    best: dict[str, float] | None = None
    results: list[dict[str, float]] = []

    for alpha in run_cfg.alphas:
        for beta in run_cfg.betas:
            fused_scores = (
                hog_scores
                + float(alpha) * fmt_scores
                + float(beta) * pi_scores
            )

            predictions = fused_scores.argmax(axis=1)

            accuracy = float(
                np.mean(predictions == y_val)
            )

            result = {
                "accuracy": accuracy,
                "alpha": float(alpha),
                "beta": float(beta),
            }
            results.append(result)

            if best is None or accuracy > best["accuracy"]:
                best = result

    if best is None:
        raise RuntimeError(
            "Fusion search produced no results. Check alpha/beta grids."
        )

    return best, results
    
def tune_alpha_cached(
    channel_scores: Mapping[str, np.ndarray],
    y_val: np.ndarray,
    run_cfg: RunConfig,
    beta: float | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """
    Tune only alpha while keeping the topology weight beta fixed.

    Fusion rule:
        HOG + alpha * Zernike + beta * topology compatibility.
    """
    y_val = np.asarray(y_val, dtype=np.int64).reshape(-1)

    hog_scores = np.asarray(
        channel_scores["hog"],
        dtype=np.float32,
    )
    fmt_scores = np.asarray(
        channel_scores["fourier_mellin"],
        dtype=np.float32,
    )
    pi_scores = np.asarray(
        channel_scores["persistent_image"],
        dtype=np.float32,
    )

    if fmt_scores.shape != hog_scores.shape:
        raise ValueError(
            "HOG and Fourier-Mellin score shapes do not match: "
            f"{hog_scores.shape} versus {fmt_scores.shape}."
        )

    if pi_scores.shape != hog_scores.shape:
        raise ValueError(
            "HOG and persistent-image score shapes do not match: "
            f"{hog_scores.shape} versus {pi_scores.shape}."
        )

    if len(y_val) != hog_scores.shape[0]:
        raise ValueError(
            f"Received {len(y_val)} labels but "
            f"{hog_scores.shape[0]} score rows."
        )

    if beta is None:
        beta = run_cfg.topology_beta

    beta = float(beta)
    best: dict[str, float] | None = None
    results: list[dict[str, float]] = []

    # This term does not change while alpha is searched.
    fixed_scores = (
        hog_scores
        + beta * pi_scores
    )

    for alpha in run_cfg.alphas:
        alpha = float(alpha)

        fused_scores = (
            fixed_scores
            + alpha * fmt_scores
        )

        predictions = fused_scores.argmax(axis=1)
        accuracy = float(np.mean(predictions == y_val))

        result = {
            "accuracy": accuracy,
            "alpha": alpha,
            "beta": beta,
        }
        results.append(result)

        if best is None or accuracy > best["accuracy"]:
            best = result

    if best is None:
        raise RuntimeError(
            "Alpha search produced no results. Check run_cfg.alphas."
        )

    print(
        f"Best alpha={best['alpha']:.3f}, "
        f"fixed beta={best['beta']:.3f}, "
        f"validation accuracy={best['accuracy']:.4f}"
    )

    return best, results


def predict_from_cached_scores(
    channel_scores: Mapping[str, np.ndarray],
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Predict using HOG, Fourier-Mellin, and persistent-image scores.

    Parameters
    ----------
    alpha:
        FMT score weight.

    beta:
        PI score weight.
    """
    hog_scores = np.asarray(
        channel_scores["hog"],
        dtype=np.float32,
    )

    fmt_scores = np.asarray(
        channel_scores["fourier_mellin"],
        dtype=np.float32,
    )

    pi_scores = np.asarray(
        channel_scores["persistent_image"],
        dtype=np.float32,
    )

    if fmt_scores.shape != hog_scores.shape:
        raise ValueError(
            "HOG and fmt score shapes do not match: "
            f"{hog_scores.shape} versus {fmt_scores.shape}."
        )

    if pi_scores.shape != hog_scores.shape:
        raise ValueError(
            "HOG and pi score shapes do not match: "
            f"{hog_scores.shape} versus {pi_scores.shape}."
        )

    fused_scores = (
        hog_scores
        + float(alpha) * fmt_scores
        + float(beta) * pi_scores
    )

    return fused_scores.argmax(axis=1).astype(np.int64)
