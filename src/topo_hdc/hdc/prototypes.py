import torch
from collections.abc import Callable
import numpy as np
from topo_hdc.config import RunConfig, DatasetConfig
 
@torch.no_grad()
def train_prototypes_stream_batches(H_batches_fn: Callable, 
                                    y_np: np.ndarray, 
                                    run_cfg: RunConfig, 
                                    data_cfg: DatasetConfig) -> torch.IntTensor:
    """
    H_batches: iterable yielding (idxb, Hb) where:
        - idxb is indices for this batch (1D numpy array or 1D torch tensor), length B
        - Hb is (B,D) int8 on GPU
    y_np: numpy labels length N
    """
    y = torch.as_tensor(y_np, dtype=torch.int64, device=run_cfg.device)
    protos_int = None

    for idxb, Hb in H_batches_fn(shuffle=False):
        Hb = Hb.to(run_cfg.device)
        if protos_int is None:
            protos_int = torch.zeros((data_cfg.num_classes, Hb.shape[1]), dtype=torch.int32, device=run_cfg.device)

        # idxb -> torch on GPU for indexing labels
        if isinstance(idxb, np.ndarray):
            idxb_t = torch.as_tensor(idxb, dtype=torch.int64, device=run_cfg.device)
        elif torch.is_tensor(idxb):
            idxb_t = idxb.to(device=run_cfg.device, dtype=torch.int64)
        else:
            idxb_t = torch.as_tensor(list(idxb), dtype=torch.int64, device=run_cfg.device)

        yb = y.index_select(0, idxb_t)          # (B,)
        Hb_i32 = Hb.to(torch.int32)             # (B,D)
        
        # Vectorized prototype accumulation:
        # protos_int[c] += sum of hypervectors whose label is c
        protos_int.index_add_(0, yb, Hb_i32)

        # for c in range(data_cfg.n_classes):
        #     m = (yb == c)
        #     if m.any():
        #         protos_int[c] += Hb_i32[m].sum(dim=0)
    return protos_int  # int32
    
def train_prototypes_stream_all(generators, y_np, run_cfg: RunConfig, data_cfg: DatasetConfig) -> dict[str, torch.Tensor]:
    protos = {}
    for name, H_batches_fn in generators.items():
        protos[name] = train_prototypes_stream_batches(H_batches_fn=H_batches_fn,
        y_np=y_np, 
        run_cfg=run_cfg,
        data_cfg=data_cfg)
    return protos
    
@torch.no_grad()
def onlinehd_train_stream(
    H_batches_fn: Callable,
    y_np: np.ndarray,
    run_cfg,
    data_cfg,
    init_protos: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    OnlineHD training for one HV channel.

    If init_protos is None, prototypes are initialized by one streaming pass.
    """
    device = torch.device(run_cfg.device)
    y = torch.as_tensor(y_np, dtype=torch.int64, device=device)

    # Initialize prototypes
    if init_protos is None:
        protos = train_prototypes_stream(
            H_batches_fn=H_batches_fn,
            y_np=y_np,
            run_cfg=run_cfg,
            data_cfg=data_cfg,
        )
    else:
        protos = init_protos.to(device)

    # If lr is non-integer, use float prototypes.
    # If lr = 1, int32 prototypes are fine.
    lr = run_cfg.lr
    if not float(lr).is_integer():
        protos = protos.to(torch.float32)
        update_dtype = torch.float32
    else:
        update_dtype = torch.int32
        lr = int(lr)

    for ep in range(run_cfg.epochs):
        mistakes = 0

        for idxb, Hb in H_batches_fn(shuffle=True):
            Hb = Hb.to(device)

            idxb_t = torch.as_tensor(idxb, dtype=torch.int64, device=device)
            yb = y.index_select(0, idxb_t)

            Hf = Hb.to(torch.float32)
            Pf = protos.to(torch.float32)

            scores = (Hf @ Pf.T) / (
                (Hf.norm(dim=1, keepdim=True) + 1e-8)
                * (Pf.norm(dim=1, keepdim=True).T + 1e-8)
            )

            pred = scores.argmax(dim=1)
            wrong = pred != yb

            if not wrong.any():
                continue

            mistakes += int(wrong.sum().item())

            Hwrong = Hb[wrong].to(update_dtype)
            ytrue = yb[wrong]
            ypred = pred[wrong]

            # Vectorized OnlineHD updates:
            # P[y_true] += lr * H
            # P[y_pred] -= lr * H
            protos.index_add_(0, ytrue, lr * Hwrong)
            protos.index_add_(0, ypred, -lr * Hwrong)

        print(f"epoch {ep + 1}: mistakes {mistakes}/{len(y_np)}")

    return protos
    
@torch.no_grad()
def onlinehd_train_stream_all(
    generators: dict[str, Callable],
    y_np: np.ndarray,
    run_cfg,
    data_cfg,
    init_protos: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """
    OnlineHD training for all channels.

    Returns:
        {
            "hog": protos_h,
            "zernike": protos_z,
            "holes": protos_holes,
        }
    """
    if init_protos is None:
        init_protos = train_prototypes_stream_all(
            generators=generators,
            y_np=y_np,
            run_cfg=run_cfg,
            data_cfg=data_cfg,
        )

    updated_protos = {}

    for name, H_batches_fn in generators.items():
        print(f"\nOnlineHD training channel: {name}")

        updated_protos[name] = onlinehd_train_stream(
            H_batches_fn=H_batches_fn,
            y_np=y_np,
            run_cfg=run_cfg,
            data_cfg=data_cfg,
            init_protos=init_protos[name],
        )
    return updated_protos

@torch.no_grad()
def predict_cosine_batched(H_batches, protos, device="cuda") -> np.ndarray:
    """
    Returns numpy predictions length N.
    """
    preds = []
    Pf = protos.to(torch.float32)
    Pn = torch.norm(Pf, dim=1, keepdim=True).T + 1e-8

    for _, Hb in H_batches:
        Hf = Hb.to(torch.float32)
        Hn = torch.norm(Hf, dim=1, keepdim=True) + 1e-8
        scores = (Hf @ Pf.T) / (Hn * Pn)
        preds.append(torch.argmax(scores, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)

