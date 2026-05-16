import torch
from topo_hdc.config import RunConfig
import numpy as np
 
@torch.no_grad()
def cosine_scores(Hb_int8, protos_i32):
    Hf = Hb_int8.to(torch.float32)
    Pf = protos_i32.to(torch.float32)
    Hn = torch.norm(Hf, dim=1, keepdim=True) + 1e-8
    Pn = torch.norm(Pf, dim=1, keepdim=True).T + 1e-8
    return (Hf @ Pf.T) / (Hn * Pn)  # (B,C)
    
@torch.no_grad()
def predict_late_fusion_batched(generators, protos, alpha=0.0, beta=0.0, N=None):
    """
    Hhog_batches: yields (key, Hh) int8
    Hz_batches:   yields (key, Hz) int8  (same keys, same order)
    alpha=0 disables Zernike.
    Returns numpy preds length N (if N provided and keys are indices),
    else concatenates in yielded order.
    """
    preds = []
    out = None
    
    Hhog_batches = generators["hog"](shuffle=False)
    Hz_batches = generators["zernike"](shuffle=False)
    Hholes_batches = generators["holes"](shuffle=False)
    
    protos_h = protos["hog"]
    protos_z = protos["zernike"]
    protos_holes = protos["holes"]


    for (key_h, Hh), (key_z, Hz), (key_o, Hholes) in zip(
        Hhog_batches,
        Hz_batches,
        Hholes_batches,
    ):
        # sanity: same batch alignment
        if isinstance(key_h, (int, np.integer)) and isinstance(key_z, (int, np.integer)) and isinstance(key_o, (int, np.integer)):
            assert int(key_h) == int(key_z) == int(key_o)
        # compute scores
        s = cosine_scores(Hh, protos_h)
        if alpha != 0.0:
            sz = cosine_scores(Hz, protos_z)
            s = s + float(alpha) * sz
        if beta != 0.0:
            so = cosine_scores(Hholes, protos_holes)
            s = s + float(beta) * so
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
    
def pick_alpha_beta_on_val(generators, protos, y_val, run_cfg: RunConfig):
    best = (-1.0, 0.0, 0.0)
    for a in run_cfg.alphas:
        for b in run_cfg.betas:
            y_pred = predict_late_fusion_batched(generators, protos, alpha=a, beta=b, N=len(y_val))
            acc = (y_pred == y_val).mean()
            print(f"alpha={a:.2f}: beta={b:.2f} val_acc={acc:.4f}")
            if acc > best[0]:
                best = (acc, a, b)
    print("best:", best)
    return best[1], best[2]
