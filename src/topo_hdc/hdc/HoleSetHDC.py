import torch

class HoleSetHDCTorch:
    def __init__(self, D=10000, Q=51, k=14, seed=123, device="cuda"):
        self.D = D
        self.Q = Q
        self.k = k
        self.device = torch.device(device)
        g = torch.Generator(device=self.device).manual_seed(seed)

        # item memory (k,D) and correlated levels (Q,D)
        self.I = (torch.randint(0, 2, (k, D), generator=g, device=self.device, dtype=torch.int8) * 2 - 1)
        self.L = self.make_correlated_levels(Q, D, g)

        self.mu = None
        self.sigma = None

        # explicit token for "no holes"
        self.NO_HOLE = (torch.randint(0, 2, (D,), generator=g, device=self.device, dtype=torch.int8) * 2 - 1)

    def make_correlated_levels(self, Q, D, g):
        base = (torch.randint(0, 2, (D,), generator=g, device=self.device, dtype=torch.int8) * 2 - 1)
        levels = torch.empty((Q, D), device=self.device, dtype=torch.int8)
        levels[0] = base
        flip_idx = torch.randperm(D, generator=g, device=self.device)
        step = max(1, D // (Q - 1))
        cur = base.clone()
        for q in range(1, Q):
            lo = (q - 1) * step
            hi = q * step if q < Q - 1 else D
            cur[flip_idx[lo:hi]] *= -1
            levels[q] = cur
        return levels

    def fit_scaler(self, HFEATS_np):
        F = torch.tensor(HFEATS_np, dtype=torch.float32, device=self.device)
        if F.numel() == 0:
            self.mu = torch.zeros((self.k,), device=self.device)
            self.sigma = torch.ones((self.k,), device=self.device)
        else:
            self.mu = F.mean(dim=0)
            self.sigma = torch.clamp(F.std(dim=0), min=1e-8)
        return self

    @torch.no_grad()
    def quantize(self, f):
        # f: (..., k) float32
        z = (f - self.mu) / self.sigma
        z = torch.clamp(z, -3.0, 3.0)
        q = torch.floor((z + 3.0) / 6.0 * (self.Q - 1) + 1e-6).to(torch.int64)
        return q  # (..., k)

    @torch.no_grad()
    def encode_padded(self, HF_np, mask_np):
        """
        Memory-safe holes-as-set encoding (no (N,H,k,D) materialization).

        Inputs
        ------
        HF_np   : (N,Hmax,k) float32 numpy array of padded hole features
        mask_np : (N,Hmax)   float32/bool numpy array, 1 for real hole else 0

        Returns
        -------
        hv_set  : (N,D) int8 torch tensor on self.device
        """
        HF = torch.as_tensor(HF_np, dtype=torch.float32, device=self.device)  # (N,H,k)
        M = torch.as_tensor(mask_np, dtype=torch.float32, device=self.device)  # (N,H)

        N, Hmax, k = HF.shape
        if k != self.k:
            raise ValueError(f"HF last dim k={k} != self.k={self.k}")

        # Quantize: (N,H,k) int64 in [0, Q-1]
        q = self.quantize(HF)  # (N,H,k)

        # Accumulate per-hole hypervectors in int32: (N,H,D)
        # hv_hole = sign( sum_j I[j] * L[q_j] )
        hv_acc = torch.zeros((N, Hmax, self.D), dtype=torch.int32, device=self.device)

        # Loop over k (small, e.g., 14). This avoids huge intermediate tensors.
        for j in range(k):
            qj = q[:, :, j].reshape(-1)  # (N*H,)
            Lj = self.L.index_select(0, qj).view(N, Hmax, self.D)  # (N,H,D) int8
            Ij = self.I[j].view(1, 1, self.D)  # (1,1,D) int8

            hv_acc += (Lj.to(torch.int16) * Ij.to(torch.int16)).to(torch.int32)

        hv_hole = torch.where(hv_acc >= 0, 1, -1).to(torch.int8)  # (N,H,D)

        # Mask padded holes and bundle (sum then sign) across H
        hv_hole_i32 = hv_hole.to(torch.int32) * M[:, :, None].to(torch.int32)  # (N,H,D)
        acc = hv_hole_i32.sum(dim=1)  # (N,D) int32
        hv_set = torch.where(acc >= 0, 1, -1).to(torch.int8)  # (N,D)

        # If you want to match your original code more closely, you may want to
        # NOT inject a NO_HOLE token here; instead handle "no holes" upstream by
        # skipping the holes term in bundling. If you do want the token, keep this:
        noholes = (M.sum(dim=1) == 0)
        # hv_set[noholes] = self.NO_HOLE
        return hv_set, noholes
