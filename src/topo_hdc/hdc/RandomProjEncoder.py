import torch

class RandomProjEncoderTorch:
    def __init__(self, D=10000, seed=0, device="cuda"):
        self.D = D
        self.seed = seed
        self.device = torch.device(device)
        self.W = None
        self.mu = None
        self.sigma = None

    def fit(self, F_np):
        # F_np: (N,M) numpy float64/float32
        F = torch.tensor(F_np, dtype=torch.float32, device=self.device)
        self.mu = F.mean(dim=0)
        self.sigma = F.std(dim=0)
        self.sigma = torch.clamp(self.sigma, min=1e-8)

        M = F.shape[1]
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed)

        # bipolar projection matrix (D,M) in int8; matmul uses float32
        self.W = torch.randint(0, 2, (self.D, M), generator=g, device=self.device, dtype=torch.int8)
        self.W = self.W * 2 - 1  # {0,1} -> {-1,+1}
        return self

    @torch.no_grad()
    def encode_batch(self, F_np, batch_size=4096):
        # returns (N,D) int8 bipolar HVs on GPU
        F = torch.tensor(F_np, dtype=torch.float32, device=self.device)
        N = F.shape[0]
        out = torch.empty((N, self.D), dtype=torch.int8, device=self.device)

        for i in range(0, N, batch_size):
            x = F[i:i+batch_size]
            z = (x - self.mu) / self.sigma
            z = torch.clamp(z, -3.0, 3.0)

            # L2 normalize per sample
            # z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-12)

            y = (z @ self.W.T.to(torch.float32))  # (B,D)
            y = y - y.mean(dim=1, keepdim=True)   # center per sample
            out[i:i+batch_size] = torch.where(y >= 0, 1, -1).to(torch.int8)

        return out
