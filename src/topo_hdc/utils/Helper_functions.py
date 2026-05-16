import torch


@torch.no_grad()
def rand_bipolar(D, seed=0, device="cuda"):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    v = torch.randint(0, 2, (D,), generator=g, device=device, dtype=torch.int8)
    return v * 2 - 1  # {-1,+1}

@torch.no_grad()
def bind(hv, role):
    # hv: (N,D) or (D,), role: (D,)
    return (hv.to(torch.int16) * role.to(torch.int16)).to(torch.int8)

@torch.no_grad()
def bundle_sum_sign(hvs):
    # hvs: (N,D) int8
    s = hvs.to(torch.int32).sum(dim=0)
    return torch.where(s >= 0, 1, -1).to(torch.int8)

@torch.no_grad()
def bundle2(a, b):
    # a,b: (N,D) int8
    s = a.to(torch.int32) + b.to(torch.int32)
    return torch.where(s >= 0, 1, -1).to(torch.int8)