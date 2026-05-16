import os
import torch

def set_seeds(seed: int, deterministic_torch:bool = True) -> None:
    """
    Sets seeds for all random number generators to ensure reproducibility
    """
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic_torch:
        # Set CuDNN to be deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        
    print(f"Seeds set to {seed}. deterministic_torch={deterministic_torch}")
        
    
