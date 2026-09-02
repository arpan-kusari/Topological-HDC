import torch
from pathlib import Path
import json
import numpy as np
from dataclasses import asdict, is_dataclass
from topo_hdc.hdc.encoders import Encoders

def make_run_dir(run_dir_name, cfg) -> Path:
    path = Path(run_dir_name)
    path.mkdir(parents=True, exist_ok=True)
    return path

def make_json_serializable(obj):
    """Recursively convert non-JSON type to JSON-safe types"""
    if isinstance(obj, Path):
       return str(obj)
    if isinstance(obj, dict):
        return {k:make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj,  np.ndarray):
        return obj.tolist()
    return obj
    
def save_config(path, cfg, name) -> None:
    with open(path / f"{name}.json", "w") as f:
        if is_dataclass(cfg):
            cfg_dict = asdict(cfg)
        else:
            cfg_dict = cfg
        cfg_dict = make_json_serializable(cfg_dict)
        json.dump(cfg_dict, f, indent=2)
    
def save_json(path, obj, name) -> None:
    with open(path / f"{name}.json", "w") as f:
        json.dump(obj, f)
        
def save_model_encoder(path, encoder, name) -> None:
    torch.save(
        {
            "encoder" :encoder,
        }, path / f"{name}.pt",
    )
    
def save_model_protos(path, protos, name) -> None:
    torch.save(
        {
            "protos": protos,
        }, path / f"{name}.pt"
    )
    
def save_model_alpha_beta(path, alpha, beta, name) -> None:
    with open(path / f"{name}.txt", "w") as f:
        f.write(f"Alpha = {alpha}, Beta = , {beta}")

def save_run_outputs(
    out_dir: str,
    cfg,
    metrics: dict,
    alpha_before: float,
    beta_before: float,
    alpha_after: float,
    beta_after: float,
    encoder: Encoders,
    protos_before: torch.IntTensor,
    protos_after: torch.IntTensor,
):
    path = make_run_dir(out_dir, cfg)
    save_config(path, cfg, "Config")
    save_json(path, metrics, "Metrics")
    save_model_encoder(path, encoder, "Train_Encoder")
    save_model_protos(path, protos_before, "Protos_before")
    save_model_protos(path, protos_after, "Protos_after")
    save_model_alpha_beta(path, alpha_before, beta_before, "Alpha_beta_before")
    save_model_alpha_beta(path, alpha_after, beta_after, "Alpha_beta_after")

