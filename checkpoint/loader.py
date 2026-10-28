"""
Checkpoint Loader: Utilities for saving and loading rank-partitioned PyTorch checkpoints.
"""

from typing import Dict, Optional
import os
import torch


def save_rank_checkpoints(
    rank_dicts: Dict[int, Dict[str, torch.Tensor]],
    output_dir: str,
    prefix: str = "model_tp",
) -> Dict[int, str]:
    """
    Saves partitioned state_dicts to disk as rank-indexed checkpoint files.
    Returns mapping of {rank: file_path}.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for rank, state_dict in rank_dicts.items():
        filename = f"{prefix}_rank_{rank}.pt"
        path = os.path.join(output_dir, filename)
        torch.save(state_dict, path)
        paths[rank] = path
    return paths


def load_rank_checkpoint(
    checkpoint_dir: str,
    rank: int,
    prefix: str = "model_tp",
) -> Dict[str, torch.Tensor]:
    """
    Loads a specific rank's state_dict from disk.
    """
    filename = f"{prefix}_rank_{rank}.pt"
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Rank checkpoint not found at: {path}")
    return torch.load(path, map_location="cpu")
