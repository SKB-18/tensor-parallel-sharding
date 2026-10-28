"""
Model Checkpoint Sharder: Slices monolithic state_dict into rank-specific partitions.
Supports automated round-trip reconstitution and pattern-based weight slicing.
"""

from typing import Dict, List, Optional, Tuple, Set
import os
import re
import torch


class ModelSharder:
    """
    Slices a monolithic model state_dict into P rank-specific state_dicts for Tensor Parallelism.
    
    Rules:
    - Column-Parallel: Sliced along dimension 1 (weight[:, start:end]), bias sliced along dim 0.
    - Row-Parallel: Sliced along dimension 0 (weight[start:end, :]), bias replicated intact.
    - Replicated: Copied identically to all ranks (e.g., normalization weights, embeddings).
    """

    DEFAULT_COLUMN_PATTERNS = {
        r".*q_proj\.weight",
        r".*k_proj\.weight",
        r".*v_proj\.weight",
        r".*gate_proj\.weight",
        r".*up_proj\.weight",
        r".*fc1\.weight",
    }

    DEFAULT_ROW_PATTERNS = {
        r".*out_proj\.weight",
        r".*down_proj\.weight",
        r".*fc2\.weight",
    }

    def __init__(
        self,
        world_size: int,
        column_patterns: Optional[Set[str]] = None,
        row_patterns: Optional[Set[str]] = None,
    ):
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        self.world_size = world_size
        self.column_patterns = column_patterns or self.DEFAULT_COLUMN_PATTERNS
        self.row_patterns = row_patterns or self.DEFAULT_ROW_PATTERNS

    def is_column_parallel(self, key: str) -> bool:
        """Checks if a parameter key corresponds to a Column-Parallel weight."""
        return any(re.match(pat, key) for pat in self.column_patterns)

    def is_row_parallel(self, key: str) -> bool:
        """Checks if a parameter key corresponds to a Row-Parallel weight."""
        return any(re.match(pat, key) for pat in self.row_patterns)

    def shard_state_dict(
        self,
        monolithic_dict: Dict[str, torch.Tensor],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Splits a monolithic state_dict into {rank: rank_state_dict} for all ranks in [0, world_size).
        """
        rank_dicts: Dict[int, Dict[str, torch.Tensor]] = {
            r: {} for r in range(self.world_size)
        }

        for key, tensor in monolithic_dict.items():
            if self.is_column_parallel(key):
                # Weight sliced along dim 1
                dim = 1
                if tensor.shape[dim] % self.world_size != 0:
                    raise ValueError(
                        f"Key '{key}' shape {tensor.shape} dim {dim} not divisible by world_size {self.world_size}"
                    )
                chunks = torch.chunk(tensor, chunks=self.world_size, dim=dim)
                for r in range(self.world_size):
                    rank_dicts[r][key] = chunks[r].clone()

                # Check corresponding bias
                bias_key = key.replace(".weight", ".bias")
                if bias_key in monolithic_dict:
                    bias_tensor = monolithic_dict[bias_key]
                    bias_chunks = torch.chunk(bias_tensor, chunks=self.world_size, dim=0)
                    for r in range(self.world_size):
                        rank_dicts[r][bias_key] = bias_chunks[r].clone()

            elif self.is_row_parallel(key):
                # Weight sliced along dim 0
                dim = 0
                if tensor.shape[dim] % self.world_size != 0:
                    raise ValueError(
                        f"Key '{key}' shape {tensor.shape} dim {dim} not divisible by world_size {self.world_size}"
                    )
                chunks = torch.chunk(tensor, chunks=self.world_size, dim=dim)
                for r in range(self.world_size):
                    rank_dicts[r][key] = chunks[r].clone()

                # Row parallel bias is added after reduction -> replicated on all ranks
                bias_key = key.replace(".weight", ".bias")
                if bias_key in monolithic_dict:
                    bias_tensor = monolithic_dict[bias_key]
                    for r in range(self.world_size):
                        rank_dicts[r][bias_key] = bias_tensor.clone()

            else:
                # Replicated parameter (e.g. norm weights, embeddings)
                if not key.endswith(".bias") or (
                    not self.is_column_parallel(key.replace(".bias", ".weight"))
                    and not self.is_row_parallel(key.replace(".bias", ".weight"))
                ):
                    for r in range(self.world_size):
                        rank_dicts[r][key] = tensor.clone()

        return rank_dicts

    def reconstitute_monolithic_state_dict(
        self,
        rank_dicts: Dict[int, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Inverse operation: Merges rank-specific state_dicts back into the monolithic state_dict.
        """
        if len(rank_dicts) != self.world_size:
            raise ValueError(f"Expected {self.world_size} rank dicts, got {len(rank_dicts)}")

        sample_keys = list(rank_dicts[0].keys())
        reconstituted = {}

        for key in sample_keys:
            if self.is_column_parallel(key):
                # Concat along dim 1
                tensors = [rank_dicts[r][key] for r in range(self.world_size)]
                reconstituted[key] = torch.cat(tensors, dim=1)

                bias_key = key.replace(".weight", ".bias")
                if bias_key in rank_dicts[0]:
                    biases = [rank_dicts[r][bias_key] for r in range(self.world_size)]
                    reconstituted[bias_key] = torch.cat(biases, dim=0)

            elif self.is_row_parallel(key):
                # Concat along dim 0
                tensors = [rank_dicts[r][key] for r in range(self.world_size)]
                reconstituted[key] = torch.cat(tensors, dim=0)

                bias_key = key.replace(".weight", ".bias")
                if bias_key in rank_dicts[0]:
                    reconstituted[bias_key] = rank_dicts[0][bias_key].clone()

            else:
                if not key.endswith(".bias") or (
                    not self.is_column_parallel(key.replace(".bias", ".weight"))
                    and not self.is_row_parallel(key.replace(".bias", ".weight"))
                ):
                    reconstituted[key] = rank_dicts[0][key].clone()

        return reconstituted
