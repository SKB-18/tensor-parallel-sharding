"""
Sequence-Parallel Normalization and Scatter/Gather Primitives.
Partitions activation memory across the sequence length dimension (S / P).
"""

from typing import Optional
import torch
import torch.nn as nn
from parallel.context import ParallelContext


class SequenceParallelRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization executed over sequence-parallel shards.
    Operates on [batch, seq_len / P, hidden_size] local slices.
    """

    def __init__(self, hidden_size: int, ctx: ParallelContext, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.ctx = ctx
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len_local, hidden_size]
        Normalizes along hidden_size dimension independently per token.
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * self.weight


class SequenceParallelLayerNorm(nn.Module):
    """
    Standard Layer Normalization executed over sequence-parallel shards.
    """

    def __init__(self, hidden_size: int, ctx: ParallelContext, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.ctx = ctx
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        normed = (x - mean) * torch.rsqrt(variance + self.eps)
        return normed * self.weight + self.bias


def scatter_to_sequence_parallel(x: torch.Tensor, ctx: ParallelContext, seq_dim: int = 1) -> torch.Tensor:
    """Slices a full sequence [batch, S, hidden] into rank-specific shard [batch, S/P, hidden]."""
    if ctx.world_size == 1:
        return x
    seq_len = x.shape[seq_dim]
    if seq_len % ctx.world_size != 0:
        raise ValueError(f"Sequence length ({seq_len}) must be divisible by world_size ({ctx.world_size})")
    chunks = torch.chunk(x, chunks=ctx.world_size, dim=seq_dim)
    return chunks[ctx.rank].clone()


def gather_from_sequence_parallel(x: torch.Tensor, ctx: ParallelContext, seq_dim: int = 1) -> torch.Tensor:
    """Gathers sequence shards from all ranks to reconstitute full sequence [batch, S, hidden]."""
    if ctx.world_size == 1:
        return x
    return ctx.all_gather(x, dim=seq_dim)
