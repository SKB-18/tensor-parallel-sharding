"""
Full Tensor-Parallel Transformer Block and Multi-Layer Model.
Pairs ParallelMHA and ParallelMLP with residual connections and layer normalization.
"""

from typing import Optional, List
import torch
import torch.nn as nn
from parallel.context import ParallelContext
from layers.sequence_parallel import SequenceParallelRMSNorm
from .parallel_mha import ParallelMultiHeadAttention
from .parallel_mlp import ParallelMLP


class ParallelTransformerBlock(nn.Module):
    """
    Standard Transformer Decoder Block sharded via Tensor Parallelism.
    Total inter-rank communication: exactly 2 AllReduce(SUM) operations per layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        ctx: ParallelContext,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ctx = ctx

        self.norm1 = SequenceParallelRMSNorm(hidden_size, ctx)
        self.attn = ParallelMultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            ctx=ctx,
            dropout=dropout,
        )

        self.norm2 = SequenceParallelRMSNorm(hidden_size, ctx)
        self.mlp = ParallelMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ctx=ctx,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [batch_size, seq_len, hidden_size]
        """
        # Pre-norm Self-Attention + Residual
        normed1 = self.norm1(x)
        attn_out = self.attn(normed1, mask=mask)
        x = x + attn_out

        # Pre-norm MLP + Residual
        normed2 = self.norm2(x)
        mlp_out = self.mlp(normed2)
        x = x + mlp_out

        return x


class ParallelTransformer(nn.Module):
    """
    Complete Multi-Layer Transformer Model partitioned across P ranks.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        ctx: ParallelContext,
        vocab_size: Optional[int] = None,
        activation: str = "gelu",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.ctx = ctx

        if vocab_size is not None:
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        else:
            self.embed = None
            self.lm_head = None

        self.layers = nn.ModuleList([
            ParallelTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                ctx=ctx,
                activation=activation,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = SequenceParallelRMSNorm(hidden_size, ctx)

    def forward(
        self,
        input_ids_or_x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        If embed exists, takes integer token IDs [B, S].
        Otherwise, takes continuous embeddings [B, S, hidden_size].
        """
        if self.embed is not None and input_ids_or_x.dtype in (torch.int32, torch.int64):
            x = self.embed(input_ids_or_x)
        else:
            x = input_ids_or_x

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.final_norm(x)

        if self.lm_head is not None:
            x = self.lm_head(x)

        return x
