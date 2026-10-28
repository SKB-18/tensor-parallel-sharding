"""
Tensor-Parallel Multi-Head Attention (MHA).
Partitions attention heads across P ranks (ColumnParallel QKV -> Local Attention -> RowParallel OutProj).
"""

from typing import Optional
import math
import torch
import torch.nn as nn
from parallel.context import ParallelContext
from layers.column_parallel import ColumnParallelLinear
from layers.row_parallel import RowParallelLinear


class ParallelMultiHeadAttention(nn.Module):
    """
    Megatron-LM style Tensor-Parallel Multi-Head Attention.
    
    Architecture:
    1. Q, K, V Projections: ColumnParallelLinear (partitioned across heads). No communication.
    2. Attention Core: Local scaled dot-product attention per rank on local_num_heads. No communication.
    3. Output Projection: RowParallelLinear with AllReduce(SUM) aggregation. Exactly 1 communication.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ctx: ParallelContext,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.ctx = ctx
        self.dropout_p = dropout

        if num_heads % ctx.world_size != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by world_size ({ctx.world_size})"
            )

        self.local_num_heads = num_heads // ctx.world_size
        self.head_dim = hidden_size // num_heads

        # Q, K, V are column-parallel: each rank produces [..., local_num_heads * head_dim]
        self.q_proj = ColumnParallelLinear(hidden_size, hidden_size, ctx, bias=bias, gather_output=False)
        self.k_proj = ColumnParallelLinear(hidden_size, hidden_size, ctx, bias=bias, gather_output=False)
        self.v_proj = ColumnParallelLinear(hidden_size, hidden_size, ctx, bias=bias, gather_output=False)

        # Output projection is row-parallel: consumes column-sharded heads, aggregates via AllReduce
        self.out_proj = RowParallelLinear(hidden_size, hidden_size, ctx, bias=bias, input_is_parallel=True)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def load_unsharded_weights(
        self,
        q_w: torch.Tensor,
        k_w: torch.Tensor,
        v_w: torch.Tensor,
        out_w: torch.Tensor,
        q_b: Optional[torch.Tensor] = None,
        k_b: Optional[torch.Tensor] = None,
        v_b: Optional[torch.Tensor] = None,
        out_b: Optional[torch.Tensor] = None,
    ):
        """Loads rank's weight partitions from monolithic weight tensors."""
        self.q_proj.load_unsharded_weights(q_w, q_b)
        self.k_proj.load_unsharded_weights(k_w, k_b)
        self.v_proj.load_unsharded_weights(v_w, v_b)
        self.out_proj.load_unsharded_weights(out_w, out_b)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass:
        x: [batch_size, seq_len, hidden_size]
        mask: optional [batch_size, 1, seq_len, seq_len] attention mask
        """
        batch_size, seq_len, _ = x.shape

        # 1. Local Column-Parallel Projections (no communication)
        q = self.q_proj(x)  # [B, S, local_num_heads * head_dim]
        k = self.k_proj(x)  # [B, S, local_num_heads * head_dim]
        v = self.v_proj(x)  # [B, S, local_num_heads * head_dim]

        # 2. Reshape into [B, local_num_heads, S, head_dim]
        q = q.view(batch_size, seq_len, self.local_num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.local_num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.local_num_heads, self.head_dim).transpose(1, 2)

        # 3. Local Scaled Dot-Product Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, local_heads, S, S]

        if mask is not None:
            scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)  # [B, local_heads, S, head_dim]

        # 4. Concatenate local heads: [B, S, local_num_heads * head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.local_num_heads * self.head_dim)

        # 5. Row-Parallel Out Projection + Single AllReduce(SUM) communication
        output = self.out_proj(attn_out)  # [B, S, hidden_size]

        return output
