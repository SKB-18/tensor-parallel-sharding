"""
Tensor-Parallel Multi-Layer Perceptron (MLP).
Partitions intermediate features across P ranks (ColumnParallel FC1 -> Local Act -> RowParallel FC2).
Supports standard 2-layer MLP (GeLU) and 3-layer SwiGLU (SiLU).
"""

from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from parallel.context import ParallelContext
from layers.column_parallel import ColumnParallelLinear
from layers.row_parallel import RowParallelLinear


class ParallelMLP(nn.Module):
    """
    Megatron-LM style Tensor-Parallel Feed-Forward Network.
    
    Standard 2-layer mode:
    1. FC1: ColumnParallelLinear [hidden_size -> intermediate_size / P]. No communication.
    2. Activation (GELU/SiLU): Local elementwise operation. No communication.
    3. FC2: RowParallelLinear [intermediate_size / P -> hidden_size]. Single AllReduce(SUM).
    
    SwiGLU 3-layer mode (LLaMA style):
    1. Gate & Up: ColumnParallelLinear [hidden_size -> intermediate_size / P]. No communication.
    2. Act: Local silu(gate) * up. No communication.
    3. Down: RowParallelLinear [intermediate_size / P -> hidden_size]. Single AllReduce(SUM).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        ctx: ParallelContext,
        activation: Literal["gelu", "silu", "swiglu"] = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.ctx = ctx
        self.activation_type = activation

        if intermediate_size % ctx.world_size != 0:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be divisible by world_size ({ctx.world_size})"
            )

        self.local_intermediate_size = intermediate_size // ctx.world_size

        if activation == "swiglu":
            self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, ctx, bias=bias, gather_output=False)
            self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, ctx, bias=bias, gather_output=False)
            self.down_proj = RowParallelLinear(intermediate_size, hidden_size, ctx, bias=bias, input_is_parallel=True)
        else:
            self.fc1 = ColumnParallelLinear(hidden_size, intermediate_size, ctx, bias=bias, gather_output=False)
            self.fc2 = RowParallelLinear(intermediate_size, hidden_size, ctx, bias=bias, input_is_parallel=True)

    def load_unsharded_weights(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        b1: Optional[torch.Tensor] = None,
        b2: Optional[torch.Tensor] = None,
        w_up: Optional[torch.Tensor] = None,
        b_up: Optional[torch.Tensor] = None,
    ):
        """Loads rank's weight partitions from monolithic weight tensors."""
        if self.activation_type == "swiglu":
            self.gate_proj.load_unsharded_weights(w1, b1)
            if w_up is not None:
                self.up_proj.load_unsharded_weights(w_up, b_up)
            self.down_proj.load_unsharded_weights(w2, b2)
        else:
            self.fc1.load_unsharded_weights(w1, b1)
            self.fc2.load_unsharded_weights(w2, b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        x: [batch_size, seq_len, hidden_size]
        Returns: [batch_size, seq_len, hidden_size]
        """
        if self.activation_type == "swiglu":
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            hidden = gate * up  # [B, S, intermediate_size / P]
            return self.down_proj(hidden)
        else:
            hidden = self.fc1(x)  # [B, S, intermediate_size / P]
            if self.activation_type == "gelu":
                hidden = F.gelu(hidden)
            elif self.activation_type == "silu":
                hidden = F.silu(hidden)
            return self.fc2(hidden)  # [B, S, hidden_size] (AllReduce inside)
