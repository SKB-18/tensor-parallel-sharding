"""
Column-Parallel Linear Layer for Tensor Parallelism.
Splits weight matrix along columns across P ranks; optional AllGather output.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
from parallel.context import ParallelContext


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with weight matrix partitioned along output features (columns).
    
    Unsharded: Y = X * W (where W has shape [in_features, out_features])
    Sharded:   Y_i = X * W_i (where W_i has shape [in_features, out_features / P])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ctx: ParallelContext,
        bias: bool = True,
        gather_output: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ctx = ctx
        self.gather_output = gather_output

        if out_features % ctx.world_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by world_size ({ctx.world_size})"
            )

        self.local_out_features = out_features // ctx.world_size

        # Weight is [in_features, local_out_features]
        self.weight = nn.Parameter(torch.empty(in_features, self.local_out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.local_out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights using standard Kaiming uniform and bias to zeros."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def load_unsharded_weights(self, full_weight: torch.Tensor, full_bias: Optional[torch.Tensor] = None):
        """
        Loads the rank's slice from a monolithic [in_features, out_features] weight.
        """
        # Slice columns: rank * local_out_features : (rank + 1) * local_out_features
        start = self.ctx.rank * self.local_out_features
        end = start + self.local_out_features

        with torch.no_grad():
            self.weight.copy_(full_weight[:, start:end])
            if full_bias is not None and self.bias is not None:
                self.bias.copy_(full_bias[start:end])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        X: [..., in_features]
        Local Y_i: [..., local_out_features]
        If gather_output: AllGather along last dim -> [..., out_features]
        """
        output = torch.matmul(x, self.weight)
        if self.bias is not None:
            output = output + self.bias

        if self.gather_output:
            return self.ctx.all_gather(output, dim=-1)
        return output
