"""
Row-Parallel Linear Layer for Tensor Parallelism.
Splits weight matrix along rows across P ranks; performs AllReduce(SUM) aggregation.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
from parallel.context import ParallelContext


class RowParallelLinear(nn.Module):
    """
    Linear layer with weight matrix partitioned along input features (rows).
    
    Unsharded: Y = X * W + b (where W has shape [in_features, out_features])
    Sharded:   Z_i = X_i * W_i (where W_i has shape [in_features / P, out_features])
               Y = AllReduce(SUM_i(Z_i)) + b
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ctx: ParallelContext,
        bias: bool = True,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ctx = ctx
        self.input_is_parallel = input_is_parallel

        if in_features % ctx.world_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by world_size ({ctx.world_size})"
            )

        self.local_in_features = in_features // ctx.world_size

        # Weight is [local_in_features, out_features]
        self.weight = nn.Parameter(torch.empty(self.local_in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
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
        # Slice rows: rank * local_in_features : (rank + 1) * local_in_features
        start = self.ctx.rank * self.local_in_features
        end = start + self.local_in_features

        with torch.no_grad():
            self.weight.copy_(full_weight[start:end, :])
            if full_bias is not None and self.bias is not None:
                self.bias.copy_(full_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        If input_is_parallel is True, x is already partitioned: [..., local_in_features]
        Otherwise, x is sliced along the last dimension.
        Local dot product Z_i = x_i * W_i -> [..., out_features]
        AllReduce sum aggregation -> [..., out_features]
        Bias added post-reduction.
        """
        if not self.input_is_parallel:
            start = self.ctx.rank * self.local_in_features
            end = start + self.local_in_features
            x = x[..., start:end]

        local_output = torch.matmul(x, self.weight)

        # Communicate: AllReduce sum across all ranks
        reduced_output = self.ctx.all_reduce(local_output, op="sum")

        if self.bias is not None:
            reduced_output = reduced_output + self.bias

        return reduced_output
