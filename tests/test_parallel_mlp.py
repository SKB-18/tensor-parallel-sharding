"""
Unit tests for ParallelMLP (standard GeLU and SwiGLU).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from parallel.context import ParallelContext, run_parallel_simulation
from models.parallel_mlp import ParallelMLP


class ReferenceGELUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class ReferenceSwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def test_mlp_indivisible_intermediate():
    """Verify ParallelMLP raises ValueError when intermediate_size is not divisible by world_size."""
    ctx = ParallelContext(rank=0, world_size=3)
    with pytest.raises(ValueError, match="divisible by world_size"):
        ParallelMLP(hidden_size=64, intermediate_size=100, ctx=ctx)


def test_mlp_gelu_parity():
    """Verify exact parity between ReferenceGELUMLP and ParallelMLP on P=2 ranks."""
    hidden_size = 64
    intermediate_size = 256
    world_size = 2
    b, s = 2, 4
    torch.manual_seed(42)
    x = torch.randn(b, s, hidden_size)

    ref = ReferenceGELUMLP(hidden_size, intermediate_size)
    with torch.no_grad():
        ref_out = ref(x)

    def _worker(ctx: ParallelContext):
        tp_mlp = ParallelMLP(hidden_size, intermediate_size, ctx, activation="gelu", bias=True)
        tp_mlp.load_unsharded_weights(
            w1=ref.fc1.weight.t(),
            w2=ref.fc2.weight.t(),
            b1=ref.fc1.bias,
            b2=ref.fc2.bias,
        )
        with torch.no_grad():
            return tp_mlp(x)

    results = run_parallel_simulation(world_size, _worker)
    for r in range(world_size):
        diff = (results[r] - ref_out).abs().max().item()
        assert diff < 1e-6, f"Rank {r} diff {diff} exceeds threshold"


def test_mlp_swiglu_parity():
    """Verify exact parity for 3-layer LLaMA-style SwiGLU on P=4 ranks."""
    hidden_size = 64
    intermediate_size = 128
    world_size = 4
    b, s = 2, 4
    torch.manual_seed(99)
    x = torch.randn(b, s, hidden_size)

    ref = ReferenceSwiGLUMLP(hidden_size, intermediate_size)
    with torch.no_grad():
        ref_out = ref(x)

    def _worker(ctx: ParallelContext):
        tp_mlp = ParallelMLP(hidden_size, intermediate_size, ctx, activation="swiglu", bias=False)
        tp_mlp.load_unsharded_weights(
            w1=ref.gate_proj.weight.t(),
            w2=ref.down_proj.weight.t(),
            w_up=ref.up_proj.weight.t(),
        )
        with torch.no_grad():
            return tp_mlp(x)

    results = run_parallel_simulation(world_size, _worker)
    for r in range(world_size):
        diff = (results[r] - ref_out).abs().max().item()
        assert diff < 1e-6, f"Rank {r} SwiGLU diff {diff} exceeds threshold"


def test_mlp_load_without_bias():
    """Verify ParallelMLP can load weights when bias=False."""
    hidden_size = 64
    intermediate_size = 128
    ctx = ParallelContext(rank=0, world_size=2)
    mlp = ParallelMLP(hidden_size, intermediate_size, ctx, activation="gelu", bias=False)

    w1 = torch.randn(hidden_size, intermediate_size)
    w2 = torch.randn(intermediate_size, hidden_size)

    mlp.load_unsharded_weights(w1, w2)
    assert mlp.fc1.bias is None
    assert mlp.fc2.bias is None
