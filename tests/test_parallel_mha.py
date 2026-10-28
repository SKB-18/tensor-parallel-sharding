"""
Unit tests for ParallelMultiHeadAttention (MHA).
"""

import pytest
import torch
import torch.nn as nn
from parallel.context import ParallelContext, run_parallel_simulation
from models.parallel_mha import ParallelMultiHeadAttention


class ReferenceMHA(nn.Module):
    """Standard un-sharded PyTorch Multi-Head Attention for parity testing."""

    def __init__(self, hidden_size: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(b, s, self.hidden_size)
        return self.out_proj(out)


def test_mha_indivisible_heads():
    """Verify ParallelMultiHeadAttention raises ValueError if heads not divisible by world_size."""
    ctx = ParallelContext(rank=0, world_size=3)
    with pytest.raises(ValueError, match="divisible by world_size"):
        ParallelMultiHeadAttention(hidden_size=64, num_heads=8, ctx=ctx)


def test_mha_numerical_parity_p2():
    """Verify exact parity between ReferenceMHA and ParallelMHA on P=2 ranks."""
    hidden_size = 128
    num_heads = 4
    world_size = 2
    b, s = 2, 8
    torch.manual_seed(42)
    x = torch.randn(b, s, hidden_size)

    ref = ReferenceMHA(hidden_size, num_heads, bias=True)
    with torch.no_grad():
        ref_out = ref(x)

    def _worker(ctx: ParallelContext):
        tp_mha = ParallelMultiHeadAttention(hidden_size, num_heads, ctx, bias=True)
        tp_mha.load_unsharded_weights(
            q_w=ref.q_proj.weight.t(),
            k_w=ref.k_proj.weight.t(),
            v_w=ref.v_proj.weight.t(),
            out_w=ref.out_proj.weight.t(),
            q_b=ref.q_proj.bias,
            k_b=ref.k_proj.bias,
            v_b=ref.v_proj.bias,
            out_b=ref.out_proj.bias,
        )
        with torch.no_grad():
            return tp_mha(x)

    tp_results = run_parallel_simulation(world_size, _worker)
    for r in range(world_size):
        diff = (tp_results[r] - ref_out).abs().max().item()
        assert diff < 1e-6, f"Rank {r} diff {diff} exceeds threshold"


def test_mha_numerical_parity_with_causal_mask():
    """Verify attention parity holds under causal autoregressive mask on P=4 ranks."""
    hidden_size = 128
    num_heads = 8
    world_size = 4
    b, s = 2, 6
    torch.manual_seed(123)
    x = torch.randn(b, s, hidden_size)

    # Causal lower-triangular mask
    mask = torch.triu(torch.full((s, s), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)

    ref = ReferenceMHA(hidden_size, num_heads, bias=True)
    with torch.no_grad():
        ref_out = ref(x, mask=mask)

    def _worker(ctx: ParallelContext):
        tp_mha = ParallelMultiHeadAttention(hidden_size, num_heads, ctx, bias=True)
        tp_mha.load_unsharded_weights(
            q_w=ref.q_proj.weight.t(),
            k_w=ref.k_proj.weight.t(),
            v_w=ref.v_proj.weight.t(),
            out_w=ref.out_proj.weight.t(),
            q_b=ref.q_proj.bias,
            k_b=ref.k_proj.bias,
            v_b=ref.v_proj.bias,
            out_b=ref.out_proj.bias,
        )
        with torch.no_grad():
            return tp_mha(x, mask=mask)

    tp_results = run_parallel_simulation(world_size, _worker)
    for r in range(world_size):
        diff = (tp_results[r] - ref_out).abs().max().item()
        assert diff < 1e-6, f"Rank {r} causal mask diff {diff} exceeds threshold"


def test_mha_load_without_bias():
    """Verify ParallelMultiHeadAttention can load weights when bias=False."""
    hidden_size = 64
    num_heads = 4
    world_size = 2
    ctx = ParallelContext(rank=0, world_size=world_size)
    mha = ParallelMultiHeadAttention(hidden_size, num_heads, ctx, bias=False)

    q_w = torch.randn(hidden_size, hidden_size)
    k_w = torch.randn(hidden_size, hidden_size)
    v_w = torch.randn(hidden_size, hidden_size)
    out_w = torch.randn(hidden_size, hidden_size)

    mha.load_unsharded_weights(q_w, k_w, v_w, out_w)
    assert mha.q_proj.bias is None
    assert mha.out_proj.bias is None
