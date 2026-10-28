"""
Unit tests for ColumnParallelLinear layer.
"""

import pytest
import torch
import torch.nn as nn
from parallel.context import ParallelContext, run_parallel_simulation
from layers.column_parallel import ColumnParallelLinear


def test_column_parallel_indivisible_features():
    """Verify ColumnParallelLinear rejects out_features indivisible by world_size."""
    ctx = ParallelContext(rank=0, world_size=3)
    with pytest.raises(ValueError, match="divisible by world_size"):
        ColumnParallelLinear(in_features=16, out_features=10, ctx=ctx)


def test_column_parallel_forward_shapes():
    """Verify local output shape without gather is [..., out_features / world_size]."""
    in_features = 32
    out_features = 64
    world_size = 4
    x = torch.randn(2, 8, in_features)

    def _worker(ctx: ParallelContext):
        layer = ColumnParallelLinear(in_features, out_features, ctx, bias=True, gather_output=False)
        assert layer.local_out_features == 16
        out = layer(x)
        assert list(out.shape) == [2, 8, 16]
        return out

    results = run_parallel_simulation(world_size, _worker)
    assert len(results) == world_size


def test_column_parallel_gather_output_parity():
    """Verify gather_output=True produces output strictly matching monolithic layer."""
    in_features = 16
    out_features = 32
    world_size = 2
    x = torch.randn(2, 4, in_features)

    mono = nn.Linear(in_features, out_features, bias=True)
    mono_weight_t = mono.weight.t()  # [in_features, out_features]

    def _worker(ctx: ParallelContext):
        layer = ColumnParallelLinear(in_features, out_features, ctx, bias=True, gather_output=True)
        layer.load_unsharded_weights(mono_weight_t, mono.bias)
        with torch.no_grad():
            out = layer(x)
        return out

    results = run_parallel_simulation(world_size, _worker)
    with torch.no_grad():
        expected = mono(x)

    for r in range(world_size):
        assert torch.allclose(results[r], expected, atol=1e-6)


def test_column_parallel_no_bias():
    """Verify ColumnParallelLinear functions correctly with bias=False."""
    ctx = ParallelContext(rank=0, world_size=2)
    layer = ColumnParallelLinear(in_features=16, out_features=32, ctx=ctx, bias=False)
    assert layer.bias is None
    x = torch.randn(2, 4, 16)
    out = layer(x)
    assert list(out.shape) == [2, 4, 16]


def test_column_parallel_single_rank():
    """Verify ColumnParallelLinear on world_size=1 functions as standard linear layer."""
    ctx = ParallelContext(rank=0, world_size=1)
    layer = ColumnParallelLinear(in_features=16, out_features=32, ctx=ctx, bias=True)
    assert layer.local_out_features == 32
    x = torch.randn(2, 4, 16)
    out = layer(x)
    assert list(out.shape) == [2, 4, 32]
