"""
Unit tests for RowParallelLinear layer.
"""

import pytest
import torch
import torch.nn as nn
from parallel.context import ParallelContext, run_parallel_simulation
from layers.row_parallel import RowParallelLinear


def test_row_parallel_indivisible_features():
    """Verify RowParallelLinear rejects in_features indivisible by world_size."""
    ctx = ParallelContext(rank=0, world_size=3)
    with pytest.raises(ValueError, match="divisible by world_size"):
        RowParallelLinear(in_features=10, out_features=16, ctx=ctx)


def test_row_parallel_forward_parity():
    """Verify RowParallelLinear computes exact dot product sum matching monolithic linear."""
    in_features = 32
    out_features = 16
    world_size = 4
    x = torch.randn(2, 6, in_features)

    mono = nn.Linear(in_features, out_features, bias=True)
    mono_weight_t = mono.weight.t()  # [in_features, out_features]

    def _worker(ctx: ParallelContext):
        # input_is_parallel=False means rank slices x automatically
        layer = RowParallelLinear(in_features, out_features, ctx, bias=True, input_is_parallel=False)
        layer.load_unsharded_weights(mono_weight_t, mono.bias)
        with torch.no_grad():
            out = layer(x)
        return out

    results = run_parallel_simulation(world_size, _worker)
    with torch.no_grad():
        expected = mono(x)

    for r in range(world_size):
        assert torch.allclose(results[r], expected, atol=1e-6)


def test_row_parallel_bias_not_multiplied():
    """Verify bias is added post-AllReduce so it is not multiplied by world_size."""
    in_features = 8
    out_features = 4
    world_size = 2
    # Zero input: output must equal pure bias
    x = torch.zeros(1, 2, in_features)
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0])

    def _worker(ctx: ParallelContext):
        layer = RowParallelLinear(in_features, out_features, ctx, bias=True, input_is_parallel=False)
        with torch.no_grad():
            layer.bias.copy_(bias)
            out = layer(x)
        return out

    results = run_parallel_simulation(world_size, _worker)
    expected = bias.unsqueeze(0).unsqueeze(0).expand(1, 2, 4)
    for r in range(world_size):
        assert torch.allclose(results[r], expected, atol=1e-6)


def test_row_parallel_no_bias():
    """Verify RowParallelLinear functions correctly with bias=False."""
    world_size = 2
    x = torch.randn(2, 4, 32)

    def _worker(ctx: ParallelContext):
        layer = RowParallelLinear(in_features=32, out_features=16, ctx=ctx, bias=False, input_is_parallel=False)
        assert layer.bias is None
        return layer(x)

    results = run_parallel_simulation(world_size, _worker)
    assert list(results[0].shape) == [2, 4, 16]


def test_row_parallel_single_rank():
    """Verify RowParallelLinear on world_size=1 functions as standard linear layer."""
    ctx = ParallelContext(rank=0, world_size=1)
    layer = RowParallelLinear(in_features=32, out_features=16, ctx=ctx, bias=True, input_is_parallel=False)
    assert layer.local_in_features == 32
    x = torch.randn(2, 4, 32)
    out = layer(x)
    assert list(out.shape) == [2, 4, 16]
