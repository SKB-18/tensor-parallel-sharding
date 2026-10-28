"""
Stress, Scaling, and Numerical Stability Tests for Tensor Parallelism.
"""

import pytest
import torch
from parallel.context import ParallelContext, run_parallel_simulation
from models.transformer import ParallelTransformer
from checkpoint.sharder import ModelSharder


def test_tp_scaling_world_sizes():
    """Verify transformer forward execution completes across world sizes P=1, 2, 4, 8."""
    hidden_size = 128
    num_heads = 8
    intermediate_size = 256
    num_layers = 2
    b, s = 2, 8
    torch.manual_seed(42)
    x = torch.randn(b, s, hidden_size)

    # Master model weights
    master_ctx = ParallelContext(rank=0, world_size=1)
    master_model = ParallelTransformer(num_layers, hidden_size, num_heads, intermediate_size, master_ctx)
    master_dict = master_model.state_dict()

    for p in [1, 2, 4, 8]:
        sharder = ModelSharder(world_size=p)
        rank_state_dicts = sharder.shard_state_dict(master_dict)

        def _worker(ctx: ParallelContext):
            model = ParallelTransformer(num_layers, hidden_size, num_heads, intermediate_size, ctx)
            model.load_state_dict(rank_state_dicts[ctx.rank])
            model.eval()
            with torch.no_grad():
                return model(x)

        results = run_parallel_simulation(p, _worker)
        assert len(results) == p
        # Check all ranks agree
        for r in range(1, p):
            assert torch.allclose(results[r], results[0], atol=1e-5)


def test_large_batch_and_sequence():
    """Verify model stability on large batch and sequence dimensions with no NaNs or Infs."""
    hidden_size = 128
    num_heads = 4
    intermediate_size = 256
    num_layers = 2
    b, s = 8, 64  # 512 total tokens
    world_size = 2
    x = torch.randn(b, s, hidden_size)

    def _worker(ctx: ParallelContext):
        model = ParallelTransformer(num_layers, hidden_size, num_heads, intermediate_size, ctx)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        return out

    results = run_parallel_simulation(world_size, _worker)
    assert results[0].shape == torch.Size([8, 64, 128])


def test_deterministic_reproducibility():
    """Verify bitwise identical output across independent runs with identical weights."""
    hidden_size = 64
    num_heads = 4
    intermediate_size = 128
    b, s = 2, 4
    torch.manual_seed(999)
    x = torch.randn(b, s, hidden_size)

    # Prepare deterministic master weights
    master_ctx = ParallelContext(rank=0, world_size=1)
    master_model = ParallelTransformer(1, hidden_size, num_heads, intermediate_size, master_ctx)
    sharder = ModelSharder(world_size=2)
    rank_state_dicts = sharder.shard_state_dict(master_model.state_dict())

    def _run_pass():
        def _worker(ctx: ParallelContext):
            model = ParallelTransformer(1, hidden_size, num_heads, intermediate_size, ctx)
            model.load_state_dict(rank_state_dicts[ctx.rank])
            model.eval()
            with torch.no_grad():
                return model(x)
        return run_parallel_simulation(2, _worker)[0]

    out1 = _run_pass()
    out2 = _run_pass()
    assert torch.equal(out1, out2)
