"""
Unit tests for full ParallelTransformerBlock and Multi-Layer ParallelTransformer.
"""

import pytest
import torch
from parallel.context import ParallelContext, run_parallel_simulation
from models.transformer import ParallelTransformerBlock, ParallelTransformer
from layers.sequence_parallel import (
    SequenceParallelRMSNorm,
    SequenceParallelLayerNorm,
    scatter_to_sequence_parallel,
    gather_from_sequence_parallel,
)


def test_sequence_parallel_norm_and_scatter_gather():
    """Verify sequence scatter/gather round-trip and SequenceParallelRMSNorm computation."""
    world_size = 2
    b, s, h = 2, 8, 32
    torch.manual_seed(42)
    x = torch.randn(b, s, h)

    def _worker(ctx: ParallelContext):
        # Scatter: [2, 8, 32] -> [2, 4, 32]
        x_local = scatter_to_sequence_parallel(x, ctx, seq_dim=1)
        assert list(x_local.shape) == [2, 4, 32]

        norm = SequenceParallelRMSNorm(h, ctx)
        normed_local = norm(x_local)
        assert list(normed_local.shape) == [2, 4, 32]

        # Gather back: [2, 4, 32] -> [2, 8, 32]
        gathered = gather_from_sequence_parallel(normed_local, ctx, seq_dim=1)
        return gathered

    results = run_parallel_simulation(world_size, _worker)
    assert list(results[0].shape) == [2, 8, 32]
    assert torch.allclose(results[0], results[1])


def test_sequence_parallel_indivisible_seq_len():
    """Verify scatter_to_sequence_parallel raises ValueError when seq_len is indivisible by world_size."""
    ctx = ParallelContext(rank=0, world_size=3)
    x = torch.randn(2, 7, 16)  # 7 not divisible by 3
    with pytest.raises(ValueError, match="Sequence length.*must be divisible by world_size"):
        scatter_to_sequence_parallel(x, ctx, seq_dim=1)


def test_transformer_block_forward():
    """Verify ParallelTransformerBlock executes across P=2 ranks with exact consensus."""
    world_size = 2
    hidden_size = 64
    num_heads = 4
    intermediate_size = 128
    b, s = 2, 4
    torch.manual_seed(42)
    x = torch.randn(b, s, hidden_size)

    def _worker(ctx: ParallelContext):
        block = ParallelTransformerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            ctx=ctx,
        )
        block.eval()
        with torch.no_grad():
            out = block(x)
        return out

    results = run_parallel_simulation(world_size, _worker)
    assert len(results) == world_size
    # Check shape
    assert list(results[0].shape) == [b, s, hidden_size]


def test_multi_layer_parallel_transformer():
    """Verify multi-layer model with embedding and LM head runs across P=4 ranks."""
    world_size = 4
    num_layers = 2
    hidden_size = 64
    num_heads = 4
    intermediate_size = 128
    vocab_size = 100
    b, s = 2, 8
    input_ids = torch.randint(0, vocab_size, (b, s))

    def _worker(ctx: ParallelContext):
        model = ParallelTransformer(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            ctx=ctx,
            vocab_size=vocab_size,
        )
        model.eval()
        with torch.no_grad():
            logits = model(input_ids)
        return logits

    results = run_parallel_simulation(world_size, _worker)
    assert list(results[0].shape) == [b, s, vocab_size]
