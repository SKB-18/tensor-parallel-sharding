"""
Edge cases, failure handling, and communication invariants for Tensor Parallel Sharding.
Validates robust execution under boundary conditions, non-standard shapes,
desynchronization guards, and deep multi-layer stack scaling.
"""

import pytest
import torch
import torch.nn as nn
from parallel.comm import VirtualCommBus
from parallel.context import ParallelContext, run_parallel_simulation
from layers.column_parallel import ColumnParallelLinear
from layers.row_parallel import RowParallelLinear
from layers.sequence_parallel import (
    SequenceParallelRMSNorm,
    SequenceParallelLayerNorm,
    scatter_to_sequence_parallel,
    gather_from_sequence_parallel,
)
from models.transformer import ParallelTransformer, ParallelTransformerBlock
from checkpoint.sharder import ModelSharder


def test_single_token_generation_step():
    """Verify inference single-token step (B=1, T=1) across P=2."""
    world_size = 2
    d_model = 32
    n_heads = 4
    d_ff = 64

    def _worker(ctx: ParallelContext):
        block = ParallelTransformerBlock(
            hidden_size=d_model,
            num_heads=n_heads,
            intermediate_size=d_ff,
            ctx=ctx,
        )
        x = torch.randn(1, 1, d_model)
        out = block(x)
        assert out.shape == (1, 1, d_model)
        assert not torch.isnan(out).any()

    run_parallel_simulation(world_size=world_size, worker_fn=_worker)


def test_irregular_prime_dimensions():
    """Verify irregular dimensions (B=7, T=11, d_model=64, n_heads=4, d_ff=128)."""
    world_size = 4
    d_model = 64
    n_heads = 4
    d_ff = 128
    batch_size = 7
    seq_len = 11

    def _worker(ctx: ParallelContext):
        block = ParallelTransformerBlock(
            hidden_size=d_model,
            num_heads=n_heads,
            intermediate_size=d_ff,
            ctx=ctx,
        )
        x = torch.randn(batch_size, seq_len, d_model)
        out = block(x)
        assert out.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(out).any()

    run_parallel_simulation(world_size=world_size, worker_fn=_worker)


def test_deep_stack_exact_allreduce_invariant():
    """Verify an 8-layer parallel transformer executes exactly 16 AllReduces per forward."""
    world_size = 2
    d_model = 32
    n_heads = 4
    d_ff = 64
    n_layers = 8
    vocab_size = 100

    def _worker(ctx: ParallelContext):
        model = ParallelTransformer(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_heads=n_heads,
            intermediate_size=d_ff,
            num_layers=n_layers,
            ctx=ctx,
        )
        x = torch.randint(0, vocab_size, (2, 8))
        logits = model(x)
        assert logits.shape == (2, 8, vocab_size)

        # 8 layers * 2 AllReduces per layer = 16 AllReduces total
        assert ctx.op_counter == 16

    run_parallel_simulation(world_size=world_size, worker_fn=_worker)


def test_comm_shape_mismatch_detection():
    """Verify that attempting to all-reduce tensors of mismatched shapes raises RuntimeError."""
    world_size = 2

    def _worker(ctx: ParallelContext):
        # Rank 0 passes shape (4, 4), Rank 1 passes shape (4, 8)
        shape = (4, 4) if ctx.rank == 0 else (4, 8)
        tensor = torch.zeros(shape)
        ctx.all_reduce(tensor, op="sum")

    with pytest.raises(RuntimeError, match=r"stack expects each tensor to be equal size"):
        run_parallel_simulation(world_size=world_size, worker_fn=_worker)


def test_sequence_parallel_norm_gradient_backprop():
    """Verify gradient flow through SequenceParallelRMSNorm."""
    world_size = 2
    d_model = 32
    batch_size = 2

    def _worker(ctx: ParallelContext):
        norm = SequenceParallelRMSNorm(hidden_size=d_model, ctx=ctx)
        
        # Local sequence chunk is 4
        x_local = torch.randn(batch_size, 4, d_model, requires_grad=True)
        out = norm(x_local)
        loss = out.sum()
        loss.backward()

        assert x_local.grad is not None
        assert x_local.grad.shape == (batch_size, 4, d_model)
        assert not torch.isnan(x_local.grad).any()
        assert norm.weight.grad is not None

    run_parallel_simulation(world_size=world_size, worker_fn=_worker)


def test_sharder_non_tensor_metadata_preservation():
    """Verify ModelSharder handles dictionaries containing scalar or buffer metadata."""
    world_size = 2
    sharder = ModelSharder(world_size=world_size)
    # in_features=32, out_features=64 -> column parallel splits dim 1: shape (32, 32)
    state = {
        "transformer.layers.0.attn.qkv_proj.weight": torch.randn(32, 64),
        "transformer.layers.0.attn.qkv_proj.bias": torch.zeros(64),
        "meta_step": torch.tensor(42),
    }

    shards = sharder.shard_state_dict(state)
    assert len(shards) == 2
    for r in range(2):
        assert "meta_step" in shards[r]
        assert shards[r]["meta_step"].item() == 42
        assert shards[r]["transformer.layers.0.attn.qkv_proj.weight"].shape == (32, 32)
        assert shards[r]["transformer.layers.0.attn.qkv_proj.bias"].shape == (32,)
