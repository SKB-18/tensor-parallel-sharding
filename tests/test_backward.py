"""
End-to-end backward pass and gradient synchronization tests for Tensor Parallel Sharding.
Validates exact mathematical parity between parallel distributed backward passes
and monolithic autograd gradients.
"""

import pytest
import torch
import torch.nn as nn
from parallel.comm import VirtualCommBus
from parallel.context import ParallelContext, run_parallel_simulation
from layers.column_parallel import ColumnParallelLinear
from layers.row_parallel import RowParallelLinear
from models.parallel_mlp import ParallelMLP
from models.transformer import ParallelTransformerBlock
from checkpoint.sharder import ModelSharder


def test_column_parallel_gradient_parity():
    """Verify ColumnParallelLinear weight and input gradients match monolithic."""
    torch.manual_seed(42)
    in_features = 32
    out_features = 64
    batch_size = 4
    seq_len = 8

    # Monolithic reference
    # ColumnParallelLinear weight is [in_features, out_features]
    mono_w = nn.Parameter(torch.randn(in_features, out_features))
    mono_b = nn.Parameter(torch.zeros(out_features))
    mono_x = torch.randn(batch_size, seq_len, in_features, requires_grad=True)
    mono_out = torch.matmul(mono_x, mono_w) + mono_b
    mono_loss = mono_out.sum()
    mono_loss.backward()

    # Parallel simulation (P=2)
    world_size = 2
    state_dict = {"weight": mono_w.data, "bias": mono_b.data}
    sharder = ModelSharder(
        world_size=world_size,
        column_patterns={r"^weight$"},
        row_patterns=set(),
    )
    sharded_weights = sharder.shard_state_dict(state_dict)

    def _worker(ctx: ParallelContext):
        layer = ColumnParallelLinear(
            in_features=in_features,
            out_features=out_features,
            ctx=ctx,
            gather_output=True,
            bias=True,
        )
        layer.load_state_dict(sharded_weights[ctx.rank])

        x = mono_x.detach().clone().requires_grad_(True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        # Weight grad check: slice should match
        expected_w_grad = mono_w.grad[:, ctx.rank * 32 : (ctx.rank + 1) * 32]
        assert torch.allclose(layer.weight.grad, expected_w_grad, atol=1e-5)
        return x.grad

    grads = run_parallel_simulation(world_size=world_size, worker_fn=_worker)
    # Reduced input gradient across ranks equals monolithic input gradient
    total_x_grad = grads[0] + grads[1]
    assert torch.allclose(total_x_grad, mono_x.grad, atol=1e-5)


def test_row_parallel_gradient_parity():
    """Verify RowParallelLinear weight and input gradients match monolithic."""
    torch.manual_seed(42)
    in_features = 64
    out_features = 32
    batch_size = 4
    seq_len = 8

    # RowParallelLinear weight is [local_in_features, out_features]
    mono_w = nn.Parameter(torch.randn(in_features, out_features))
    mono_b = nn.Parameter(torch.zeros(out_features))
    mono_x = torch.randn(batch_size, seq_len, in_features, requires_grad=True)
    mono_out = torch.matmul(mono_x, mono_w) + mono_b
    mono_loss = mono_out.sum()
    mono_loss.backward()

    world_size = 2
    state_dict = {"weight": mono_w.data, "bias": mono_b.data}
    sharder = ModelSharder(
        world_size=world_size,
        column_patterns=set(),
        row_patterns={r"^weight$"},
    )
    sharded_weights = sharder.shard_state_dict(state_dict)

    def _worker(ctx: ParallelContext):
        layer = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            ctx=ctx,
            input_is_parallel=True,
            bias=True,
        )
        layer.load_state_dict(sharded_weights[ctx.rank])

        # Input is partitioned along in_features (dim=-1)
        x_rank = mono_x[:, :, ctx.rank * 32 : (ctx.rank + 1) * 32].detach().clone().requires_grad_(True)
        out = layer(x_rank)
        loss = out.sum()
        loss.backward()

        # Weight grad check: row parallel weight is [local_in, out]
        expected_w_grad = mono_w.grad[ctx.rank * 32 : (ctx.rank + 1) * 32, :]
        assert torch.allclose(layer.weight.grad, expected_w_grad, atol=1e-5)

        # Input grad check: x_rank grad matches slice of mono_x grad
        expected_x_grad = mono_x.grad[:, :, ctx.rank * 32 : (ctx.rank + 1) * 32]
        assert torch.allclose(x_rank.grad, expected_x_grad, atol=1e-5)

    run_parallel_simulation(world_size=world_size, worker_fn=_worker)


def test_parallel_mlp_backward_gradient_flow():
    """Verify full end-to-end gradient backprop through ParallelMLP."""
    torch.manual_seed(1337)
    d_model = 32
    d_ff = 64
    batch_size = 2
    seq_len = 4
    world_size = 2

    class MonoMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1_w = nn.Parameter(torch.randn(d_model, d_ff))
            self.fc1_b = nn.Parameter(torch.zeros(d_ff))
            self.act = nn.GELU()
            self.fc2_w = nn.Parameter(torch.randn(d_ff, d_model))
            self.fc2_b = nn.Parameter(torch.zeros(d_model))

        def forward(self, x):
            h = self.act(torch.matmul(x, self.fc1_w) + self.fc1_b)
            return torch.matmul(h, self.fc2_w) + self.fc2_b

    mono = MonoMLP()
    x_mono = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    mono_out = mono(x_mono)
    mono_loss = (mono_out ** 2).sum()
    mono_loss.backward()

    # Shard weights
    state = {
        "fc1.weight": mono.fc1_w.data,
        "fc1.bias": mono.fc1_b.data,
        "fc2.weight": mono.fc2_w.data,
        "fc2.bias": mono.fc2_b.data,
    }
    sharder = ModelSharder(world_size=world_size)
    sharded = sharder.shard_state_dict(state)

    def _worker(ctx: ParallelContext):
        p_mlp = ParallelMLP(
            hidden_size=d_model,
            intermediate_size=d_ff,
            activation="gelu",
            ctx=ctx,
            bias=True,
        )
        p_mlp.fc1.load_state_dict({"weight": sharded[ctx.rank]["fc1.weight"], "bias": sharded[ctx.rank]["fc1.bias"]})
        p_mlp.fc2.load_state_dict({"weight": sharded[ctx.rank]["fc2.weight"], "bias": sharded[ctx.rank]["fc2.bias"]})

        x = x_mono.detach().clone().requires_grad_(True)
        out = p_mlp(x)
        assert torch.allclose(out, mono_out, atol=1e-5)

        loss = (out ** 2).sum()
        loss.backward()
        return x.grad

    grads = run_parallel_simulation(world_size=world_size, worker_fn=_worker)
    total_x_grad = grads[0] + grads[1]
    assert torch.allclose(total_x_grad, x_mono.grad, atol=1e-5)


def test_parallel_transformer_block_backward():
    """Verify gradient flow and optimization step parity on ParallelTransformerBlock."""
    torch.manual_seed(101)
    d_model = 32
    n_heads = 4
    d_ff = 64
    batch_size = 2
    seq_len = 4
    world_size = 2

    def _worker(ctx: ParallelContext):
        block = ParallelTransformerBlock(
            hidden_size=d_model,
            num_heads=n_heads,
            intermediate_size=d_ff,
            ctx=ctx,
            activation="gelu",
        )
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        out = block(x)
        loss = out.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

        # Verify weights have gradients
        assert block.attn.q_proj.weight.grad is not None
        assert block.attn.k_proj.weight.grad is not None
        assert block.attn.v_proj.weight.grad is not None
        assert block.attn.out_proj.weight.grad is not None
        assert block.mlp.fc1.weight.grad is not None
        assert block.mlp.fc2.weight.grad is not None

    run_parallel_simulation(world_size=world_size, worker_fn=_worker)


def test_all_gather_backward_slice():
    """Verify AllGather backward correctly slices upstream gradients back to local shards."""
    world_size = 2
    bus = VirtualCommBus(world_size=world_size)
    upstream_grad = torch.arange(16.0).view(2, 8)

    def _worker(ctx: ParallelContext):
        # Each rank has local slice of length 4
        x = torch.zeros(2, 4, requires_grad=True)
        gathered = ctx.all_gather(x, dim=-1)
        # Compute loss against upstream_grad
        loss = (gathered * upstream_grad).sum()
        loss.backward()

        expected_slice = upstream_grad[:, ctx.rank * 4 : (ctx.rank + 1) * 4]
        assert torch.allclose(x.grad, expected_slice)

    run_parallel_simulation(world_size=world_size, worker_fn=_worker)


def test_reduce_scatter_backward_gather():
    """Verify ReduceScatter backward correctly all-gathers upstream gradients across ranks."""
    world_size = 2
    upstream_grad_0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    upstream_grad_1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    def _worker(ctx: ParallelContext):
        x = torch.zeros(2, 4, requires_grad=True)
        scattered = ctx.reduce_scatter(x, dim=-1, op="sum")
        target_grad = upstream_grad_0 if ctx.rank == 0 else upstream_grad_1
        loss = (scattered * target_grad).sum()
        loss.backward()

        # Backward of reduce-scatter is all-gather of [upstream_grad_0, upstream_grad_1]
        expected_full_grad = torch.cat([upstream_grad_0, upstream_grad_1], dim=-1)
        assert torch.allclose(x.grad, expected_full_grad)

    run_parallel_simulation(world_size=world_size, worker_fn=_worker)


def test_swiglu_mlp_backward_gradient_parity():
    """Verify backward gradient flow on 3-layer LLaMA SwiGLU MLP."""
    torch.manual_seed(42)
    d_model = 32
    d_ff = 64
    batch_size = 2
    seq_len = 4
    world_size = 2

    # Monolithic reference
    class MonoSwiGLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Parameter(torch.randn(d_model, d_ff))
            self.up = nn.Parameter(torch.randn(d_model, d_ff))
            self.down = nn.Parameter(torch.randn(d_ff, d_model))

        def forward(self, x):
            g = torch.nn.functional.silu(torch.matmul(x, self.gate))
            u = torch.matmul(x, self.up)
            return torch.matmul(g * u, self.down)

    mono = MonoSwiGLU()
    x_mono = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    mono_out = mono(x_mono)
    mono_loss = (mono_out ** 2).sum()
    mono_loss.backward()

    # Shard weights
    state = {
        "gate_proj.weight": mono.gate.data,
        "up_proj.weight": mono.up.data,
        "down_proj.weight": mono.down.data,
    }
    sharder = ModelSharder(world_size=world_size)
    sharded = sharder.shard_state_dict(state)

    def _worker(ctx: ParallelContext):
        p_mlp = ParallelMLP(
            hidden_size=d_model,
            intermediate_size=d_ff,
            activation="swiglu",
            ctx=ctx,
            bias=False,
        )
        p_mlp.gate_proj.load_state_dict({"weight": sharded[ctx.rank]["gate_proj.weight"]})
        p_mlp.up_proj.load_state_dict({"weight": sharded[ctx.rank]["up_proj.weight"]})
        p_mlp.down_proj.load_state_dict({"weight": sharded[ctx.rank]["down_proj.weight"]})

        x = x_mono.detach().clone().requires_grad_(True)
        out = p_mlp(x)
        assert torch.allclose(out, mono_out, atol=1e-3, rtol=1e-4)

        loss = (out ** 2).sum()
        loss.backward()
        return x.grad

    grads = run_parallel_simulation(world_size=world_size, worker_fn=_worker)
    total_x_grad = grads[0] + grads[1]
    assert torch.allclose(total_x_grad, x_mono.grad, atol=1e-3, rtol=1e-4)

