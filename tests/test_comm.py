"""
Unit tests for VirtualCommBus collectives: AllReduce, AllGather, and ReduceScatter.
"""

import pytest
import torch
from parallel.comm import VirtualCommBus
from parallel.context import ParallelContext, run_parallel_simulation


def test_comm_invalid_world_size():
    """Verify bus and context reject invalid world_size or rank."""
    with pytest.raises(ValueError, match="world_size must be positive"):
        VirtualCommBus(world_size=0)

    with pytest.raises(ValueError, match="Invalid rank"):
        ParallelContext(rank=4, world_size=4)


def test_all_reduce_sum():
    """Verify AllReduce correctly computes element-wise sum across all ranks."""
    world_size = 4
    shape = (2, 3)

    def _worker(ctx: ParallelContext):
        # Rank r sends tensor filled with r + 1: [1, 2, 3, 4] -> sum = 10
        t = torch.full(shape, float(ctx.rank + 1))
        reduced = ctx.all_reduce(t, op="sum")
        return reduced

    results = run_parallel_simulation(world_size, _worker)
    assert len(results) == world_size
    expected = torch.full(shape, 10.0)
    for r in range(world_size):
        assert torch.allclose(results[r], expected)


def test_all_reduce_mean():
    """Verify AllReduce correctly computes element-wise mean across all ranks."""
    world_size = 4
    shape = (4, 4)

    def _worker(ctx: ParallelContext):
        t = torch.full(shape, float(ctx.rank * 2))  # [0, 2, 4, 6] -> mean = 3.0
        return ctx.all_reduce(t, op="mean")

    results = run_parallel_simulation(world_size, _worker)
    expected = torch.full(shape, 3.0)
    for r in range(world_size):
        assert torch.allclose(results[r], expected)


def test_all_reduce_max():
    """Verify AllReduce correctly computes element-wise maximum across all ranks."""
    world_size = 3

    def _worker(ctx: ParallelContext):
        t = torch.tensor([ctx.rank, ctx.rank * 10, -ctx.rank], dtype=torch.float32)
        return ctx.all_reduce(t, op="max")

    results = run_parallel_simulation(world_size, _worker)
    expected = torch.tensor([2.0, 20.0, 0.0])
    for r in range(world_size):
        assert torch.allclose(results[r], expected)


def test_all_gather():
    """Verify AllGather concatenates rank tensors along specified dimension."""
    world_size = 3

    def _worker(ctx: ParallelContext):
        # Each rank has tensor of shape [2, 1]
        t = torch.full((2, 1), float(ctx.rank))
        # Gather along dim 1 -> [2, 3]
        return ctx.all_gather(t, dim=1)

    results = run_parallel_simulation(world_size, _worker)
    expected = torch.tensor([
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
    ])
    for r in range(world_size):
        assert torch.allclose(results[r], expected)


def test_reduce_scatter():
    """Verify ReduceScatter sums across ranks and scatters chunks."""
    world_size = 2

    def _worker(ctx: ParallelContext):
        # Rank 0: [1, 2], Rank 1: [3, 4] -> sum = [4, 6]
        # Scattered: Rank 0 gets [4], Rank 1 gets [6]
        t = torch.tensor([1.0, 2.0]) if ctx.rank == 0 else torch.tensor([3.0, 4.0])
        return ctx.reduce_scatter(t, dim=0, op="sum")

    results = run_parallel_simulation(world_size, _worker)
    assert torch.allclose(results[0], torch.tensor([4.0]))
    assert torch.allclose(results[1], torch.tensor([6.0]))


def test_comm_single_rank_pass_through():
    """Verify single rank world_size=1 passes tensors directly without communication barriers."""
    bus = VirtualCommBus(world_size=1)
    t = torch.randn(3, 3)

    reduced = bus.all_reduce(t, rank=0, op_id="test_red", op="sum")
    assert torch.equal(reduced, t)

    gathered = bus.all_gather(t, rank=0, op_id="test_gath", dim=-1)
    assert torch.equal(gathered, t)


def test_parallel_context_op_counter():
    """Verify ParallelContext produces unique, incrementing collective op IDs."""
    ctx = ParallelContext(rank=0, world_size=2)
    id1 = ctx.next_op_id("all_reduce")
    id2 = ctx.next_op_id("all_reduce")
    id3 = ctx.next_op_id("all_gather")

    assert id1 == "all_reduce_1"
    assert id2 == "all_reduce_2"
    assert id3 == "all_gather_3"
    assert len({id1, id2, id3}) == 3
