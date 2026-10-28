"""
Communication Primitives and Virtual Bus for Tensor Parallelism.
Implements AllReduce, AllGather, and ReduceScatter over virtual in-memory ranks
with full PyTorch autograd gradient propagation support.
"""

from typing import List, Dict, Optional, Tuple
import threading
import torch


class _AllReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, bus: "VirtualCommBus", rank: int, op_id: str, op: str) -> torch.Tensor:
        ctx.op = op
        ctx.world_size = bus.world_size
        return bus._sync_all_reduce(tensor, rank, op_id, op)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.op == "sum":
            grad_input = grad_output.clone()
        elif ctx.op == "mean":
            grad_input = grad_output / ctx.world_size
        else:
            grad_input = grad_output.clone()
        return grad_input, None, None, None, None


class _AllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, bus: "VirtualCommBus", rank: int, op_id: str, dim: int) -> torch.Tensor:
        ctx.dim = dim
        ctx.rank = rank
        ctx.world_size = bus.world_size
        ctx.local_dim_size = tensor.shape[dim]
        return bus._sync_all_gather(tensor, rank, op_id, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dim = ctx.dim
        start = ctx.rank * ctx.local_dim_size
        end = start + ctx.local_dim_size
        slc = [slice(None)] * grad_output.ndim
        slc[dim] = slice(start, end)
        grad_input = grad_output[tuple(slc)].contiguous()
        return grad_input, None, None, None, None


class _ReduceScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, bus: "VirtualCommBus", rank: int, op_id: str, dim: int, op: str) -> torch.Tensor:
        ctx.dim = dim
        ctx.rank = rank
        ctx.op_id = op_id
        ctx.op = op
        ctx.bus = bus
        return bus._sync_reduce_scatter(tensor, rank, op_id, dim, op)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        bwd_op_id = f"{ctx.op_id}_bwd"
        all_gathered = ctx.bus.all_gather(grad_output, rank=ctx.rank, op_id=bwd_op_id, dim=ctx.dim)
        if ctx.op == "mean":
            all_gathered = all_gathered / ctx.bus.world_size
        return all_gathered, None, None, None, None, None


class VirtualCommBus:
    """
    In-memory virtual communication bus simulating distributed tensor collectives
    (AllReduce, AllGather, ReduceScatter) with exact synchronization across virtual ranks.
    Supports autograd backward differentiation through collective calls.
    """

    def __init__(self, world_size: int):
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        self.world_size = world_size
        self._buffers: Dict[str, Dict[int, torch.Tensor]] = {}
        self._barriers: Dict[str, threading.Barrier] = {}
        self._lock = threading.Lock()

    def _get_barrier(self, op_id: str) -> threading.Barrier:
        with self._lock:
            if op_id not in self._barriers:
                self._barriers[op_id] = threading.Barrier(self.world_size)
                self._buffers[op_id] = {}
            return self._barriers[op_id]

    def all_reduce(self, tensor: torch.Tensor, rank: int, op_id: str, op: str = "sum") -> torch.Tensor:
        """
        Executes an AllReduce collective across all ranks.
        Autograd-transparent: backpropagates gradients through reduction.
        """
        if self.world_size == 1:
            return tensor.clone() if not tensor.requires_grad else tensor
        if tensor.requires_grad:
            return _AllReduceFunction.apply(tensor, self, rank, op_id, op)
        return self._sync_all_reduce(tensor, rank, op_id, op)

    def _sync_all_reduce(self, tensor: torch.Tensor, rank: int, op_id: str, op: str = "sum") -> torch.Tensor:
        barrier = self._get_barrier(op_id)

        # Store rank's contribution
        with self._lock:
            self._buffers[op_id][rank] = tensor.detach().clone()

        # Wait for all ranks to deposit their tensor
        barrier.wait()

        # Aggregate across ranks
        with self._lock:
            collected = [self._buffers[op_id][r] for r in range(self.world_size)]
            if op == "sum":
                result = torch.stack(collected, dim=0).sum(dim=0)
            elif op == "mean":
                result = torch.stack(collected, dim=0).mean(dim=0)
            elif op == "max":
                result = torch.stack(collected, dim=0).max(dim=0)[0]
            else:
                raise ValueError(f"Unsupported reduction op {op}")

        # Wait before release
        barrier.wait()
        return result

    def all_gather(self, tensor: torch.Tensor, rank: int, op_id: str, dim: int = -1) -> torch.Tensor:
        """
        Executes an AllGather collective across all ranks.
        Autograd-transparent: slices upstream gradient along dimension during backward pass.
        """
        if self.world_size == 1:
            return tensor.clone() if not tensor.requires_grad else tensor
        if tensor.requires_grad:
            return _AllGatherFunction.apply(tensor, self, rank, op_id, dim)
        return self._sync_all_gather(tensor, rank, op_id, dim)

    def _sync_all_gather(self, tensor: torch.Tensor, rank: int, op_id: str, dim: int = -1) -> torch.Tensor:
        barrier = self._get_barrier(op_id)

        with self._lock:
            self._buffers[op_id][rank] = tensor.detach().clone()

        barrier.wait()

        with self._lock:
            collected = [self._buffers[op_id][r] for r in range(self.world_size)]
            result = torch.cat(collected, dim=dim)

        barrier.wait()
        return result

    def reduce_scatter(self, tensor: torch.Tensor, rank: int, op_id: str, dim: int = -1, op: str = "sum") -> torch.Tensor:
        """
        Executes a ReduceScatter collective across all ranks.
        Autograd-transparent: all-gathers upstream gradient during backward pass.
        """
        if self.world_size == 1:
            return tensor.clone() if not tensor.requires_grad else tensor
        if tensor.requires_grad:
            return _ReduceScatterFunction.apply(tensor, self, rank, op_id, dim, op)
        return self._sync_reduce_scatter(tensor, rank, op_id, dim, op)

    def _sync_reduce_scatter(self, tensor: torch.Tensor, rank: int, op_id: str, dim: int = -1, op: str = "sum") -> torch.Tensor:
        reduced = self._sync_all_reduce(tensor, rank=rank, op_id=f"{op_id}_reduce", op=op)
        slices = torch.chunk(reduced, chunks=self.world_size, dim=dim)
        return slices[rank].clone()
