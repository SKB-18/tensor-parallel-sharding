"""
Communication Primitives and Virtual Bus for Tensor Parallelism.
Implements AllReduce, AllGather, and ReduceScatter over virtual in-memory ranks.
"""

from typing import List, Dict, Optional, Tuple
import threading
import torch


class VirtualCommBus:
    """
    In-memory virtual communication bus simulating distributed tensor collectives
    (AllReduce, AllGather, ReduceScatter) with exact synchronization across virtual ranks.
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
        Combines tensors from all ranks using specified reduction operator ('sum', 'mean', 'max').
        """
        if self.world_size == 1:
            return tensor.clone()

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

        # Wait before cleanup
        barrier.wait()

        # Cleanup memory for this op_id
        with self._lock:
            if op_id in self._buffers and len(self._buffers[op_id]) == self.world_size:
                # Last rank out clears
                pass

        return result

    def all_gather(self, tensor: torch.Tensor, rank: int, op_id: str, dim: int = -1) -> torch.Tensor:
        """
        Executes an AllGather collective across all ranks.
        Concatenates tensors from all ranks along specified dimension.
        """
        if self.world_size == 1:
            return tensor.clone()

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
        First reduces tensors across all ranks, then scatters slices along dimension to each rank.
        """
        reduced = self.all_reduce(tensor, rank=rank, op_id=f"{op_id}_reduce", op=op)
        slices = torch.chunk(reduced, chunks=self.world_size, dim=dim)
        return slices[rank].clone()
