"""
Parallel Context and Simulation Orchestrator for Tensor Parallelism.
Manages rank IDs, world size, collective sequencing, and multi-threaded simulation.
"""

from typing import Callable, Any, List, Optional
import threading
from .comm import VirtualCommBus


class ParallelContext:
    """
    Context holder for a specific rank in a tensor-parallel process group.
    """

    def __init__(self, rank: int, world_size: int, bus: Optional[VirtualCommBus] = None):
        if rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid rank {rank} for world_size {world_size}")
        self.rank = rank
        self.world_size = world_size
        self.bus = bus or VirtualCommBus(world_size)
        self._op_counter = 0

    @property
    def op_counter(self) -> int:
        return self._op_counter

    def next_op_id(self, prefix: str = "op") -> str:
        """Generates a synchronized sequence ID for collective operations."""
        self._op_counter += 1
        return f"{prefix}_{self._op_counter}"

    def all_reduce(self, tensor, op: str = "sum") -> Any:
        """Executes an AllReduce over the context's virtual bus."""
        op_id = self.next_op_id("all_reduce")
        return self.bus.all_reduce(tensor, rank=self.rank, op_id=op_id, op=op)

    def all_gather(self, tensor, dim: int = -1) -> Any:
        """Executes an AllGather over the context's virtual bus."""
        op_id = self.next_op_id("all_gather")
        return self.bus.all_gather(tensor, rank=self.rank, op_id=op_id, dim=dim)

    def reduce_scatter(self, tensor, dim: int = -1, op: str = "sum") -> Any:
        """Executes a ReduceScatter over the context's virtual bus."""
        op_id = self.next_op_id("reduce_scatter")
        return self.bus.reduce_scatter(tensor, rank=self.rank, op_id=op_id, dim=dim, op=op)


def run_parallel_simulation(world_size: int, worker_fn: Callable[[ParallelContext], Any]) -> List[Any]:
    """
    Executes a worker function concurrently across `world_size` simulated worker threads.
    Returns collected results from all ranks [rank_0_res, rank_1_res, ...].
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    bus = VirtualCommBus(world_size)
    results = [None] * world_size
    exceptions = []

    def _target(r: int):
        ctx = ParallelContext(rank=r, world_size=world_size, bus=bus)
        try:
            results[r] = worker_fn(ctx)
        except Exception as e:
            exceptions.append((r, e))

    threads = [threading.Thread(target=_target, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if exceptions:
        rank, exc = exceptions[0]
        raise RuntimeError(f"Error on rank {rank}: {exc}") from exc

    return results
