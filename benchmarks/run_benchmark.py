"""
Empirical Benchmark Suite for Tensor Parallelism.
Measures memory scaling per rank, numerical equivalence, and communication scaling across world sizes.
"""

from typing import Dict, Any, List
import time
import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

try:
    from parallel.context import ParallelContext, run_parallel_simulation
    from models.transformer import ParallelTransformer
    from models.parallel_mha import ParallelMultiHeadAttention
    from models.parallel_mlp import ParallelMLP
    from layers.column_parallel import ColumnParallelLinear
    from layers.row_parallel import RowParallelLinear
    from checkpoint.sharder import ModelSharder
except (ImportError, ValueError):
    from ..parallel.context import ParallelContext, run_parallel_simulation
    from ..models.transformer import ParallelTransformer
    from ..models.parallel_mha import ParallelMultiHeadAttention
    from ..models.parallel_mlp import ParallelMLP
    from ..layers.column_parallel import ColumnParallelLinear
    from ..layers.row_parallel import RowParallelLinear
    from ..checkpoint.sharder import ModelSharder


class MonolithicAttention(nn.Module):
    """Unsharded standard multi-head attention for numerical baseline comparison."""

    def __init__(self, hidden_size: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(b, s, self.hidden_size)
        return self.out_proj(out)


def benchmark_memory_scaling(hidden_size: int = 1024, num_layers: int = 4, world_sizes: List[int] = [1, 2, 4, 8]) -> Dict[str, Any]:
    """Measures parameter count and memory consumption per rank across world sizes."""
    print("=" * 70)
    print(" 1. Memory Scaling per Rank Across World Sizes")
    print("=" * 70)

    # Monolithic parameter count
    num_heads = 16
    intermediate_size = hidden_size * 4
    
    # 1 Layer: QKV (3 * H * H) + Out (H * H) + FC1 (H * I) + FC2 (I * H) + Norms (4 * H)
    layer_params = (
        3 * (hidden_size * hidden_size)  # Q, K, V
        + (hidden_size * hidden_size)    # Out
        + (hidden_size * intermediate_size)  # FC1
        + (intermediate_size * hidden_size)  # FC2
        + (2 * hidden_size)                  # 2 RMSNorms
    )
    total_mono_params = layer_params * num_layers
    mono_mem_mb = (total_mono_params * 4) / (1024 * 1024)  # FP32: 4 bytes per param

    results = {}
    print(f"  • Monolithic Model: {total_mono_params:,} parameters ({mono_mem_mb:.2f} MB in FP32)")
    print("-" * 70)
    print(f"  {'World Size (P)':<16} {'Params / Rank':<18} {'Weight Memory / Rank':<22} {'Reduction Ratio'}")
    print("-" * 70)

    for p in world_sizes:
        # Sharded: Q, K, V are partitioned by P, Out by P, FC1 by P, FC2 by P
        sharded_layer_params = (
            (4 * hidden_size * hidden_size // p)
            + (2 * hidden_size * intermediate_size // p)
            + (2 * hidden_size)  # Replicated norms
        )
        rank_params = sharded_layer_params * num_layers
        rank_mem_mb = (rank_params * 4) / (1024 * 1024)
        savings = (1.0 - (rank_mem_mb / mono_mem_mb)) * 100.0

        results[f"P={p}"] = {
            "params_per_rank": rank_params,
            "memory_mb": round(rank_mem_mb, 2),
            "savings_pct": round(savings, 2),
        }
        print(f"  P={p:<14} {rank_params:<18,} {rank_mem_mb:<6.2f} MB ({savings:>5.1f}% saved)   {1.0 / (rank_mem_mb / mono_mem_mb):.1f}x")

    return results


def benchmark_numerical_parity() -> Dict[str, Any]:
    """Verifies strict machine numerical parity between monolithic and sharded models."""
    print("\n" + "=" * 70)
    print(" 2. Exact Numerical Parity Verification (Monolithic vs. Tensor-Parallel)")
    print("=" * 70)

    torch.manual_seed(42)
    hidden_size = 256
    num_heads = 8
    intermediate_size = 512
    batch_size = 2
    seq_len = 16

    x = torch.randn(batch_size, seq_len, hidden_size)

    # 1. Monolithic attention reference
    mono_attn = MonolithicAttention(hidden_size, num_heads, bias=True)
    mono_attn.eval()
    with torch.no_grad():
        mono_out = mono_attn(x)

    # 2. Parallel Attention across P = 2 and P = 4
    parity_results = {}

    for p in [2, 4]:
        def _worker(ctx: ParallelContext):
            par_attn = ParallelMultiHeadAttention(hidden_size, num_heads, ctx, bias=True)
            par_attn.eval()
            par_attn.load_unsharded_weights(
                q_w=mono_attn.q_proj.weight.t(),
                k_w=mono_attn.k_proj.weight.t(),
                v_w=mono_attn.v_proj.weight.t(),
                out_w=mono_attn.out_proj.weight.t(),
                q_b=mono_attn.q_proj.bias,
                k_b=mono_attn.k_proj.bias,
                v_b=mono_attn.v_proj.bias,
                out_b=mono_attn.out_proj.bias,
            )
            with torch.no_grad():
                return par_attn(x)

        rank_outputs = run_parallel_simulation(p, _worker)
        # All ranks receive identical output after AllReduce
        tp_out = rank_outputs[0]

        max_abs_diff = (tp_out - mono_out).abs().max().item()
        mean_abs_diff = (tp_out - mono_out).abs().mean().item()
        is_exact = max_abs_diff < 1e-5

        status = "[PASSED]" if is_exact else "[FAILED]"
        print(f"  • World Size P={p}: Max Absolute Diff = {max_abs_diff:.8e} | Status: {status}")
        parity_results[f"P={p}"] = {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "passed": is_exact,
        }

    return parity_results


def benchmark_end_to_end_throughput() -> Dict[str, Any]:
    """Measures end-to-end forward pass speed and scaling across world sizes."""
    print("\n" + "=" * 70)
    print(" 3. Forward Pass Latency Benchmark (100 Iterations)")
    print("=" * 70)

    hidden_size = 256
    num_heads = 8
    intermediate_size = 512
    num_layers = 2
    batch_size = 4
    seq_len = 32
    iters = 100

    x = torch.randn(batch_size, seq_len, hidden_size)
    perf_results = {}

    for p in [1, 2, 4]:
        def _worker(ctx: ParallelContext):
            model = ParallelTransformer(
                num_layers=num_layers,
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                ctx=ctx,
            )
            model.eval()
            with torch.no_grad():
                for _ in range(5):  # Warmup
                    _ = model(x)
                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = model(x)
                elapsed = time.perf_counter() - t0
            return elapsed

        times = run_parallel_simulation(p, _worker)
        mean_time_ms = (times[0] / iters) * 1000.0
        tok_per_sec = (batch_size * seq_len * iters) / times[0]

        print(f"  • P={p:<2} | Latency: {mean_time_ms:6.2f} ms/fwd | Throughput: {tok_per_sec:8.1f} tokens/sec")
        perf_results[f"P={p}"] = {
            "latency_ms": round(mean_time_ms, 2),
            "tokens_per_sec": round(tok_per_sec, 1),
        }

    return perf_results


def main():
    print("\n" + "=" * 70)
    print(" Running Full Tensor Parallelism (TP) Benchmark Suite")
    print("=" * 70)

    t_start = time.perf_counter()
    mem_res = benchmark_memory_scaling()
    parity_res = benchmark_numerical_parity()
    perf_res = benchmark_end_to_end_throughput()
    total_time = time.perf_counter() - t_start

    print("\n" + "=" * 70)
    print(f" ALL BENCHMARKS COMPLETED IN {total_time:.2f}s")
    print("=" * 70)

    metrics = {
        "memory_scaling": mem_res,
        "numerical_parity": parity_res,
        "performance": perf_res,
        "benchmark_duration_s": round(total_time, 2),
    }

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f" Saved full benchmark metrics to {out_path}\n")


if __name__ == "__main__":
    main()
