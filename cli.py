"""
Interactive and Scriptable CLI for Tensor Parallelism (TP) Systems.
Supports inspecting tensor sharding layouts, running forward passes, slicing checkpoints, and executing benchmarks.
"""

import argparse
import sys
import os
import json
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from parallel.context import ParallelContext, run_parallel_simulation
from layers.column_parallel import ColumnParallelLinear
from layers.row_parallel import RowParallelLinear
from models.parallel_mha import ParallelMultiHeadAttention
from models.parallel_mlp import ParallelMLP
from models.transformer import ParallelTransformer
from checkpoint.sharder import ModelSharder


def cmd_inspect(args):
    """Inspects sharding distribution and parameter memory for a model across P ranks."""
    print("=" * 68)
    print(f" Tensor Parallelism Model Sharding Inspector (World Size P={args.world_size})")
    print("=" * 68)

    hidden = args.hidden_size
    heads = args.num_heads
    inter = hidden * 4
    p = args.world_size

    if heads % p != 0:
        print(f"  [ERROR] num_heads ({heads}) not divisible by world_size ({p})")
        return

    head_dim = hidden // heads
    local_heads = heads // p
    local_inter = inter // p

    print(f"  • Hidden Size:              {hidden}")
    print(f"  • Total Attention Heads:    {heads} (Head Dim = {head_dim})")
    print(f"  • Heads per Rank:           {local_heads}")
    print(f"  • Intermediate (MLP) Dim:   {inter}")
    print(f"  • MLP Dim per Rank:         {local_inter}")
    print("-" * 68)
    print(" Layer Sharding Breakdown per Transformer Block:")
    print(f"   [ColumnParallel] Q, K, V Proj : [{hidden} -> {local_heads * head_dim}] per rank (x3)")
    print(f"   [RowParallel]    Attn Out Proj: [{local_heads * head_dim} -> {hidden}] per rank -> AllReduce(SUM)")
    print(f"   [ColumnParallel] MLP Gate/Up  : [{hidden} -> {local_inter}] per rank")
    print(f"   [RowParallel]    MLP Down Proj: [{local_inter} -> {hidden}] per rank -> AllReduce(SUM)")
    print("-" * 68)

    layer_params = 4 * (hidden * hidden // p) + 2 * (hidden * inter // p) + 2 * hidden
    total_params = layer_params * args.num_layers
    mem_mb = (total_params * 4) / (1024 * 1024)

    print(f"  • Layers:                   {args.num_layers}")
    print(f"  • Parameters per Rank:      {total_params:,}")
    print(f"  • Weight Memory per Rank:   {mem_mb:.2f} MB (in FP32)")
    print(f"  • Memory Reduction Ratio:   {p:.1f}x vs Monolithic")
    print("=" * 68)


def cmd_forward(args):
    """Executes a simulated tensor-parallel forward pass across all ranks."""
    print("=" * 68)
    print(f" Simulating Tensor-Parallel Forward Pass (P={args.world_size})")
    print("=" * 68)

    b, s, h = args.batch_size, args.seq_len, args.hidden_size
    torch.manual_seed(42)
    x = torch.randn(b, s, h)

    # Initialize master weights and shard across world_size
    master_ctx = ParallelContext(rank=0, world_size=1)
    master_model = ParallelTransformer(
        num_layers=args.num_layers,
        hidden_size=h,
        num_heads=args.num_heads,
        intermediate_size=h * 4,
        ctx=master_ctx,
    )
    sharder = ModelSharder(world_size=args.world_size)
    rank_state_dicts = sharder.shard_state_dict(master_model.state_dict())

    def _worker(ctx: ParallelContext):
        model = ParallelTransformer(
            num_layers=args.num_layers,
            hidden_size=h,
            num_heads=args.num_heads,
            intermediate_size=h * 4,
            ctx=ctx,
        )
        model.load_state_dict(rank_state_dicts[ctx.rank])
        model.eval()
        with torch.no_grad():
            out = model(x)
        return {
            "rank": ctx.rank,
            "out_shape": list(out.shape),
            "sample_val": out[0, 0, 0].item(),
        }

    results = run_parallel_simulation(args.world_size, _worker)

    print(f"  • Input Shape:       {list(x.shape)}")
    print(f"  • World Size:        {args.world_size}")
    print("-" * 68)
    print(" Rank Forward Execution Status:")
    for r in results:
        print(f"   [Rank {r['rank']}] Output Shape: {r['out_shape']} | Anchor Token: {r['sample_val']:.6f}")

    # Check that all ranks computed matching output after AllReduce
    anchors = [r["sample_val"] for r in results]
    is_synced = all(abs(a - anchors[0]) < 1e-6 for a in anchors)
    print("-" * 68)
    print(f"  • Cross-Rank AllReduce Consensus: {'[SYNCHRONIZED]' if is_synced else '[DESYNC]'}")
    print("=" * 68)


def cmd_parity(args):
    """Runs numerical parity test comparing monolithic vs tensor-parallel outputs."""
    from benchmarks.run_benchmark import benchmark_numerical_parity
    benchmark_numerical_parity()


def cmd_benchmark(args):
    """Runs the full empirical benchmark suite."""
    from benchmarks.run_benchmark import main as run_bench
    run_bench()


def main():
    parser = argparse.ArgumentParser(
        description="tensor-parallel-sharding CLI: Megatron-LM Tensor Parallelism and Model Slicing"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_insp = subparsers.add_parser("inspect", help="Inspect sharding layout and memory distribution")
    p_insp.add_argument("--hidden-size", "-H", type=int, default=1024, help="Model hidden size")
    p_insp.add_argument("--num-heads", "-a", type=int, default=16, help="Attention heads")
    p_insp.add_argument("--num-layers", "-l", type=int, default=12, help="Number of transformer layers")
    p_insp.add_argument("--world-size", "-p", type=int, default=4, help="Tensor parallel world size")
    p_insp.set_defaults(func=cmd_inspect)

    # forward
    p_fwd = subparsers.add_parser("forward", help="Run simulated parallel forward pass")
    p_fwd.add_argument("--batch-size", "-b", type=int, default=2, help="Batch size")
    p_fwd.add_argument("--seq-len", "-s", type=int, default=8, help="Sequence length")
    p_fwd.add_argument("--hidden-size", "-H", type=int, default=128, help="Hidden size")
    p_fwd.add_argument("--num-heads", "-a", type=int, default=4, help="Attention heads")
    p_fwd.add_argument("--num-layers", "-l", type=int, default=2, help="Number of layers")
    p_fwd.add_argument("--world-size", "-p", type=int, default=2, help="World size")
    p_fwd.set_defaults(func=cmd_forward)

    # parity
    p_par = subparsers.add_parser("parity", help="Verify numerical equivalence vs monolithic baseline")
    p_par.set_defaults(func=cmd_parity)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run full performance and scaling benchmark suite")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
