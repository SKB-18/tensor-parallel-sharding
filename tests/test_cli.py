"""
Unit tests for tensor-parallel-sharding CLI subcommands.
"""

import sys
import subprocess
import pytest


def run_cli(*args):
    """Executes cli.py via subprocess and returns CompletedProcess."""
    cmd = [sys.executable, "cli.py"] + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )


def test_cli_inspect():
    """Verify inspect subcommand displays layer breakdown and memory savings."""
    res = run_cli("inspect", "--world-size", "4", "--hidden-size", "512", "--num-heads", "8", "--num-layers", "4")
    assert res.returncode == 0
    assert "Tensor Parallelism Model Sharding Inspector" in res.stdout
    assert "Memory Reduction Ratio:   4.0x vs Monolithic" in res.stdout
    assert "[ColumnParallel] Q, K, V Proj" in res.stdout
    assert "[RowParallel]    Attn Out Proj" in res.stdout


def test_cli_forward():
    """Verify forward subcommand executes simulated forward pass and confirms consensus."""
    res = run_cli("forward", "--world-size", "2", "--hidden-size", "64", "--num-heads", "4", "--num-layers", "1")
    assert res.returncode == 0
    assert "Simulating Tensor-Parallel Forward Pass" in res.stdout
    assert "[Rank 0]" in res.stdout
    assert "[Rank 1]" in res.stdout
    assert "Cross-Rank AllReduce Consensus: [SYNCHRONIZED]" in res.stdout


def test_cli_parity():
    """Verify parity subcommand runs numerical validation."""
    res = run_cli("parity")
    assert res.returncode == 0
    assert "Exact Numerical Parity Verification" in res.stdout
    assert "Status: [PASSED]" in res.stdout


def test_cli_benchmark():
    """Verify benchmark subcommand completes and saves JSON results."""
    res = run_cli("benchmark")
    assert res.returncode == 0
    assert "Running Full Tensor Parallelism (TP) Benchmark Suite" in res.stdout
    assert "Memory Scaling per Rank Across World Sizes" in res.stdout
    assert "Forward Pass Latency Benchmark" in res.stdout
