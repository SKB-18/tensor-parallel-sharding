# Tensor Parallel Sharding Engine

[![Tests](https://img.shields.io/badge/tests-41%2F41%20passed-brightgreen.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](#)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](#)

A first-principles, industrial-grade implementation of **Megatron-LM style Tensor Parallelism (TP)** and **Sequence Parallelism (SP)** for Large Language Models in PyTorch. Partitions parameter weights across $P$ virtual or physical accelerator ranks to divide model memory by $P$ while guaranteeing exact machine-level numerical parity ($\le 2 \times 10^{-7}$ maximum absolute difference vs. un-sharded monolithic models).

---

## Architecture & Communication Topology

```
                  Input Activations X [B, S, H]
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
       Rank 0 Worker                     Rank 1 Worker
  ┌────────────────────────┐        ┌────────────────────────┐
  │ ColumnParallel Q, K, V │        │ ColumnParallel Q, K, V │
  │ W_col: [H, H/P] (x3)   │        │ W_col: [H, H/P] (x3)   │
  │ No Communication       │        │ No Communication       │
  └───────────┬────────────┘        └───────────┬────────────┘
              │                                 │
              ▼                                 ▼
  ┌────────────────────────┐        ┌────────────────────────┐
  │ Local Attention Core   │        │ Local Attention Core   │
  │ Softmax(QK^T / sqrt(d))│        │ Softmax(QK^T / sqrt(d))│
  │ Heads: [0 .. H/P - 1]  │        │ Heads: [H/P .. H - 1]  │
  └───────────┬────────────┘        └───────────┬────────────┘
              │                                 │
              ▼                                 ▼
  ┌────────────────────────┐        ┌────────────────────────┐
  │ RowParallel Out Proj   │        │ RowParallel Out Proj   │
  │ W_row: [H/P, H]        │        │ W_row: [H/P, H]        │
  │ Local MatMul Z_0       │        │ Local MatMul Z_1       │
  └───────────┬────────────┘        └───────────┬────────────┘
              │                                 │
              └───────────────┬─────────────────┘
                              │
                    AllReduce(SUM): Z_0 + Z_1
                              │
                              ▼
            Post-Attention Norm & Residual Addition
                              │
               ┌──────────────┴──────────────────┐
               ▼                                 ▼
  ┌────────────────────────┐        ┌────────────────────────┐
  │ ColumnParallel MLP     │        │ ColumnParallel MLP     │
  │ W_gate/up: [H, 4H/P]   │        │ W_gate/up: [H, 4H/P]   │
  │ Local Activation       │        │ Local Activation       │
  └───────────┬────────────┘        └───────────┬────────────┘
              │                                 │
              ▼                                 ▼
  ┌────────────────────────┐        ┌────────────────────────┐
  │ RowParallel MLP Down   │        │ RowParallel MLP Down   │
  │ W_down: [4H/P, H]      │        │ W_down: [4H/P, H]      │
  │ Local MatMul           │        │ Local MatMul           │
  └───────────┬────────────┘        └───────────┬────────────┘
              │                                 │
              └───────────────┬─────────────────┘
                              │
                    AllReduce(SUM): Down Proj
                              │
                              ▼
                  Output Tensor Y [B, S, H]
```

### Key Communication Invariant
By alternating **Column-Parallel** projections with **Row-Parallel** projections:
- There is **zero communication** between QKV projections and attention head computation.
- There is **zero communication** between MLP Up-projections and activation functions.
- Exactly **two AllReduce(SUM)** operations occur per Transformer block (1 at attention exit, 1 at MLP exit).

---

## Mathematical Formulations

### 1. Column-Parallel Linear Layer
Given input $X \in \mathbb{R}^{B \times S \times d_{\text{in}}}$ and weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$:
Partition $W$ across $P$ ranks along columns:
$$W = \begin{bmatrix} W_1 & W_2 & \dots & W_P \end{bmatrix}, \quad W_i \in \mathbb{R}^{d_{\text{in}} \times \frac{d_{\text{out}}}{P}}$$
Each rank $i$ computes local matrix multiplication without inter-rank communication:
$$Y_i = X W_i \in \mathbb{R}^{B \times S \times \frac{d_{\text{out}}}{P}}$$

### 2. Row-Parallel Linear Layer
Given column-partitioned input $X_i \in \mathbb{R}^{B \times S \times \frac{d_{\text{in}}}{P}}$ and weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$:
Partition $W$ across $P$ ranks along rows:
$$W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_P \end{bmatrix}, \quad W_i \in \mathbb{R}^{\frac{d_{\text{in}}}{P} \times d_{\text{out}}}$$
Each rank computes local dot products:
$$Z_i = X_i W_i \in \mathbb{R}^{B \times S \times d_{\text{out}}}$$
Aggregated via an inter-rank `AllReduce(SUM)`:
$$Y = \sum_{i=1}^P Z_i + b$$
Bias $b$ is added post-reduction, preventing $P \times$ bias duplication.

### 3. Model Checkpoint Slicing (`ModelSharder`)
Given a standard monolithic checkpoint `state_dict`, partitions weights according to pattern matching:
- **Column-Parallel** (`.*q_proj\.weight`, `.*k_proj\.weight`, `.*v_proj\.weight`, `.*fc1\.weight`): Sliced along dimension 1.
- **Row-Parallel** (`.*out_proj\.weight`, `.*fc2\.weight`, `.*down_proj\.weight`): Sliced along dimension 0.
- **Replicated** (RMSNorms, LayerNorms): Copied intact to all ranks.

---

## Empirical Benchmarks

Run benchmarks via `python cli.py benchmark`:

### 1. Memory Scaling per Rank Across World Sizes
Evaluated on a 50.3M parameter model ($H=1024, L=4, H_{\text{heads}}=16, I=4096$):
| World Size ($P$) | Parameters per Rank | Weight Memory per Rank | Memory Savings | Reduction Ratio |
| :---: | :---: | :---: | :---: | :---: |
| **$P=1$ (Mono)** | 50,339,840 | 192.03 MB | Baseline | $1.0\times$ |
| **$P=2$** | 25,174,016 | 96.03 MB | **50.0%** | **$2.0\times$** |
| **$P=4$** | 12,591,104 | 48.03 MB | **75.0%** | **$4.0\times$** |
| **$P=8$** | 6,299,648 | 24.03 MB | **87.5%** | **$8.0\times$** |

### 2. Exact Numerical Parity Verification
Comparing monolithic baseline vs. tensor-parallel model across identical inputs:
| Metric | World Size $P=2$ | World Size $P=4$ | Status |
| :--- | :---: | :---: | :---: |
| **Max Absolute Difference** | $1.788 \times 10^{-7}$ | $2.086 \times 10^{-7}$ | ✅ **[PASSED]** |
| **Mean Absolute Difference** | $4.291 \times 10^{-8}$ | $5.127 \times 10^{-8}$ | ✅ **[PASSED]** |
| **Numerical Tolerance** | $< 1.000 \times 10^{-5}$ | $< 1.000 \times 10^{-5}$ | ✅ **[PASSED]** |

---

## Quickstart & CLI

### Installation
```bash
git clone https://github.com/SKB-18/tensor-parallel-sharding.git
cd tensor-parallel-sharding
pip install -e .
```

### 1. Inspect Sharding Distribution
```bash
python cli.py inspect --world-size 4 --hidden-size 1024 --num-heads 16 --num-layers 12
```

### 2. Run Simulated Parallel Forward Pass
```bash
python cli.py forward --world-size 2 --hidden-size 128 --num-heads 4 --num-layers 2
```

### 3. Verify Exact Numerical Parity
```bash
python cli.py parity
```

---

## Automated Test Suite

Run full automated tests:
```bash
python -m pytest tests/ -v
```

```
tests/test_cli.py::test_cli_inspect PASSED                               [  2%]
tests/test_cli.py::test_cli_forward PASSED                               [  4%]
tests/test_cli.py::test_cli_parity PASSED                                [  7%]
tests/test_cli.py::test_cli_benchmark PASSED                             [  9%]
tests/test_column_parallel.py::test_column_parallel_indivisible_features PASSED [ 12%]
tests/test_column_parallel.py::test_column_parallel_forward_shapes PASSED [ 14%]
tests/test_column_parallel.py::test_column_parallel_gather_output_parity PASSED [ 17%]
tests/test_column_parallel.py::test_column_parallel_no_bias PASSED       [ 19%]
tests/test_column_parallel.py::test_column_parallel_single_rank PASSED   [ 21%]
tests/test_comm.py::test_comm_invalid_world_size PASSED                  [ 24%]
tests/test_comm.py::test_all_reduce_sum PASSED                           [ 26%]
tests/test_comm.py::test_all_reduce_mean PASSED                          [ 29%]
tests/test_comm.py::test_all_reduce_max PASSED                           [ 31%]
tests/test_comm.py::test_all_gather PASSED                               [ 34%]
tests/test_comm.py::test_reduce_scatter PASSED                           [ 36%]
tests/test_comm.py::test_comm_single_rank_pass_through PASSED            [ 39%]
tests/test_comm.py::test_parallel_context_op_counter PASSED              [ 41%]
tests/test_parallel_mha.py::test_mha_indivisible_heads PASSED            [ 43%]
tests/test_parallel_mha.py::test_mha_numerical_parity_p2 PASSED          [ 46%]
tests/test_parallel_mha.py::test_mha_numerical_parity_with_causal_mask PASSED [ 48%]
tests/test_parallel_mha.py::test_mha_load_without_bias PASSED            [ 51%]
tests/test_parallel_mlp.py::test_mlp_indivisible_intermediate PASSED     [ 53%]
tests/test_parallel_mlp.py::test_mlp_gelu_parity PASSED                  [ 56%]
tests/test_parallel_mlp.py::test_mlp_swiglu_parity PASSED                [ 58%]
tests/test_parallel_mlp.py::test_mlp_load_without_bias PASSED            [ 60%]
tests/test_row_parallel.py::test_row_parallel_indivisible_features PASSED [ 63%]
tests/test_row_parallel.py::test_row_parallel_forward_parity PASSED      [ 65%]
tests/test_row_parallel.py::test_row_parallel_bias_not_multiplied PASSED [ 68%]
tests/test_row_parallel.py::test_row_parallel_no_bias PASSED             [ 70%]
tests/test_row_parallel.py::test_row_parallel_single_rank PASSED         [ 73%]
tests/test_sharder.py::test_sharder_round_trip_bit_parity PASSED         [ 75%]
tests/test_sharder.py::test_sharder_indivisible_dimension_raises PASSED  [ 78%]
tests/test_sharder.py::test_sharder_save_and_load_disk PASSED            [ 80%]
tests/test_sharder.py::test_sharder_custom_patterns PASSED               [ 82%]
tests/test_stress.py::test_tp_scaling_world_sizes PASSED                 [ 85%]
tests/test_stress.py::test_large_batch_and_sequence PASSED               [ 87%]
tests/test_stress.py::test_deterministic_reproducibility PASSED          [ 90%]
tests/test_transformer.py::test_sequence_parallel_norm_and_scatter_gather PASSED [ 92%]
tests/test_transformer.py::test_sequence_parallel_indivisible_seq_len PASSED [ 95%]
tests/test_transformer.py::test_transformer_block_forward PASSED         [ 97%]
tests/test_transformer.py::test_multi_layer_parallel_transformer PASSED  [100%]

============================= 41 passed in 28.81s =============================
```

---

## License
MIT License. Created by [SKB-18](https://github.com/SKB-18).
