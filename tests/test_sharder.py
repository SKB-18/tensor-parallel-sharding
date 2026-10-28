"""
Unit tests for ModelSharder checkpoint slicing and reconstruction.
"""

import pytest
import os
import tempfile
import torch
from checkpoint.sharder import ModelSharder
from checkpoint.loader import save_rank_checkpoints, load_rank_checkpoint


def test_sharder_round_trip_bit_parity():
    """Verify slicing and reconstituting a state_dict yields bitwise identical tensors."""
    world_size = 4
    sharder = ModelSharder(world_size=world_size)

    monolithic = {
        # Column-parallel weights: [in, out] -> sliced along dim 1
        "layers.0.attn.q_proj.weight": torch.randn(64, 128),
        "layers.0.attn.q_proj.bias": torch.randn(128),
        # Row-parallel weights: [in, out] -> sliced along dim 0
        "layers.0.attn.out_proj.weight": torch.randn(128, 64),
        "layers.0.attn.out_proj.bias": torch.randn(64),
        # Replicated weights
        "layers.0.norm1.weight": torch.randn(64),
    }

    # 1. Shard
    rank_dicts = sharder.shard_state_dict(monolithic)
    assert len(rank_dicts) == world_size

    # Verify sliced shapes on rank 0
    assert rank_dicts[0]["layers.0.attn.q_proj.weight"].shape == torch.Size([64, 32])
    assert rank_dicts[0]["layers.0.attn.q_proj.bias"].shape == torch.Size([32])
    assert rank_dicts[0]["layers.0.attn.out_proj.weight"].shape == torch.Size([32, 64])
    assert rank_dicts[0]["layers.0.attn.out_proj.bias"].shape == torch.Size([64])  # Replicated
    assert rank_dicts[0]["layers.0.norm1.weight"].shape == torch.Size([64])  # Replicated

    # 2. Reconstitute
    reconstituted = sharder.reconstitute_monolithic_state_dict(rank_dicts)

    # 3. Assert bitwise equivalence
    assert set(reconstituted.keys()) == set(monolithic.keys())
    for k in monolithic:
        assert torch.equal(reconstituted[k], monolithic[k]), f"Mismatch in key {k}"


def test_sharder_indivisible_dimension_raises():
    """Verify sharder raises ValueError when weight dimension is not divisible by world_size."""
    sharder = ModelSharder(world_size=4)
    bad_dict = {
        # Dim 1 is 10 (not divisible by 4)
        "layers.0.attn.q_proj.weight": torch.randn(64, 10),
    }
    with pytest.raises(ValueError, match="not divisible by world_size"):
        sharder.shard_state_dict(bad_dict)


def test_sharder_save_and_load_disk():
    """Verify rank state_dicts save to disk and reload accurately."""
    world_size = 2
    sharder = ModelSharder(world_size=world_size)
    monolithic = {
        "layers.0.attn.q_proj.weight": torch.randn(32, 64),
        "layers.0.norm1.weight": torch.randn(32),
    }

    rank_dicts = sharder.shard_state_dict(monolithic)

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = save_rank_checkpoints(rank_dicts, tmp_dir, prefix="test_tp")
        assert len(paths) == 2

        # Reload rank 1
        loaded_r1 = load_rank_checkpoint(tmp_dir, rank=1, prefix="test_tp")
        assert "layers.0.attn.q_proj.weight" in loaded_r1
        assert torch.equal(loaded_r1["layers.0.attn.q_proj.weight"], rank_dicts[1]["layers.0.attn.q_proj.weight"])


def test_sharder_custom_patterns():
    """Verify ModelSharder correctly applies user-defined custom column/row regex patterns."""
    custom_col = {r"custom_encoder\.dense_in\.weight"}
    custom_row = {r"custom_encoder\.dense_out\.weight"}
    sharder = ModelSharder(world_size=2, column_patterns=custom_col, row_patterns=custom_row)

    assert sharder.is_column_parallel("custom_encoder.dense_in.weight")
    assert sharder.is_row_parallel("custom_encoder.dense_out.weight")
    assert not sharder.is_column_parallel("other.weight")
    assert not sharder.is_row_parallel("other.weight")
