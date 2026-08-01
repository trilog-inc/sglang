from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.kt_ep_wrapper import (
    _moe_layer_indices,
    mask_and_remap_expert_ids,
)


def test_mask_and_remap_arbitrary_gpu_experts_and_padding():
    gpu_mask = torch.tensor([False, False, True, False, False, True])
    logical_to_gpu = torch.tensor([-1, -1, 0, -1, -1, 1])
    logical_ids = torch.tensor([[2, 5, 1, -1, 6]])

    actual = mask_and_remap_expert_ids(logical_ids, gpu_mask, logical_to_gpu)

    torch.testing.assert_close(actual, torch.tensor([[0, 1, -1, -1, -1]]))


def test_mask_and_remap_compact_cpu_complement():
    cpu_mask = torch.tensor([True, True, False, True, True, False])
    logical_to_cpu = torch.tensor([0, 1, -1, 2, 3, -1])
    logical_ids = torch.tensor([[0, 2, 5, 4, -1, 6]])

    actual = mask_and_remap_expert_ids(logical_ids, cpu_mask, logical_to_cpu)

    torch.testing.assert_close(actual, torch.tensor([[0, -1, -1, 3, -1, -1]]))


def test_dsv4_hash_prefix_is_included_in_moe_layers():
    config = SimpleNamespace(
        num_hidden_layers=43,
        num_hash_layers=3,
        first_k_dense_replace=3,
        moe_layer_freq=1,
    )

    assert _moe_layer_indices(config) == list(range(43))


def test_sparse_moe_frequency_is_respected():
    config = SimpleNamespace(
        num_hidden_layers=8,
        num_hash_layers=0,
        first_k_dense_replace=2,
        moe_layer_freq=2,
    )

    assert _moe_layer_indices(config) == [2, 4, 6]
