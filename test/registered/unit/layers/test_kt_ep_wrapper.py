from types import SimpleNamespace

import torch

from sglang.srt.layers.moe import kt_ep_wrapper
from sglang.srt.layers.moe.kt_ep_wrapper import (
    _moe_layer_indices,
    create_kt_config_from_server_args,
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


def test_create_config_uses_server_args_model_config(monkeypatch):
    hf_config = SimpleNamespace(
        num_hidden_layers=6,
        num_hash_layers=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        n_routed_experts=4,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(text_config=hf_config),
    )
    server_args = SimpleNamespace(
        get_model_config=lambda: model_config,
        kt_weight_path="/weights",
        enable_eplb=False,
        kt_method="MXFP4",
        kt_mxfp4_backend="amx",
        kt_mxfp4_amx_min_tokens_per_expert=4,
        kt_gpu_experts_ratio=None,
        kt_num_gpu_experts=2,
        kt_expert_placement_strategy="uniform",
        init_expert_location=None,
        kt_threadpool_count=1,
        kt_cpuinfer=8,
        kt_numa_nodes=[0],
        chunked_prefill_size=4096,
        kt_max_deferred_experts_per_token=None,
    )
    monkeypatch.setattr(kt_ep_wrapper, "_KT_GPU_EXPERTS_MASKS", None)
    monkeypatch.setattr(
        kt_ep_wrapper,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0),
    )
    monkeypatch.setattr(kt_ep_wrapper.dist, "is_initialized", lambda: False)

    config = create_kt_config_from_server_args(server_args, layer_idx=2)

    assert config is not None
    assert config.num_layers == 6
    assert config.method == "MXFP4"
    assert config.gpu_experts_mask.tolist() == [True, True, False, False]
