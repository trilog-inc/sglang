"""CPU-only contracts for KT heterogeneous NVFP4 expert storage and loading."""

import inspect
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt import server_args as server_args_module
from sglang.srt.layers.moe import kt_ep_wrapper
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization import modelopt_quant


def _identity_swizzle(scale: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(scale)


def _make_nvfp4_method():
    method = object.__new__(modelopt_quant.ModelOptNvFp4FusedMoEMethod)
    method.quant_config = SimpleNamespace(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
    )
    return method


def test_kt_nvfp4_allocates_compact_resident_weights_and_global_input_scales(
    monkeypatch,
):
    monkeypatch.setattr(modelopt_quant, "swizzle_blockscale", _identity_swizzle)
    method = _make_nvfp4_method()
    layer = torch.nn.Module()
    layer.num_experts = 8
    layer.num_local_experts = 8
    layer.moe_runner_config = SimpleNamespace(is_gated=True)

    method.create_weights(
        layer=layer,
        num_experts=3,
        hidden_size=32,
        intermediate_size_per_partition=16,
        params_dtype=torch.bfloat16,
    )

    compact_names = (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_weight_scale_2",
        "w2_weight_scale_2",
    )
    assert {name: getattr(layer, name).shape[0] for name in compact_names} == {
        name: 3 for name in compact_names
    }
    assert layer.w13_input_scale.shape == (8, 2)
    assert layer.w2_input_scale.shape == (8,)
    assert layer.w13_input_scale._sglang_require_global_experts is True
    assert layer.w2_input_scale._sglang_require_global_experts is True


def _make_loader_layer():
    layer = object.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.num_experts = 8
    layer.num_local_experts = 8
    layer.num_fused_shared_experts = 0
    layer.moe_ep_rank = 0
    layer.layer_id = 0
    layer.quant_method = SimpleNamespace(
        _quant_wrapper_id="kt_ep",
        num_gpu_experts=3,
        gpu_experts_mask=torch.tensor(
            [False, True, False, False, True, False, False, True]
        ),
        logical_to_gpu_index=torch.tensor(
            [-1, 0, -1, -1, 1, -1, -1, 2], dtype=torch.int32
        ),
    )

    def copy_loaded_value(
        *, param, loaded_weight, weight_name, shard_id, expert_id
    ):
        del weight_name, shard_id
        param.data[expert_id, 0].copy_(loaded_weight)

    layer._weight_loader_impl = mock.Mock(side_effect=copy_loaded_value)
    return layer


class _IdentityExpertLocationMetadata:
    def logical_to_all_physical(
        self, layer_id, logical_expert_id, require_global_experts=False
    ):
        del layer_id, require_global_experts
        return [logical_expert_id]


@pytest.mark.parametrize("metadata", [None, _IdentityExpertLocationMetadata()])
def test_kt_loader_remaps_only_compact_resident_tensors(monkeypatch, metadata):
    monkeypatch.setattr(
        fused_moe_layer,
        "get_global_expert_location_metadata",
        lambda: metadata,
    )
    layer = _make_loader_layer()
    resident_param = torch.nn.Parameter(torch.empty(3, 1))
    global_param = torch.nn.Parameter(torch.empty(8, 1))
    global_param._sglang_require_global_experts = True
    loaded = torch.tensor(4.0)

    layer.weight_loader(resident_param, loaded, "weight", "w2", expert_id=4)
    assert layer._weight_loader_impl.call_args.kwargs["expert_id"] == 1
    assert resident_param[1, 0].item() == 4.0

    layer._weight_loader_impl.reset_mock()
    layer.weight_loader(resident_param, loaded, "weight", "w2", expert_id=2)
    layer._weight_loader_impl.assert_not_called()

    layer.weight_loader(global_param, loaded, "input_scale", "w2", expert_id=2)
    assert layer._weight_loader_impl.call_args.kwargs["expert_id"] == 2
    assert global_param[2, 0].item() == 4.0


def test_kt_runtime_ids_use_compact_resident_indices():
    topk_ids = torch.tensor([[1, 2, 7, 4], [0, 4, 5, 1]])
    gpu_experts_mask = torch.tensor(
        [False, True, False, False, True, False, False, True]
    )
    logical_to_gpu_index = torch.tensor(
        [-1, 0, -1, -1, 1, -1, -1, 2], dtype=torch.int32
    )
    eager_mask_and_remap = getattr(
        kt_ep_wrapper.mask_and_remap_expert_ids,
        "_torchdynamo_orig_callable",
        kt_ep_wrapper.mask_and_remap_expert_ids,
    )

    actual = eager_mask_and_remap(
        topk_ids, gpu_experts_mask, logical_to_gpu_index
    )

    assert torch.equal(
        actual,
        torch.tensor([[0, -1, 2, 1], [-1, 1, -1, 0]], dtype=torch.int32),
    )


def test_kt_cpu_gpu_merge_reuses_the_gpu_output_buffer():
    apply_source = inspect.getsource(kt_ep_wrapper.KTEPWrapperMethod.apply)

    assert "output.add_(cpu_output)" in apply_source
    assert "output = output + cpu_output" not in apply_source


def test_sm120_modelopt_fp4_auto_selects_flashinfer_cutlass(monkeypatch):
    monkeypatch.setattr(server_args_module, "is_sm120_supported", lambda: True)
    monkeypatch.setattr(
        server_args_module, "get_bool_env_var", lambda _name: False
    )
    args = object.__new__(server_args_module.ServerArgs)
    args.quantization = "modelopt_fp4"
    args.moe_runner_backend = "auto"
    args.ep_size = 1
    args.tp_size = 2

    args._handle_moe_kernel_config()

    assert args.moe_runner_backend == "flashinfer_cutlass"


@pytest.mark.parametrize(
    ("gpu_prefill_token_threshold", "dynamic_update", "option"),
    [
        (64, False, "kt-gpu-prefill-token-threshold"),
        (0, True, "kt-enable-dynamic-expert-update"),
    ],
)
def test_kt_nvfp4_rejects_non_static_resident_modes(
    gpu_prefill_token_threshold, dynamic_update, option
):
    nvfp4_method = type("ModelOptNvFp4FusedMoEMethod", (), {})()
    config = SimpleNamespace(
        gpu_prefill_token_threshold=gpu_prefill_token_threshold,
        kt_enable_dynamic_expert_update=dynamic_update,
    )

    with pytest.raises(ValueError, match=option):
        kt_ep_wrapper._validate_kt_nvfp4_static_resident_config(
            nvfp4_method, config
        )
