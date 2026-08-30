import sys
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.nsa.flashinfer_sparse_mla import (
    flashinfer_sparse_mla_forward,
    validate_flashinfer_sparse_mla_backend,
)


def _mock_flashinfer(op):
    flashinfer = ModuleType("flashinfer")
    flashinfer.__path__ = []
    mla = ModuleType("flashinfer.mla")
    mla.trtllm_batch_decode_with_kv_cache_mla = op
    flashinfer.mla = mla
    return patch.dict(
        sys.modules,
        {"flashinfer": flashinfer, "flashinfer.mla": mla},
    )


def test_maps_sglang_layout_to_public_flashinfer_api():
    captured = {}

    def fake_op(**kwargs):
        captured.update(kwargs)
        query = kwargs["query"]
        return query.new_full((*query.shape[:-1], kwargs["kv_lora_rank"]), 2)

    with _mock_flashinfer(fake_op):
        output = flashinfer_sparse_mla_forward(
            q=torch.zeros((2, 8, 576), dtype=torch.bfloat16),
            kv_cache=torch.zeros((128, 1, 656), dtype=torch.uint8),
            indices=torch.tensor(
                [[7, 9, -1, -1], [4, 6, 8, -1]], dtype=torch.int32
            ),
            seq_lens=torch.tensor([2, 3], dtype=torch.int32),
            workspace_buffer=torch.zeros(1024, dtype=torch.uint8),
            page_size=64,
            kv_cache_dim=656,
            qk_nope_head_dim=192,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            sm_scale=0.125,
        )

    assert captured["query"].shape == (2, 1, 8, 576)
    assert captured["kv_cache"].shape == (2, 1, 64, 656)
    assert captured["block_tables"].shape == (2, 1, 4)
    assert captured["seq_lens"].tolist() == [2, 3]
    assert captured["max_seq_len"] == 4
    assert captured["sparse_mla_top_k"] == 4
    assert captured["bmm1_scale"] == 0.125
    assert captured["kv_scale_format"] == "arbitrary_fp32"
    assert "backend" not in captured
    assert output.shape == (2, 8, 512)


def _validate(prefill, decode, model_arch="GlmMoeDsaForCausalLM"):
    return validate_flashinfer_sparse_mla_backend(
        model_arch=model_arch,
        device_sm_major=12,
        kv_cache_dtype=torch.float8_e4m3fn,
        prefill_impl=prefill,
        decode_impl=decode,
        is_hip=False,
    )


@pytest.mark.parametrize(
    "model_arch",
    ["GlmMoeDsaForCausalLM", "GlmMoeDsaForCausalLMNextN"],
)
def test_accepts_flashinfer_for_both_phases(model_arch):
    assert _validate(
        "flashinfer_sparse_mla", "flashinfer_sparse_mla", model_arch
    )


@pytest.mark.parametrize(
    ("prefill", "decode"),
    [("trtllm", "trtllm"), ("flashinfer_sparse_mla", "trtllm")],
)
def test_rejects_other_or_mixed_backends(prefill, decode):
    with pytest.raises(ValueError, match="only flashinfer_sparse_mla"):
        _validate(prefill, decode)


def test_reports_unsupported_configuration():
    with pytest.raises(ValueError, match="DeepseekV3ForCausalLM"):
        _validate(
            "flashinfer_sparse_mla",
            "flashinfer_sparse_mla",
            "DeepseekV3ForCausalLM",
        )
