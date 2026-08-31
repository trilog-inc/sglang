import torch

from sglang.srt.layers.attention.nsa_backend import (
    _deep_gemm_paged_mqa_context_lens,
)


def test_deep_gemm_paged_mqa_context_lens_adds_next_n_axis():
    context_lens = torch.tensor([128, 256], dtype=torch.int32)

    normalized = _deep_gemm_paged_mqa_context_lens(context_lens)

    assert normalized.shape == (2, 1)
    assert normalized.is_contiguous()
    assert torch.equal(normalized[:, 0], context_lens)


def test_deep_gemm_paged_mqa_context_lens_preserves_2d_layout():
    context_lens = torch.tensor([[128, 129], [256, 257]], dtype=torch.int32)

    normalized = _deep_gemm_paged_mqa_context_lens(context_lens)

    assert normalized.shape == (2, 2)
    assert normalized.is_contiguous()
    assert torch.equal(normalized, context_lens)
