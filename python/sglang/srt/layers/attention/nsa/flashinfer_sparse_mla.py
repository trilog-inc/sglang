from typing import Optional

import torch


_GLM_DSA_MODEL_ARCHS = (
    "GlmMoeDsaForCausalLM",
    "GlmMoeDsaForCausalLMNextN",
)


def validate_flashinfer_sparse_mla_backend(
    *,
    model_arch: str,
    device_sm_major: int,
    kv_cache_dtype: torch.dtype,
    prefill_impl: str,
    decode_impl: str,
    is_hip: bool,
) -> bool:
    selected = {prefill_impl, decode_impl}
    uses_flashinfer_sparse_mla = "flashinfer_sparse_mla" in selected
    is_glm_sm12_fp8 = (
        model_arch in _GLM_DSA_MODEL_ARCHS
        and device_sm_major == 12
        and kv_cache_dtype == torch.float8_e4m3fn
        and not is_hip
    )

    if uses_flashinfer_sparse_mla and not is_glm_sm12_fp8:
        raise ValueError(
            "flashinfer_sparse_mla supports only GLM DSA with FP8 KV cache "
            "on NVIDIA SM120/SM121; got "
            f"model_arch={model_arch!r}, sm_major={device_sm_major}, "
            f"kv_cache_dtype={kv_cache_dtype}."
        )
    if is_glm_sm12_fp8:
        unsupported = selected - {"flashinfer_sparse_mla"}
        if unsupported:
            raise ValueError(
                "GLM DSA with FP8 KV cache on NVIDIA SM120/SM121 supports "
                "only flashinfer_sparse_mla; got "
                f"{sorted(unsupported)}."
            )
    return uses_flashinfer_sparse_mla


def flashinfer_sparse_mla_forward(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    page_size: int,
    kv_cache_dim: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sm_scale: float,
    skip_softmax_threshold_scale_factor: Optional[float] = None,
) -> torch.Tensor:
    """Run FlashInfer's SM120 sparse MLA kernel on SGLang's packed NSA cache."""
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    topk = indices.shape[1]
    result = trtllm_batch_decode_with_kv_cache_mla(
        query=q.unsqueeze(1),
        kv_cache=kv_cache.view(torch.uint8)
        .view(-1, page_size, kv_cache_dim)
        .unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=indices.unsqueeze(1),
        seq_lens=seq_lens,
        max_seq_len=topk,
        sparse_mla_top_k=topk,
        bmm1_scale=float(sm_scale),
        bmm2_scale=1.0,
        kv_scale_format="arbitrary_fp32",
        skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
    )
    return result.squeeze(1)
