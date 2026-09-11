import logging
from typing import Optional, Tuple

import torch
import triton

from sglang.srt.environ import envs
from sglang.srt.utils.common import is_sm120_supported

logger = logging.getLogger(__name__)

_FUSED_HC_POST_PRE_M_THRESHOLD = 64
_FUSED_HC_POST_PRE_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_TRITON_MHC_POST_PRE_OPS = None
_TRITON_MHC_POST_PRE_RUNTIME_DISABLED = False


def _is_fused_mhc_post_pre_enabled() -> bool:
    """Return whether the production TileLang post+pre path is available."""
    return (
        envs.SGLANG_OPT_FUSE_MHC_POST_PRE.get()
        and envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get()
        and (envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get() or is_sm120_supported())
    )


def is_cross_layer_mhc_fusion_enabled() -> bool:
    """Whether DeepSeek V4 may defer mHC post across a layer boundary."""
    return _is_fused_mhc_post_pre_enabled()


def _get_triton_mhc_post_pre_ops():
    global _TRITON_MHC_POST_PRE_OPS

    if _TRITON_MHC_POST_PRE_OPS is not None:
        return _TRITON_MHC_POST_PRE_OPS

    try:
        from aiter.ops.triton.fusions.mhc import mhc_post_pre
        from aiter.ops.triton.utils.mhc_config_utils import get_mhc_config
    except Exception as err:
        logger.warning(
            "Triton fused mHC (mhc_post_pre) is unavailable, falling back: %s", err
        )
        return None

    _TRITON_MHC_POST_PRE_OPS = (mhc_post_pre, get_mhc_config)
    return _TRITON_MHC_POST_PRE_OPS


def _get_fused_hc_post_pre_buffers(
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[dict[str, torch.Tensor]]:
    ops = _get_triton_mhc_post_pre_ops()
    if ops is None:
        return None
    _, get_mhc_config = ops

    key = (num_tokens, hidden_size, hc_mult, dtype, device.type, device.index)
    bufs = _FUSED_HC_POST_PRE_CACHE.get(key)
    if bufs is not None:
        return bufs

    try:
        cfg, _ = get_mhc_config("MHC_FUSED", num_tokens, hidden_size, mode="sinkhorn")
    except Exception as err:
        logger.warning("Failed to initialize fused mHC config, falling back: %s", err)
        return None

    n_total = 2 * hc_mult + hc_mult * hc_mult
    k_dim = hc_mult * hidden_size
    block_k = cfg.get("BLOCK_K", min(512, triton.next_power_of_2(k_dim)))
    block_k = min(block_k, triton.next_power_of_2(k_dim))
    block_c_split = max(block_k // hc_mult, 1)
    num_ksplit = triton.cdiv(hidden_size, block_c_split)

    bufs = {
        "residual_out": torch.empty(
            num_tokens, hc_mult, hidden_size, dtype=dtype, device=device
        ),
        "layer_input_out": torch.empty(
            num_tokens, hidden_size, dtype=dtype, device=device
        ),
        "h_post": torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device),
        "h_res": torch.empty(
            num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=device
        ),
        "acc_partial": torch.empty(
            num_ksplit, num_tokens, n_total, dtype=torch.float32, device=device
        ),
        "acc_sq_partial": torch.empty(
            num_ksplit, num_tokens, dtype=torch.float32, device=device
        ),
    }
    _FUSED_HC_POST_PRE_CACHE[key] = bufs
    return bufs


def try_fused_hc_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_fn_t: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    norm_eps: float,
    hc_eps: float,
    hc_post_mult: float,
    sinkhorn_iters: int,
    is_gfx95_supported: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]]:
    global _TRITON_MHC_POST_PRE_RUNTIME_DISABLED

    if (
        _TRITON_MHC_POST_PRE_RUNTIME_DISABLED
        or not envs.SGLANG_OPT_USE_TRITON_FUSED_MHC.get()
        or not is_gfx95_supported
        or x.shape[0] == 0
        or x.shape[0] > _FUSED_HC_POST_PRE_M_THRESHOLD
        or x.dim() != 2
        or residual.dim() != 3
    ):
        return None

    ops = _get_triton_mhc_post_pre_ops()
    if ops is None:
        return None
    mhc_post_pre, _ = ops

    bufs = _get_fused_hc_post_pre_buffers(
        x.shape[0], x.shape[1], hc_mult, residual.dtype, x.device
    )
    if bufs is None:
        return None

    try:
        _, _, layer_input_out, new_residual = mhc_post_pre(
            x,
            residual,
            post,
            comb,
            hc_fn_t,
            hc_scale,
            hc_base,
            hc_mult,
            norm_eps,
            hc_eps,
            hc_post_mult,
            sinkhorn_iters,
            # Match sglang's exp-domain asymmetric Sinkhorn used in hc_pre.
            asymmetric_exp_domain=True,
            hc_sinkhorn_eps=hc_eps,
            residual_out=bufs["residual_out"],
            h_post=bufs["h_post"],
            h_res=bufs["h_res"],
            layer_input_out=bufs["layer_input_out"],
            acc_partial=bufs["acc_partial"],
            acc_sq_partial=bufs["acc_sq_partial"],
        )
    except Exception as err:
        logger.warning(
            "Triton fused mHC kernel failed, disabling fallback path: %s", err
        )
        _TRITON_MHC_POST_PRE_RUNTIME_DISABLED = True
        return None

    return new_residual, layer_input_out, bufs["h_post"], bufs["h_res"], False


def apply_mhc_post_pre_boundary(
    layer_input: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    rms_eps: float,
    hc_eps: float,
    hc_post_mult: float,
    sinkhorn_iters: int,
    norm_weight: Optional[torch.Tensor],
    norm_eps: Optional[float],
    *,
    fn_transpose: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]]:
    """Apply the production TileLang fused mHC boundary when enabled."""
    if not _is_fused_mhc_post_pre_enabled():
        return None

    # Imported lazily to avoid a circular module import and optional CUDA
    # runtime initialization during model-registry discovery.
    from sglang.srt.models.deepseek_v4 import _get_mhc_ops

    post_in = post.unsqueeze(-1) if post.ndim == 2 else post
    (
        residual,
        post_out,
        comb_out,
        layer_input_out,
    ) = _get_mhc_ops().mhc_fused_post_pre(
        layer_input,
        residual,
        post_in,
        comb,
        hc_fn.T if fn_transpose else hc_fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        hc_eps,
        hc_post_mult,
        sinkhorn_iters,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    post_out = post_out.squeeze(-1) if post_out.ndim == 3 else post_out
    return residual, layer_input_out, post_out, comb_out, True
