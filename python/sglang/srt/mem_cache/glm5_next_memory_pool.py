"""GLM-5-Next hybrid KDA/DSA KV cache.

This module deliberately keeps the KPool=4 state behind an exact
GLM-5-Next capability gate.  It does not add KPool fields to the shared NSA
or hybrid pools, so existing Kimi and DeepSeek models retain their original
allocation and indexing contracts.

PD and CP state transfer remain outside this isolated implementation. The
checkpoint-native draft path allocates a compact one-layer DSA pool.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

import torch

from sglang.srt.configs.glm5_next import is_glm5_next, uses_kpool4_compress
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, NSATokenToKVPool
from sglang.srt.mem_cache.pool_registry import register_kv_pool_factory

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


GLM5_NEXT_INDEX_KPOOL = 4
GLM5_NEXT_INDEX_HEAD_DIM = 128
GLM5_NEXT_REAL_PAGE_SIZE = 64
GLM5_NEXT_LATENT_SCALE_GROUP_SIZE = 128
GLM5_NEXT_LATENT_SCALE_GROUPS = 4


@dataclass
class _KPoolStateSnapshot:
    """Small rollback record for one speculative KPool layer invocation."""

    index_buffer: torch.Tensor
    page_indices: torch.Tensor
    page_values: torch.Tensor
    tail_k: torch.Tensor
    tail_score: torch.Tensor
    request_indices: torch.Tensor
    tail_k_values: torch.Tensor
    tail_score_values: torch.Tensor

    def restore(self) -> None:
        if self.page_indices.numel() != 0:
            self.index_buffer.index_copy_(
                0, self.page_indices, self.page_values
            )
        if self.request_indices.numel() != 0:
            self.tail_k.index_copy_(
                0, self.request_indices, self.tail_k_values
            )
            self.tail_score.index_copy_(
                0, self.request_indices, self.tail_score_values
            )

    def capture_current(self) -> _KPoolStateSnapshot:
        """Capture the current values at this snapshot's fixed rows."""

        return _KPoolStateSnapshot(
            index_buffer=self.index_buffer,
            page_indices=self.page_indices,
            page_values=self.index_buffer.index_select(0, self.page_indices).clone(),
            tail_k=self.tail_k,
            tail_score=self.tail_score,
            request_indices=self.request_indices,
            tail_k_values=self.tail_k.index_select(0, self.request_indices).clone(),
            tail_score_values=self.tail_score
            .index_select(0, self.request_indices)
            .clone(),
        )


@dataclass
class _KPoolSpeculativeLayer:
    """Raw target-verification KPool rows awaiting acceptance."""

    key: torch.Tensor
    slot_score: torch.Tensor
    ape: torch.Tensor
    block_tables: torch.Tensor
    req_pool_indices: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    batch_size: int
    tokens_per_request: int
    round_scale: bool
    final_state: Optional[_KPoolStateSnapshot] = None


def _normalize_req_pool_indices(
    req_pool_indices: int | Iterable[int] | torch.Tensor,
    *,
    device: torch.device,
    deduplicate: bool = True,
) -> torch.Tensor:
    """Return one-dimensional request rows on ``device``."""

    if isinstance(req_pool_indices, int):
        rows = torch.tensor([req_pool_indices], dtype=torch.long, device=device)
    elif isinstance(req_pool_indices, torch.Tensor):
        rows = req_pool_indices.to(device=device, dtype=torch.long).reshape(-1)
    else:
        rows = torch.tensor(
            list(req_pool_indices), dtype=torch.long, device=device
        ).reshape(-1)
    return torch.unique(rows) if deduplicate else rows


class Glm5NextNSATokenToKVPool(NSATokenToKVPool):
    """NSA latent/index cache plus the live four-token compression tail.

    KPool's compression gate is vector-valued (one score per index-head
    dimension), so both tail tensors use ``[request, 4, 128]``.  A scalar
    ``[request, 4]`` score buffer is incompatible with the eager KPool Triton
    kernels.
    """

    is_glm5_next_kpool = True

    def __init__(
        self,
        *,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        req_pool_size: int,
        index_kpool: int = GLM5_NEXT_INDEX_KPOOL,
        index_kpool_compress: bool = True,
        index_kpool_always_select_tail: bool = True,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ) -> None:
        if page_size != GLM5_NEXT_REAL_PAGE_SIZE:
            raise ValueError(f"GLM-5-Next KPool requires page_size=64, got {page_size}")
        if index_head_dim != GLM5_NEXT_INDEX_HEAD_DIM:
            raise ValueError(
                f"GLM-5-Next KPool requires index_head_dim=128, got {index_head_dim}"
            )
        if index_kpool != GLM5_NEXT_INDEX_KPOOL:
            raise ValueError(
                f"GLM-5-Next KPool requires index_kpool=4, got {index_kpool}"
            )
        if not index_kpool_compress or not index_kpool_always_select_tail:
            raise ValueError(
                "GLM-5-Next requires compressed KPool with an always-selected tail"
            )
        if req_pool_size <= 0:
            raise ValueError(f"req_pool_size must be positive, got {req_pool_size}")
        if qk_rope_head_dim != 0:
            raise ValueError(
                "GLM-5-Next requires the zero-RoPE MLA cache layout, "
                f"got qk_rope_head_dim={qk_rope_head_dim}"
            )
        if dtype == torch.float8_e4m3fn and kv_cache_dim != kv_lora_rank:
            raise ValueError(
                "GLM-5-Next FP8 with zero-RoPE requires the TRTLLM raw KV "
                f"layout ({kv_lora_rank} bytes/token), got kv_cache_dim="
                f"{kv_cache_dim}. The scaled NSA layout is incompatible "
                "with an empty RoPE component."
            )

        super().__init__(
            size=size,
            page_size=page_size,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            device=device,
            index_head_dim=index_head_dim,
            enable_memory_saver=enable_memory_saver,
            kv_cache_dim=kv_cache_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            index_cache_dtype=(
                torch.bfloat16 if dtype == torch.bfloat16 else torch.float8_e4m3fn
            ),
        )

        # The shared NSA pool selects its scaled FP8 layout when
        # ``override_kv_cache_dim`` is present.  GLM's TRTLLM path instead
        # writes an already-quantized 512-wide key plus an empty RoPE tensor,
        # so it must stay on the raw byte layout.  Keep this assertion local
        # to GLM rather than changing the existing DeepSeek cache contract.
        if dtype == torch.float8_e4m3fn:
            assert not self.nsa_kv_cache_store_fp8
            assert self.kv_cache_dim == kv_lora_rank
        elif dtype != torch.bfloat16:
            raise ValueError(
                "GLM-5-Next consumer cache supports only BF16 or FP8 E4M3, "
                f"got {dtype}"
            )

        self.index_kpool = index_kpool
        self.index_kpool_compress = index_kpool_compress
        self.index_kpool_always_select_tail = index_kpool_always_select_tail
        self.tail_extra_slots = 0
        self.slots_per_page = self.page_size
        self.req_pool_size = req_pool_size
        self._speculative_kpool_layers: dict[int, _KPoolSpeculativeLayer] = {}

        allocation_context = (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool is not None
            else nullcontext()
        )
        with (
            self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            allocation_context,
        ):
            # Keep the 512-byte latent cache ABI consumed by TRTLLM/Triton and
            # store its four per-128-channel descales in an exact-GLM sidecar.
            # This mirrors NSA index-cache scale semantics without changing
            # any shared MLA or DeepSeek cache layout.
            self._latent_scale = (
                [
                    torch.zeros(
                        (
                            size + page_size,
                            GLM5_NEXT_LATENT_SCALE_GROUPS,
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                    for _ in range(layer_num)
                ]
                if dtype == torch.float8_e4m3fn
                else []
            )
            self._compress_tail_k = [
                torch.zeros(
                    (req_pool_size, index_kpool, index_head_dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
                for _ in range(layer_num)
            ]
            self._compress_tail_score = [
                torch.zeros(
                    (req_pool_size, index_kpool, index_head_dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
                for _ in range(layer_num)
            ]

        # NSATokenToKVPool finalizes its accounting before this subclass can
        # allocate the tail.  Refresh the value without emitting a duplicate
        # allocation log.
        self.mem_usage = self.get_kv_size_bytes() / (1024**3)

    def _local_layer_index(self, layer_id: int) -> int:
        local_idx = layer_id - self.start_layer
        if not 0 <= local_idx < self.layer_num:
            raise ValueError(
                f"layer_id={layer_id} is outside compact GLM DSA range "
                f"[{self.start_layer}, {self.start_layer + self.layer_num})"
            )
        return local_idx

    def get_compress_tail_buffers(
        self, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        local_idx = self._local_layer_index(layer_id)
        return self._compress_tail_k[local_idx], self._compress_tail_score[local_idx]

    def get_latent_scale_buffer(self, layer_id: int) -> Optional[torch.Tensor]:
        """Return the physical-token block descales for one compact DSA layer."""

        if not self._latent_scale:
            return None
        return self._latent_scale[self._local_layer_index(layer_id)]

    def set_latent_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        """Write dynamic latent descales beside their FP8 cache rows."""

        target = self.get_latent_scale_buffer(layer_id)
        if target is None:
            raise RuntimeError("latent scales are available only for GLM FP8 KV cache")
        if scale.ndim == 3 and scale.shape[-2] == 1:
            scale = scale.squeeze(-2)
        expected = (loc.numel(), GLM5_NEXT_LATENT_SCALE_GROUPS)
        if tuple(scale.shape) != expected:
            raise ValueError(
                f"GLM latent scale must have shape {expected}, got {tuple(scale.shape)}"
            )
        if scale.dtype != torch.float32:
            raise TypeError(f"GLM latent scale must be FP32, got {scale.dtype}")
        target[loc] = scale

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        """Fail closed for the top-k tree cache-relocation contract."""

        raise NotImplementedError(
            "GLM-5-Next MTP supports topk=1 only; KV-cache relocation for "
            "multi-branch speculative trees is not implemented"
        )

    def clear_compress_tail_rows(
        self, req_pool_indices: int | Iterable[int] | torch.Tensor
    ) -> None:
        """Clear stale live-tail state before reuse or before releasing rows."""

        self.discard_speculative_kpool()
        if not self._compress_tail_k:
            return
        rows = _normalize_req_pool_indices(
            req_pool_indices, device=self._compress_tail_k[0].device
        )
        if rows.numel() == 0:
            return
        if torch.any(rows < 0) or torch.any(rows >= self.req_pool_size):
            raise IndexError(
                f"request rows must be in [0, {self.req_pool_size}), got "
                f"{rows.detach().cpu().tolist()}"
            )
        for tail_k, tail_score in zip(
            self._compress_tail_k, self._compress_tail_score, strict=True
        ):
            tail_k.index_fill_(0, rows, 0)
            tail_score.index_fill_(0, rows, 0)

    def prepare_kpool_request(
        self, req_pool_indices: int | Iterable[int] | torch.Tensor
    ) -> None:
        """Initialize newly allocated rows before their first prefill write."""

        self.clear_compress_tail_rows(req_pool_indices)

    def snapshot_speculative_kpool_state(
        self,
        *,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        packed_write_locs: torch.Tensor,
        deduplicate_indices: bool = True,
    ) -> _KPoolStateSnapshot:
        """Snapshot only pages/tails a speculative KPool call can mutate."""

        local_idx = self._local_layer_index(layer_id)
        index_buffer = self.get_index_k_with_scale_buffer(layer_id)
        page_indices = torch.div(
            packed_write_locs.to(device=index_buffer.device, dtype=torch.long),
            self.slots_per_page,
            rounding_mode="floor",
        )
        if deduplicate_indices:
            page_indices = torch.unique(page_indices)
        request_indices = _normalize_req_pool_indices(
            req_pool_indices,
            device=self._compress_tail_k[local_idx].device,
            deduplicate=deduplicate_indices,
        )
        return _KPoolStateSnapshot(
            index_buffer=index_buffer,
            page_indices=page_indices,
            page_values=index_buffer.index_select(0, page_indices).clone(),
            tail_k=self._compress_tail_k[local_idx],
            tail_score=self._compress_tail_score[local_idx],
            request_indices=request_indices,
            tail_k_values=self._compress_tail_k[local_idx]
            .index_select(0, request_indices)
            .clone(),
            tail_score_values=self._compress_tail_score[local_idx]
            .index_select(0, request_indices)
            .clone(),
        )

    def stage_speculative_kpool_layer(
        self,
        *,
        layer_id: int,
        key: torch.Tensor,
        slot_score: torch.Tensor,
        ape: torch.Tensor,
        block_tables: torch.Tensor,
        req_pool_indices: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        batch_size: int,
        tokens_per_request: int,
        round_scale: bool,
    ) -> None:
        """Retain target-verification rows until the verifier accepts a prefix."""

        local_idx = self._local_layer_index(layer_id)
        rows = batch_size * tokens_per_request
        tensors = {
            "key": key,
            "slot_score": slot_score,
            "block_tables": block_tables,
            "req_pool_indices": req_pool_indices,
            "positions": positions,
            "seq_lens": seq_lens,
            "out_cache_loc": out_cache_loc,
        }
        for name, tensor in tensors.items():
            if tensor is None:
                raise ValueError(
                    f"speculative KPool staged {name} is None for layer {layer_id} "
                    f"(batch_size={batch_size}, tokens_per_request="
                    f"{tokens_per_request})"
                )
            if tensor.shape[0] < rows:
                raise ValueError(
                    f"speculative KPool {name} has {tensor.shape[0]} rows; "
                    f"expected at least {rows}"
                )
        self._speculative_kpool_layers[local_idx] = _KPoolSpeculativeLayer(
            key=key[:rows].detach(),
            slot_score=slot_score[:rows].detach(),
            ape=ape.detach(),
            block_tables=block_tables[:rows].detach(),
            req_pool_indices=req_pool_indices[:rows].detach(),
            positions=positions[:rows].detach(),
            seq_lens=seq_lens[:rows].detach(),
            out_cache_loc=out_cache_loc[:rows].detach(),
            batch_size=batch_size,
            tokens_per_request=tokens_per_request,
            round_scale=round_scale,
        )

    def commit_speculative_kpool(
        self, accepted_token_counts: Iterable[int]
    ) -> None:
        """Commit only each request's accepted top-k=1 verification prefix."""

        counts = [int(count) for count in accepted_token_counts]
        staged_layers = self._speculative_kpool_layers
        self._speculative_kpool_layers = {}
        for layer_id, staged in staged_layers.items():
            if len(counts) != staged.batch_size:
                raise ValueError(
                    "accepted KPool counts must match the staged batch size; "
                    f"got {len(counts)} and {staged.batch_size}"
                )
            if any(
                count < 0 or count > staged.tokens_per_request
                for count in counts
            ):
                raise ValueError(
                    "accepted KPool counts must remain inside the single-branch "
                    f"verification width {staged.tokens_per_request}; got {counts}"
                )

            if hasattr(self, "invalidate_index_buffer_for_layer"):
                self.invalidate_index_buffer_for_layer(layer_id)

            if staged.final_state is not None and all(
                count == staged.tokens_per_request for count in counts
            ):
                # The verify pass already computed this exact full-width state.
                # Restoring it avoids replaying every KPool update kernel.
                staged.final_state.restore()
                continue

            for step in range(staged.tokens_per_request):
                active_requests = [
                    request for request, count in enumerate(counts) if count > step
                ]
                if not active_requests:
                    continue
                rows = torch.tensor(
                    [
                        request * staged.tokens_per_request + step
                        for request in active_requests
                    ],
                    dtype=torch.long,
                    device=staged.key.device,
                )
                self.kpool_decode_update_index_cache(
                    layer_id=layer_id,
                    key=staged.key.index_select(0, rows),
                    slot_score=staged.slot_score.index_select(0, rows),
                    ape=staged.ape,
                    block_tables=staged.block_tables.index_select(0, rows),
                    req_pool_indices=staged.req_pool_indices.index_select(0, rows),
                    positions=staged.positions.index_select(0, rows),
                    seq_lens=staged.seq_lens.index_select(0, rows),
                    out_cache_loc=staged.out_cache_loc.index_select(0, rows),
                    round_scale=staged.round_scale,
                )

    def capture_speculative_kpool_final_state(
        self, *, layer_id: int, transaction: _KPoolStateSnapshot
    ) -> None:
        local_idx = self._local_layer_index(layer_id)
        staged = self._speculative_kpool_layers.get(local_idx)
        if staged is None:
            raise RuntimeError(
                f"missing staged KPool layer {layer_id} before final snapshot"
            )
        staged.final_state = transaction.capture_current()

    def discard_speculative_kpool(self) -> None:
        """Drop uncommitted verification rows during request/error cleanup."""

        self._speculative_kpool_layers.clear()

    def kpool_decode_update_index_cache(
        self,
        *,
        layer_id: int,
        key: torch.Tensor,
        slot_score: torch.Tensor,
        ape: torch.Tensor,
        block_tables: torch.Tensor,
        req_pool_indices: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        round_scale: bool = False,
    ) -> None:
        from sglang.srt.layers.attention.nsa.kpool_fp8_index import (
            kpool_decode_update_and_maybe_write_cache,
        )

        local_idx = self._local_layer_index(layer_id)
        kpool_decode_update_and_maybe_write_cache(
            pool=self,
            buf=self.get_index_k_with_scale_buffer(layer_id),
            tail_k=self._compress_tail_k[local_idx],
            tail_score=self._compress_tail_score[local_idx],
            key=key,
            slot_score=slot_score,
            ape=ape,
            block_tables=block_tables,
            req_pool_indices=req_pool_indices,
            positions=positions,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            round_scale=round_scale,
        )

    def get_kv_size_bytes(self):
        size_bytes = super().get_kv_size_bytes()
        latent_scale = getattr(self, "_latent_scale", ())
        tail_k = getattr(self, "_compress_tail_k", ())
        tail_score = getattr(self, "_compress_tail_score", ())
        return size_bytes + sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*latent_scale, *tail_k, *tail_score)
        )


class Glm5NextHybridKVPool(HybridLinearKVPool):
    """Hybrid wrapper mapping global DSA layer IDs to a compact NSA pool."""

    is_glm5_next_kpool = True

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        """Reject multi-branch speculative relocation at the public seam."""

        raise NotImplementedError(
            "GLM-5-Next MTP supports topk=1 only; KV-cache relocation for "
            "multi-branch speculative trees is not implemented"
        )

    def __init__(
        self,
        *,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: Iterable[int],
        device: str,
        mamba_pool,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        index_head_dim: int,
        kv_cache_dim: int,
        req_pool_size: int,
        enable_memory_saver: bool = False,
        index_kpool: int = GLM5_NEXT_INDEX_KPOOL,
        index_kpool_compress: bool = True,
        index_kpool_always_select_tail: bool = True,
    ) -> None:
        full_attention_layer_ids = list(full_attention_layer_ids)
        if len(set(full_attention_layer_ids)) != len(full_attention_layer_ids):
            raise ValueError("GLM DSA layer IDs must be unique")
        if full_attention_layer_ids != sorted(full_attention_layer_ids):
            raise ValueError("GLM DSA layer IDs must be sorted")

        # HybridLinearKVPool cannot construct an NSA full-attention pool in
        # this KT baseline, so initialize its stable wrapper state explicitly
        # and inherit its MLA/mamba delegation methods.
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = page_size
        self.start_layer = 0
        self.head_num = head_num
        self.head_dim = head_dim
        self.mamba_pool = mamba_pool
        self.use_mla = True
        self.use_dsa = True
        self.full_attention_layer_id_mapping = {
            global_id: compact_id
            for compact_id, global_id in enumerate(full_attention_layer_ids)
        }
        self.full_kv_pool = Glm5NextNSATokenToKVPool(
            size=size,
            page_size=page_size,
            kv_lora_rank=kv_lora_rank,
            dtype=dtype,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=self.full_layer_nums,
            device=device,
            index_head_dim=index_head_dim,
            enable_memory_saver=enable_memory_saver,
            kv_cache_dim=kv_cache_dim,
            req_pool_size=req_pool_size,
            index_kpool=index_kpool,
            index_kpool_compress=index_kpool_compress,
            index_kpool_always_select_tail=index_kpool_always_select_tail,
            start_layer=0,
        )
        self.mem_usage = self.full_kv_pool.mem_usage
        self.kpool_lifecycle_coordinator = None

    @property
    def nsa_kv_cache_store_fp8(self) -> bool:
        return self.full_kv_pool.nsa_kv_cache_store_fp8

    @property
    def kv_cache_dim(self) -> int:
        return self.full_kv_pool.kv_cache_dim

    @property
    def index_head_dim(self) -> int:
        return self.full_kv_pool.index_head_dim

    @property
    def quant_block_size(self) -> int:
        return self.full_kv_pool.quant_block_size

    @property
    def index_cache_dtype(self) -> torch.dtype:
        return self.full_kv_pool.index_cache_dtype

    @property
    def index_cache_is_bf16(self) -> bool:
        return self.full_kv_pool.index_cache_is_bf16

    @property
    def index_kpool(self) -> int:
        return self.full_kv_pool.index_kpool

    @property
    def index_kpool_compress(self) -> bool:
        return self.full_kv_pool.index_kpool_compress

    @property
    def index_kpool_always_select_tail(self) -> bool:
        return self.full_kv_pool.index_kpool_always_select_tail

    @property
    def tail_extra_slots(self) -> int:
        return self.full_kv_pool.tail_extra_slots

    @property
    def slots_per_page(self) -> int:
        return self.full_kv_pool.slots_per_page

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.full_kv_pool.get_index_k_with_scale_buffer(
            self._transfer_full_attention_id(layer_id)
        )

    def get_broadcastable_index_k_with_scale_buffer(
        self, layer_id: int
    ) -> torch.Tensor:
        return self.get_index_k_with_scale_buffer(layer_id)

    def get_index_k_continuous(
        self, layer_id: int, seq_len: int, page_indices: torch.Tensor
    ):
        return self.full_kv_pool.get_index_k_continuous(
            self._transfer_full_attention_id(layer_id), seq_len, page_indices
        )

    def get_index_k_scale_continuous(
        self, layer_id: int, seq_len: int, page_indices: torch.Tensor
    ):
        return self.full_kv_pool.get_index_k_scale_continuous(
            self._transfer_full_attention_id(layer_id), seq_len, page_indices
        )

    def get_index_k_scale_buffer(
        self, layer_id: int, seq_len: int, page_indices: torch.Tensor
    ):
        return self.full_kv_pool.get_index_k_scale_buffer(
            self._transfer_full_attention_id(layer_id), seq_len, page_indices
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        self.full_kv_pool.set_index_k_scale_buffer(
            self._transfer_full_attention_id(layer_id),
            loc,
            index_k,
            index_k_scale,
        )

    def get_compress_tail_buffers(
        self, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.full_kv_pool.get_compress_tail_buffers(
            self._transfer_full_attention_id(layer_id)
        )

    def get_latent_scale_buffer(self, layer_id: int) -> Optional[torch.Tensor]:
        return self.full_kv_pool.get_latent_scale_buffer(
            self._transfer_full_attention_id(layer_id)
        )

    def set_latent_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        self.full_kv_pool.set_latent_scale_buffer(
            self._transfer_full_attention_id(layer_id), loc, scale
        )

    def clear_compress_tail_rows(
        self, req_pool_indices: int | Iterable[int] | torch.Tensor
    ) -> None:
        self.full_kv_pool.clear_compress_tail_rows(req_pool_indices)

    def prepare_kpool_request(
        self, req_pool_indices: int | Iterable[int] | torch.Tensor
    ) -> None:
        self.full_kv_pool.prepare_kpool_request(req_pool_indices)

    def kpool_decode_update_index_cache(self, *, layer_id: int, **kwargs) -> None:
        self.full_kv_pool.kpool_decode_update_index_cache(
            layer_id=self._transfer_full_attention_id(layer_id), **kwargs
        )

    def snapshot_speculative_kpool_state(
        self, *, layer_id: int, **kwargs
    ) -> _KPoolStateSnapshot:
        return self.full_kv_pool.snapshot_speculative_kpool_state(
            layer_id=self._transfer_full_attention_id(layer_id), **kwargs
        )

    def stage_speculative_kpool_layer(self, *, layer_id: int, **kwargs) -> None:
        self.full_kv_pool.stage_speculative_kpool_layer(
            layer_id=self._transfer_full_attention_id(layer_id), **kwargs
        )

    def commit_speculative_kpool(
        self, accepted_token_counts: Iterable[int]
    ) -> None:
        self.full_kv_pool.commit_speculative_kpool(accepted_token_counts)

    def capture_speculative_kpool_final_state(
        self, *, layer_id: int, transaction: _KPoolStateSnapshot
    ) -> None:
        self.full_kv_pool.capture_speculative_kpool_final_state(
            layer_id=self._transfer_full_attention_id(layer_id),
            transaction=transaction,
        )

    def discard_speculative_kpool(self) -> None:
        self.full_kv_pool.discard_speculative_kpool()


def _glm5_next_pool_predicate(model_config, server_args) -> bool:
    del server_args
    return is_glm5_next(model_config) and uses_kpool4_compress(model_config)


def build_glm5_next_kv_pool(
    *, model_runner: "ModelRunner"
) -> Glm5NextHybridKVPool | Glm5NextNSATokenToKVPool:
    """ModelRunner-facing factory registered only for exact GLM-5-Next KPool4."""

    model_config = model_runner.model_config
    if not _glm5_next_pool_predicate(model_config, model_runner.server_args):
        raise ValueError("GLM-5-Next KV pool factory received a non-GLM configuration")
    if not model_runner.use_mla_backend:
        raise ValueError("GLM-5-Next DSA requires the MLA backend")
    if (
        not model_runner.is_draft_worker
        and not hasattr(model_runner.req_to_token_pool, "mamba_pool")
    ):
        raise RuntimeError(
            "GLM-5-Next requires HybridReqToTokenPool before constructing its KV pool"
        )

    text_config = model_config.hf_text_config
    if model_runner.is_draft_worker:
        # The appended block is one DSA layer.  Do not allocate the target's
        # 34 KDA recurrent states merely because they share one HF config.
        pool = Glm5NextNSATokenToKVPool(
            size=model_runner.max_total_num_tokens,
            page_size=model_runner.page_size,
            kv_lora_rank=model_config.kv_lora_rank,
            dtype=model_runner.kv_cache_dtype,
            qk_rope_head_dim=model_config.qk_rope_head_dim,
            layer_num=1,
            device=model_runner.device,
            index_head_dim=text_config.index_head_dim,
            enable_memory_saver=model_runner.server_args.enable_memory_saver,
            kv_cache_dim=model_runner.calculate_mla_kv_cache_dim(),
            req_pool_size=model_runner.req_to_token_pool.size,
            index_kpool=text_config.index_kpool,
            index_kpool_compress=text_config.index_kpool_compress,
            index_kpool_always_select_tail=(
                text_config.index_kpool_always_select_tail
            ),
            start_layer=0,
            end_layer=1,
        )
        from sglang.srt.managers.glm5_next_kpool_coordinator import (
            attach_glm5_next_kpool_lifecycle,
        )

        pool.kpool_lifecycle_coordinator = attach_glm5_next_kpool_lifecycle(pool)
        return pool

    full_attention_layer_ids = [
        layer_id
        for layer_id in text_config.full_attention_layer_ids
        if model_runner.start_layer <= layer_id < model_runner.end_layer
    ]

    from sglang.srt.layers.dp_attention import get_attention_tp_size

    pool = Glm5NextHybridKVPool(
        size=model_runner.max_total_num_tokens,
        dtype=model_runner.kv_cache_dtype,
        page_size=model_runner.page_size,
        head_num=model_config.get_num_kv_heads(get_attention_tp_size()),
        head_dim=model_config.head_dim,
        full_attention_layer_ids=full_attention_layer_ids,
        device=model_runner.device,
        mamba_pool=model_runner.req_to_token_pool.mamba_pool,
        kv_lora_rank=model_config.kv_lora_rank,
        qk_rope_head_dim=model_config.qk_rope_head_dim,
        index_head_dim=text_config.index_head_dim,
        kv_cache_dim=model_runner.calculate_mla_kv_cache_dim(),
        req_pool_size=model_runner.req_to_token_pool.size,
        enable_memory_saver=model_runner.server_args.enable_memory_saver,
        index_kpool=text_config.index_kpool,
        index_kpool_compress=text_config.index_kpool_compress,
        index_kpool_always_select_tail=(text_config.index_kpool_always_select_tail),
    )

    from sglang.srt.managers.glm5_next_kpool_coordinator import (
        attach_glm5_next_kpool_lifecycle,
    )

    pool.kpool_lifecycle_coordinator = attach_glm5_next_kpool_lifecycle(pool)
    return pool


register_kv_pool_factory(
    "glm5_next_kpool4", _glm5_next_pool_predicate, build_glm5_next_kv_pool
)
