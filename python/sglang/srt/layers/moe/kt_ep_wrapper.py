# SPDX-License-Identifier: Apache-2.0
"""KTransformers CPU/GPU expert parallelism for routed MoE layers.

The target keeps a configurable hot subset of native MXFP4 experts on the GPU
and executes the remaining experts with KT-Kernel.  The DSpark draft is kept
GPU-only by ``draft_worker_common.build_draft_tp_worker``.

The CPU implementation consumes the checkpoint's packed E2M1 values and UE8M0
scales directly.  It never converts the model to AMXINT4.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_compiler_backend

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)
_KT_GPU_EXPERTS_MASKS: Optional[torch.Tensor] = None
_SHARED_STAGING_BUFFER: Optional["SharedStagingBuffer"] = None


@dataclass
class KTConfig:
    layer_idx: int
    gpu_experts_mask: torch.Tensor
    cpuinfer_threads: int
    threadpool_count: int
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: Optional[int]
    method: str
    numa_nodes: Optional[list[int]] = None
    num_layers: Optional[int] = None


class SharedStagingBuffer:
    """One reusable device buffer decouples KT's D2H copy from GPU in-place MoE."""

    def __init__(
        self,
        max_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device
        self.buffer = torch.empty(
            (max_tokens, hidden_size), dtype=dtype, device=device
        )
        logger.info(
            "KT shared staging buffer: shape=%s dtype=%s device=%s size=%.1f MiB",
            tuple(self.buffer.shape),
            dtype,
            device,
            self.buffer.numel() * self.buffer.element_size() / 1024**2,
        )

    def get_slice(self, num_tokens: int) -> torch.Tensor:
        if num_tokens > self.max_tokens:
            raise RuntimeError(
                f"KT batch has {num_tokens} tokens but the shared staging buffer "
                f"was sized for {self.max_tokens}; increase --chunked-prefill-size."
            )
        return self.buffer[:num_tokens]


def _get_or_create_staging_buffer(
    *,
    max_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> SharedStagingBuffer:
    global _SHARED_STAGING_BUFFER
    if _SHARED_STAGING_BUFFER is None:
        _SHARED_STAGING_BUFFER = SharedStagingBuffer(
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
        )
    else:
        expected = (hidden_size, dtype, device)
        actual = (
            _SHARED_STAGING_BUFFER.hidden_size,
            _SHARED_STAGING_BUFFER.dtype,
            _SHARED_STAGING_BUFFER.device,
        )
        if actual != expected or _SHARED_STAGING_BUFFER.max_tokens < max_tokens:
            raise RuntimeError(
                "KT staging-buffer configuration changed after model loading: "
                f"expected={actual}, requested={expected}, "
                f"existing_max_tokens={_SHARED_STAGING_BUFFER.max_tokens}, "
                f"requested_max_tokens={max_tokens}."
            )
    return _SHARED_STAGING_BUFFER


def _unwrap_text_config(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


def _get_hf_config(server_args: "ServerArgs"):
    """Resolve the Hugging Face config through ServerArgs' public API."""
    return _unwrap_text_config(server_args.get_model_config().hf_config)


def _moe_layer_indices(hf_config) -> list[int]:
    num_layers = int(getattr(hf_config, "num_hidden_layers"))
    num_hash_layers = int(
        getattr(hf_config, "num_hash_layers", 0)
        or getattr(hf_config, "n_hash_layers", 0)
        or 0
    )
    first_moe = int(getattr(hf_config, "first_k_dense_replace", 0) or 0)
    frequency = getattr(hf_config, "moe_layer_freq", 1)

    if isinstance(frequency, (list, tuple)):
        routed = {i for i, enabled in enumerate(frequency[:num_layers]) if enabled}
    else:
        frequency = max(1, int(frequency or 1))
        routed = {
            i
            for i in range(first_moe, num_layers)
            if (i - first_moe) % frequency == 0
        }

    # DeepSeek-V4 hash layers are routed MoE layers even though inherited HF
    # configs may describe them as the dense prefix.
    routed.update(range(min(num_hash_layers, num_layers)))
    return sorted(routed)


def _load_activation_frequency(
    path: str, *, num_layers: int, num_experts: int
) -> torch.Tensor:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict):
        if "logical_count" not in loaded:
            raise ValueError(
                f"KT frequency file {path!r} has no 'logical_count' key; "
                f"keys={sorted(loaded)}"
            )
        loaded = loaded["logical_count"]
    if loaded.dim() == 3:
        loaded = loaded.sum(dim=0)
    if tuple(loaded.shape) != (num_layers, num_experts):
        raise ValueError(
            f"KT frequency tensor must have shape {(num_layers, num_experts)}, "
            f"got {tuple(loaded.shape)}"
        )
    return loaded.float().cpu()


def _build_gpu_expert_masks(server_args: "ServerArgs") -> Optional[torch.Tensor]:
    global _KT_GPU_EXPERTS_MASKS
    if _KT_GPU_EXPERTS_MASKS is not None:
        return _KT_GPU_EXPERTS_MASKS

    hf_config = _get_hf_config(server_args)
    num_layers = getattr(hf_config, "num_hidden_layers", None)
    num_experts = (
        getattr(hf_config, "n_routed_experts", None)
        or getattr(hf_config, "num_local_experts", None)
        or getattr(hf_config, "num_experts", None)
    )
    if num_layers is None or num_experts is None:
        raise ValueError(
            "KT could not determine num_hidden_layers and routed expert count "
            "from the target model config."
        )
    num_layers, num_experts = int(num_layers), int(num_experts)
    moe_layers = _moe_layer_indices(hf_config)
    if not moe_layers:
        return None

    ratio = server_args.kt_gpu_experts_ratio
    per_layer = server_args.kt_num_gpu_experts
    total_slots = len(moe_layers) * num_experts
    if ratio is not None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("--kt-gpu-experts-ratio must be in [0, 1]")
        num_gpu_experts = int(total_slots * ratio)
    else:
        per_layer = int(per_layer or 0)
        if not 0 <= per_layer <= num_experts:
            raise ValueError(
                f"--kt-num-gpu-experts must be in [0, {num_experts}]"
            )
        num_gpu_experts = per_layer * len(moe_layers)

    masks = torch.zeros((num_layers, num_experts), dtype=torch.bool, device="cpu")
    strategy = server_args.kt_expert_placement_strategy.lower()
    positions = [
        (layer_idx, expert_idx)
        for layer_idx in moe_layers
        for expert_idx in range(num_experts)
    ]

    if strategy == "frequency":
        freq_path = server_args.init_expert_location
        if not freq_path or not str(freq_path).endswith(".pt"):
            raise ValueError(
                "--kt-expert-placement-strategy frequency requires "
                "--init-expert-location pointing to a .pt expert-count file."
            )
        scores = _load_activation_frequency(
            str(freq_path), num_layers=num_layers, num_experts=num_experts
        )
        flat_scores = torch.tensor(
            [float(scores[layer_idx, expert_idx]) for layer_idx, expert_idx in positions]
        )
        selected = torch.topk(
            flat_scores,
            k=min(num_gpu_experts, len(positions)),
            largest=True,
            sorted=False,
        ).indices.tolist()
    elif strategy == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(42)
        selected = torch.randperm(len(positions), generator=generator)[
            :num_gpu_experts
        ].tolist()
    elif strategy == "front-loading":
        selected = list(range(min(num_gpu_experts, len(positions))))
    elif strategy == "uniform":
        # Round-robin by expert ordinal keeps every MoE layer near the same GPU
        # footprint while avoiding a systematic early-layer bias.
        positions = [
            (layer_idx, expert_idx)
            for expert_idx in range(num_experts)
            for layer_idx in moe_layers
        ]
        selected = list(range(min(num_gpu_experts, len(positions))))
    else:
        raise ValueError(
            "--kt-expert-placement-strategy must be one of: "
            "uniform, front-loading, random, frequency"
        )

    if get_parallel().tp_rank == 0:
        for position_idx in selected:
            layer_idx, expert_idx = positions[position_idx]
            masks[layer_idx, expert_idx] = True

    if dist.is_initialized():
        dist.broadcast(masks, src=0, group=get_tp_group().cpu_group)

    _KT_GPU_EXPERTS_MASKS = masks
    if get_parallel().tp_rank == 0:
        counts = {layer_idx: int(masks[layer_idx].sum()) for layer_idx in moe_layers}
        logger.info(
            "KT GPU placement: strategy=%s total=%d/%d per_layer=%s",
            strategy,
            int(masks[moe_layers].sum()),
            total_slots,
            counts,
        )
    return masks


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> Optional[KTConfig]:
    if server_args.kt_weight_path is None:
        return None
    if server_args.enable_eplb:
        raise ValueError(
            "KT compact GPU expert weights are incompatible with EPLB remapping. "
            "Disable --enable-eplb; frequency-based static placement remains "
            "available through --kt-expert-placement-strategy frequency."
        )

    hf_config = _get_hf_config(server_args)
    method = server_args.kt_method.upper()
    is_deepseek_v4 = bool(
        getattr(hf_config, "num_hash_layers", 0)
        or getattr(hf_config, "n_hash_layers", 0)
    )
    if is_deepseek_v4 and method != "MXFP4":
        raise ValueError(
            "DeepSeek-V4-Flash KT offload requires --kt-method MXFP4. "
            "AMXINT4 changes the model's quantization semantics and is not supported."
        )

    backend = server_args.kt_mxfp4_backend.lower()
    if backend not in ("amx", "auto", "avx2"):
        raise ValueError("--kt-mxfp4-backend must be amx, auto, or avx2")
    threshold = server_args.kt_mxfp4_amx_min_tokens_per_expert
    if not 0 <= threshold <= 1024:
        raise ValueError(
            "--kt-mxfp4-amx-min-tokens-per-expert must be in [0, 1024]"
        )
    if method == "MXFP4":
        if backend == "auto":
            os.environ.pop("KT_MXFP4_BACKEND", None)
        else:
            os.environ["KT_MXFP4_BACKEND"] = backend
        os.environ["KT_MXFP4_AMX_MIN_TOKENS_PER_EXPERT"] = str(threshold)

    masks = _build_gpu_expert_masks(server_args)
    if masks is None or layer_idx >= masks.shape[0]:
        return None
    if bool(masks[layer_idx].all()):
        # This layer is fully GPU-resident; avoid constructing a no-op KT
        # backend and preserve the GPU method's normal graph path.
        return None

    threadpool_count = int(server_args.kt_threadpool_count)
    if threadpool_count <= 0:
        raise ValueError("--kt-threadpool-count must be positive")
    cpuinfer_threads = server_args.kt_cpuinfer
    if cpuinfer_threads is None:
        cpuinfer_threads = max(threadpool_count, os.cpu_count() or 1)
    if cpuinfer_threads < threadpool_count:
        raise ValueError(
            "--kt-cpuinfer must be at least --kt-threadpool-count so every "
            "NUMA pool receives a worker."
        )
    numa_nodes = server_args.kt_numa_nodes
    if numa_nodes is not None and len(numa_nodes) != threadpool_count:
        raise ValueError(
            "--kt-numa-nodes length must equal --kt-threadpool-count"
        )

    return KTConfig(
        layer_idx=layer_idx,
        gpu_experts_mask=masks[layer_idx].clone(),
        cpuinfer_threads=int(cpuinfer_threads),
        threadpool_count=threadpool_count,
        numa_nodes=numa_nodes,
        weight_path=server_args.kt_weight_path,
        chunked_prefill_size=int(server_args.chunked_prefill_size or 8192),
        method=method,
        max_deferred_experts_per_token=server_args.kt_max_deferred_experts_per_token,
        num_layers=int(getattr(hf_config, "num_hidden_layers")),
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_and_remap_expert_ids(
    topk_ids: torch.Tensor,
    gpu_experts_mask: torch.Tensor,
    logical_to_gpu_index: torch.Tensor,
) -> torch.Tensor:
    # Dispatch padding uses -1. Index through a clamped view so padding cannot
    # alias the final logical expert before it is restored to -1.
    valid = (topk_ids >= 0) & (topk_ids < gpu_experts_mask.numel())
    safe_ids = topk_ids.clamp(0, gpu_experts_mask.numel() - 1)
    is_gpu = valid & gpu_experts_mask[safe_ids]
    return torch.where(is_gpu, logical_to_gpu_index[safe_ids], -1)


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Run a routed-expert subset on SM120 and the complement in KT-Kernel."""

    def __init__(
        self, gpu_method: FusedMoEMethodBase, kt_config: KTConfig
    ) -> None:
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is required for --kt-weight-path; install the matching "
                "KT-Kernel extension built with native MXFP4 AMX support."
            )
        if get_parallel().moe_ep_size != 1:
            raise ValueError(
                "KT native MXFP4 offload currently requires MoE EP size 1. "
                "The requested RTX PRO 6000 configuration should use --tp 1."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config
        self.gpu_experts_mask = kt_config.gpu_experts_mask.to(
            device="cpu", dtype=torch.bool
        )
        self.num_gpu_experts = int(self.gpu_experts_mask.sum())
        # A one-slot dummy keeps GPU quant-method post-processing valid for an
        # all-CPU placement. apply() never executes that slot.
        self.gpu_weight_slots = max(1, self.num_gpu_experts)
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.gpu_weight_slots
        self.tp_rank = get_parallel().tp_rank

        gpu_logical_ids = torch.where(self.gpu_experts_mask)[0]
        cpu_logical_ids = torch.where(~self.gpu_experts_mask)[0]
        self.gpu_index_to_logical = gpu_logical_ids.to(torch.int64)
        self.cpu_index_to_logical = cpu_logical_ids.to(torch.int64)
        self.num_cpu_experts = int(cpu_logical_ids.numel())
        self.logical_to_gpu_index = torch.full(
            (self.gpu_experts_mask.numel(),), -1, dtype=torch.int64
        )
        self.logical_to_gpu_index[gpu_logical_ids] = torch.arange(
            gpu_logical_ids.numel(), dtype=torch.int64
        )
        self.logical_to_cpu_index = torch.full(
            (self.gpu_experts_mask.numel(),), -1, dtype=torch.int64
        )
        self.logical_to_cpu_index[cpu_logical_ids] = torch.arange(
            cpu_logical_ids.numel(), dtype=torch.int64
        )

        self.gpu_experts_mask_cuda: Optional[torch.Tensor] = None
        self.logical_to_gpu_index_cuda: Optional[torch.Tensor] = None
        self.cpu_experts_mask_cuda: Optional[torch.Tensor] = None
        self.logical_to_cpu_index_cuda: Optional[torch.Tensor] = None
        self.wrapper: Optional[KTMoEWrapper] = None
        self._cpu_stream: Optional[torch.cuda.Stream] = None
        self._cpu_done_event: Optional[torch.cuda.Event] = None
        self._staging: Optional[SharedStagingBuffer] = None

    def gpu_weight_index(self, logical_expert_id: int) -> Optional[int]:
        if logical_expert_id < 0 or logical_expert_id >= self.logical_to_gpu_index.numel():
            return None
        index = int(self.logical_to_gpu_index[logical_expert_id])
        return index if index >= 0 else None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self.global_num_experts = num_experts
        if num_experts != self.gpu_experts_mask.numel():
            raise ValueError(
                "KT offload currently supports routed experts only; the MoE layer "
                f"has {num_experts} weight slots but the routed-expert mask has "
                f"{self.gpu_experts_mask.numel()}. Disable shared-expert fusion."
            )
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.gpu_weight_slots,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        target_device = next(layer.parameters()).device
        self.gpu_experts_mask_cuda = self.gpu_experts_mask.to(target_device)
        self.logical_to_gpu_index_cuda = self.logical_to_gpu_index.to(target_device)
        self.cpu_experts_mask_cuda = (~self.gpu_experts_mask).to(target_device)
        self.logical_to_cpu_index_cuda = self.logical_to_cpu_index.to(target_device)
        if self.tp_rank == 0:
            self._cpu_stream = torch.cuda.Stream(device=target_device)
            self._cpu_done_event = torch.cuda.Event()
            self._staging = _get_or_create_staging_buffer(
                max_tokens=self.kt_config.chunked_prefill_size,
                hidden_size=hidden_size,
                dtype=params_dtype,
                device=target_device,
            )

            runner_config = layer.moe_runner_config
            swiglu_limit = float(runner_config.swiglu_limit or 0.0)
            gemm1_limit = float(runner_config.gemm1_clamp_limit or 0.0)
            if gemm1_limit:
                swiglu_limit = gemm1_limit
            swiglu_alpha = float(runner_config.gemm1_alpha or 0.0)
            layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
            if (
                self.kt_config.num_layers is not None
                and self.kt_config.layer_idx == self.kt_config.num_layers - 1
            ):
                layer_max_deferred = 0

            self.wrapper = KTMoEWrapper(
                layer_idx=self.kt_config.layer_idx,
                # KT uses a compact CPU-only expert space. This avoids
                # allocating or copying native MXFP4 buffers for experts that
                # are already resident on the GPU.
                num_experts=self.num_cpu_experts,
                num_experts_per_tok=layer.top_k,
                hidden_size=hidden_size,
                moe_intermediate_size=(
                    layer.intermediate_size_per_partition * layer.moe_tp_size
                ),
                gpu_experts_mask=torch.zeros(
                    self.num_cpu_experts, dtype=torch.bool, device="cpu"
                ),
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                numa_nodes=self.kt_config.numa_nodes,
                weight_path=self.kt_config.weight_path,
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
                method=self.kt_config.method,
                swiglu_limit=swiglu_limit,
                swiglu_alpha=swiglu_alpha,
                max_deferred_experts_per_token=layer_max_deferred,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)

        if self.tp_rank == 0 and self.wrapper is not None:
            torch.cuda.synchronize()
            # Compact KT expert index -> checkpoint logical expert index.
            # EPLB is rejected during config creation, so this mapping remains
            # static for the lifetime of the server.
            self.wrapper.load_weights(self.cpu_index_to_logical.contiguous())

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ) -> None:
        self.moe_runner_config = moe_runner_config
        gpu_config = replace(
            moe_runner_config, num_local_experts=self.gpu_weight_slots
        )
        self.gpu_method.create_moe_runner(layer, gpu_config)

    def _apply_impl(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output
        staged = None

        if self.tp_rank == 0:
            assert self.wrapper is not None
            assert self._cpu_stream is not None
            assert self._staging is not None
            assert self.cpu_experts_mask_cuda is not None
            assert self.logical_to_cpu_index_cuda is not None
            staged = self._staging.get_slice(x.reshape(-1, x.shape[-1]).shape[0])
            staged.copy_(x.reshape_as(staged), non_blocking=True)
            cpu_topk_ids = mask_and_remap_expert_ids(
                topk_ids,
                self.cpu_experts_mask_cuda,
                self.logical_to_cpu_index_cuda,
            )
            self._cpu_stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(self._cpu_stream):
                self.wrapper.submit_forward(
                    staged,
                    cpu_topk_ids,
                    topk_weights,
                    self._cpu_stream.cuda_stream,
                )

        if self.num_gpu_experts > 0:
            assert self.gpu_experts_mask_cuda is not None
            assert self.logical_to_gpu_index_cuda is not None
            remapped_ids = mask_and_remap_expert_ids(
                topk_ids,
                self.gpu_experts_mask_cuda,
                self.logical_to_gpu_index_cuda,
            )
            gpu_topk_output = topk_output._replace(topk_ids=remapped_ids)
            gpu_dispatch_output = dispatch_output._replace(
                topk_output=gpu_topk_output
            )
            gpu_result = self.gpu_method.apply(layer, gpu_dispatch_output)
            output = gpu_result.hidden_states
        else:
            output = torch.zeros_like(x)

        if self.tp_rank == 0:
            assert staged is not None
            assert self._cpu_stream is not None
            assert self._cpu_done_event is not None
            with torch.cuda.stream(self._cpu_stream):
                cpu_output = self.wrapper.sync_forward(
                    staged, self._cpu_stream.cuda_stream
                )
                self._cpu_done_event.record(self._cpu_stream)
            torch.cuda.current_stream(x.device).wait_event(self._cpu_done_event)
            output = output + cpu_output.reshape_as(output)

        return StandardCombineInput(hidden_states=output)

    def _apply_capture_stub(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Allocate the static bridge output without invoking KT during capture."""
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        del layer
        return StandardCombineInput(
            hidden_states=torch.zeros_like(dispatch_output.hidden_states)
        )

    # KT submits host work and CUDA copies at replay time. A breakable graph
    # captures the GPU segments around this eager node while retaining their
    # launch-overhead benefit; a monolithic/full CUDA graph cannot represent it.
    apply = eager_on_graph(True, capture_stub=_apply_capture_stub)(_apply_impl)

    def __getattr__(self, name: str):
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(name)
        return getattr(self.gpu_method, name)
