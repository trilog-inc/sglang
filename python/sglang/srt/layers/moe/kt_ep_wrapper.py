# SPDX-License-Identifier: Apache-2.0
"""KTransformers CPU/GPU expert parallelism for routed MoE layers.

The target keeps a configurable hot subset of native MXFP4 experts on the GPU
and executes the remaining experts with KT-Kernel.  The DSpark draft is kept
GPU-only by ``draft_worker_common.build_draft_tp_worker``.

The CPU implementation consumes the checkpoint's packed E2M1 values and UE8M0
scales directly.  It never converts the model to AMXINT4.
"""

from __future__ import annotations

import gc
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
_MXFP4_PREFILL_LAYER_REGISTRY: dict[tuple, dict[int, tuple]] = {}
_MXFP4_LAYERWISE_MANAGERS: dict[tuple, "_Mxfp4LayerwisePrefillManager"] = {}
_MXFP4_LAYERWISE_DISABLED_REASONS: dict[tuple, str] = {}


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
    gpu_prefill_token_threshold: int = 0
    mxfp4_prefill_slots: str = "auto"
    mxfp4_prefill_host_staging_experts: int = 8
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
        self.buffer = torch.empty((max_tokens, hidden_size), dtype=dtype, device=device)
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


class KTGraphStateBridge:
    """Persistent buffers for tensors that span a KT breakable-graph seam.

    A CUDA graph records raw addresses.  DSV4 keeps several layer-local tensors
    alive across its MoE call, but those tensors are not arguments to KT's
    ``eager_on_graph`` break and therefore are invisible to the generic bridge.
    Copying them into these owner-retained buffers before the break gives both
    adjacent graph segments stable addresses.  Buffers are shared by all graph
    batch shapes for one layer and sized to the largest shape observed.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, torch.Tensor] = {}
        # Capture sizes are normally visited largest-first.  Keep an old buffer
        # alive if that invariant ever changes, because an already captured graph
        # may still contain its address.
        self._retired_buffers: list[torch.Tensor] = []

    def copy(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            raise ValueError(f"KT graph bridge {name!r} requires a row dimension")

        buffer = self._buffers.get(name)
        compatible = (
            buffer is not None
            and buffer.device == tensor.device
            and buffer.dtype == tensor.dtype
            and buffer.shape[1:] == tensor.shape[1:]
            and buffer.shape[0] >= tensor.shape[0]
        )
        if not compatible:
            if buffer is not None:
                self._retired_buffers.append(buffer)
            buffer = tensor.new_empty(tensor.shape)
            self._buffers[name] = buffer

        view = buffer[: tensor.shape[0]]
        view.copy_(tensor)
        return view


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
            i for i in range(first_moe, num_layers) if (i - first_moe) % frequency == 0
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
    if not isinstance(loaded, torch.Tensor):
        raise ValueError(
            f"KT frequency data in {path!r} must be a tensor, "
            f"got {type(loaded).__name__}"
        )
    if loaded.dim() == 3:
        loaded = loaded.sum(dim=0)
    if tuple(loaded.shape) != (num_layers, num_experts):
        raise ValueError(
            f"KT frequency tensor must have shape {(num_layers, num_experts)}, "
            f"got {tuple(loaded.shape)}"
        )
    loaded = loaded.to(dtype=torch.float64, device="cpu")
    if not torch.isfinite(loaded).all():
        raise ValueError(f"KT frequency tensor in {path!r} contains non-finite values")
    if (loaded < 0).any():
        raise ValueError(f"KT frequency tensor in {path!r} contains negative counts")
    return loaded


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
            raise ValueError(f"--kt-num-gpu-experts must be in [0, {num_experts}]")
        num_gpu_experts = per_layer * len(moe_layers)

    # Preserve the requested total memory budget while ensuring latency-sensitive
    # frequency placement has a GPU allocation in every routed layer.  The
    # remainder only applies to ratio-based placement because the explicit count
    # is already an exact per-layer value.
    base_per_layer, remainder = divmod(num_gpu_experts, len(moe_layers))
    per_layer_targets = {
        layer_idx: base_per_layer + (offset < remainder)
        for offset, layer_idx in enumerate(moe_layers)
    }

    masks = torch.zeros((num_layers, num_experts), dtype=torch.bool, device="cpu")
    strategy = server_args.kt_expert_placement_strategy.lower()
    positions = [
        (layer_idx, expert_idx)
        for layer_idx in moe_layers
        for expert_idx in range(num_experts)
    ]

    if strategy == "frequency":
        freq_path = server_args.kt_expert_frequency_file
        if not freq_path:
            raise ValueError(
                "--kt-expert-placement-strategy frequency requires "
                "--kt-expert-frequency-file pointing to an "
                "ExpertDistributionRecorder .pt file."
            )
        scores = _load_activation_frequency(
            str(freq_path), num_layers=num_layers, num_experts=num_experts
        )
        empty_layers = [
            layer_idx for layer_idx in moe_layers if float(scores[layer_idx].sum()) <= 0
        ]
        if empty_layers:
            raise ValueError(
                "KT frequency profile has no recorded routes for MoE layers "
                f"{empty_layers}. Recapture a complete target-model profile."
            )

        if get_parallel().tp_rank == 0:
            for layer_idx in moe_layers:
                target = per_layer_targets[layer_idx]
                if target:
                    # Stable sorting makes zero-count/tied experts reproducible
                    # and favors lower logical ids only as a deterministic tie-break.
                    expert_ids = torch.argsort(
                        scores[layer_idx], descending=True, stable=True
                    )[:target]
                    masks[layer_idx, expert_ids] = True
        selected = None
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

    if get_parallel().tp_rank == 0 and selected is not None:
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
        if strategy == "frequency":
            routed = scores[moe_layers].sum()
            selected_routes = scores[masks].sum()
            logger.info(
                "KT frequency profile: file=%s selected_routes=%.0f/%.0f "
                "coverage=%.2f%%",
                freq_path,
                float(selected_routes),
                float(routed),
                100.0 * float(selected_routes / routed),
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
    if server_args.init_expert_location != "trivial":
        raise ValueError(
            "KT compact GPU expert weights require --init-expert-location trivial. "
            "Use --kt-expert-frequency-file to load recorder counts without "
            "remapping logical expert ids."
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
        raise ValueError("--kt-mxfp4-amx-min-tokens-per-expert must be in [0, 1024]")
    if method == "MXFP4":
        if backend == "auto":
            os.environ.pop("KT_MXFP4_BACKEND", None)
        else:
            os.environ["KT_MXFP4_BACKEND"] = backend
        os.environ["KT_MXFP4_AMX_MIN_TOKENS_PER_EXPERT"] = str(threshold)

    gpu_prefill_threshold = int(
        getattr(server_args, "kt_gpu_prefill_token_threshold", 0)
    )
    if gpu_prefill_threshold < 0:
        raise ValueError("--kt-gpu-prefill-token-threshold must be non-negative")
    if gpu_prefill_threshold > 0 and get_parallel().tp_size != 1:
        raise ValueError("KT native-MXFP4 layerwise prefill currently requires --tp 1")
    prefill_slots = str(getattr(server_args, "kt_mxfp4_prefill_slots", "auto")).lower()
    if prefill_slots not in ("auto", "1", "2"):
        raise ValueError("--kt-mxfp4-prefill-slots must be auto, 1, or 2")
    host_staging_experts = int(
        getattr(server_args, "kt_mxfp4_prefill_host_staging_experts", 8)
    )
    if not 2 <= host_staging_experts <= 64:
        raise ValueError("--kt-mxfp4-prefill-host-staging-experts must be in [2, 64]")

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
        raise ValueError("--kt-numa-nodes length must equal --kt-threadpool-count")

    return KTConfig(
        layer_idx=layer_idx,
        gpu_experts_mask=masks[layer_idx].clone(),
        cpuinfer_threads=int(cpuinfer_threads),
        threadpool_count=threadpool_count,
        numa_nodes=numa_nodes,
        weight_path=server_args.kt_weight_path,
        chunked_prefill_size=int(server_args.chunked_prefill_size or 8192),
        method=method,
        gpu_prefill_token_threshold=gpu_prefill_threshold,
        mxfp4_prefill_slots=prefill_slots,
        mxfp4_prefill_host_staging_experts=host_staging_experts,
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


@torch.compile(
    dynamic=True,
    backend=get_compiler_backend(),
    # This helper runs inside an outer breakable CUDA graph and one output is
    # consumed on a side stream.  An inner Inductor CUDA graph may recycle its
    # static outputs on the next call even while the side-stream copy is live.
    options={"triton.cudagraphs": False},
)
def partition_and_remap_expert_ids(
    topk_ids: torch.Tensor,
    gpu_experts_mask: torch.Tensor,
    logical_to_gpu_index: torch.Tensor,
    logical_to_cpu_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition logical routing IDs into stable CPU and GPU outputs.

    Producing both tensors in one compiled invocation is important for hybrid
    replay.  The CPU result is consumed asynchronously on ``_cpu_stream``; a
    second invocation of the same compiled function on the main stream may
    reuse its output storage before that copy completes.
    """
    valid = (topk_ids >= 0) & (topk_ids < gpu_experts_mask.numel())
    safe_ids = topk_ids.clamp(0, gpu_experts_mask.numel() - 1)
    on_gpu = valid & gpu_experts_mask[safe_ids]
    gpu_ids = torch.where(on_gpu, logical_to_gpu_index[safe_ids], -1)
    cpu_ids = torch.where(valid & ~on_gpu, logical_to_cpu_index[safe_ids], -1)
    return cpu_ids, gpu_ids


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Run a routed-expert subset on SM120 and the complement in KT-Kernel."""

    def __init__(self, gpu_method: FusedMoEMethodBase, kt_config: KTConfig) -> None:
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
        # Some compact GPU quantization methods load global expert metadata
        # (for example NVFP4 tensor scales) and must slice it in the same order
        # as the resident weight slots. The all-CPU placement still needs one
        # internally consistent dummy slot for GPU-method post-processing.
        self.gpu_method.gpu_index_to_logical = (
            self.gpu_index_to_logical
            if self.num_gpu_experts > 0
            else torch.zeros(1, dtype=torch.int64)
        )
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
        self.graph_state_bridge = KTGraphStateBridge()
        self._mxfp4_pipeline_signature: Optional[tuple] = None

    def bridge_cuda_graph_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        return self.graph_state_bridge.copy(name, tensor)

    def gpu_weight_index(self, logical_expert_id: int) -> Optional[int]:
        if (
            logical_expert_id < 0
            or logical_expert_id >= self.logical_to_gpu_index.numel()
        ):
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
        if self.kt_config.gpu_prefill_token_threshold > 0:
            # FlashInfer SM120 interleaves E8M0 scales in-place. Marlin needs
            # the checkpoint order, so retain only the compact resident scale
            # payload in ordinary host memory; keeping these copies on every
            # layer's GPU would consume several GiB. FP4 weights themselves
            # remain in native layout.
            layer._kt_mxfp4_raw_w13_scale_inv = layer.w13_weight_scale_inv.detach().to(
                device="cpu", copy=True
            )
            layer._kt_mxfp4_raw_w2_scale_inv = layer.w2_weight_scale_inv.detach().to(
                device="cpu", copy=True
            )

        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)

        if self.tp_rank == 0 and self.wrapper is not None:
            torch.cuda.synchronize()
            # Compact KT expert index -> checkpoint logical expert index.
            # EPLB is rejected during config creation, so this mapping remains
            # static for the lifetime of the server.
            self.wrapper.load_weights(self.cpu_index_to_logical.contiguous())

        _register_mxfp4_prefill_layer(self, layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ) -> None:
        self.moe_runner_config = moe_runner_config
        gpu_config = replace(moe_runner_config, num_local_experts=self.gpu_weight_slots)
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

        if (
            self.kt_config.gpu_prefill_token_threshold > 0
            and x.shape[0] >= self.kt_config.gpu_prefill_token_threshold
            and self._mxfp4_pipeline_signature is not None
        ):
            manager = _MXFP4_LAYERWISE_MANAGERS.get(self._mxfp4_pipeline_signature)
            if manager is not None:
                return manager.apply(self, layer, dispatch_output)

        assert self.gpu_experts_mask_cuda is not None
        assert self.logical_to_gpu_index_cuda is not None
        assert self.logical_to_cpu_index_cuda is not None
        cpu_topk_ids, gpu_topk_ids = partition_and_remap_expert_ids(
            topk_ids,
            self.gpu_experts_mask_cuda,
            self.logical_to_gpu_index_cuda,
            self.logical_to_cpu_index_cuda,
        )

        if self.tp_rank == 0:
            assert self.wrapper is not None
            assert self._cpu_stream is not None
            assert self._staging is not None
            staged = self._staging.get_slice(x.reshape(-1, x.shape[-1]).shape[0])
            staged.copy_(x.reshape_as(staged), non_blocking=True)
            self._cpu_stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(self._cpu_stream):
                self.wrapper.submit_forward(
                    staged,
                    cpu_topk_ids,
                    topk_weights,
                    self._cpu_stream.cuda_stream,
                )

        if self.num_gpu_experts > 0:
            gpu_topk_output = topk_output._replace(topk_ids=gpu_topk_ids)
            gpu_dispatch_output = dispatch_output._replace(topk_output=gpu_topk_output)
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


class _Mxfp4PrefillSlot:
    def __init__(self, index: int, prepared) -> None:
        self.index = index
        self.prepared = prepared
        self.layer_idx: Optional[int] = None
        self.epoch = -1
        self.state = "EMPTY"
        self.ready_event = torch.cuda.Event()
        self.consumed_event = torch.cuda.Event()
        self.has_consumed_event = False


class _Mxfp4LayerwisePrefillManager:
    """Stream one native-MXFP4 expert at a time into prepared Marlin slots."""

    _RAW_NAMES = (
        "w13_weight",
        "w13_weight_scale_inv",
        "w2_weight",
        "w2_weight_scale_inv",
    )

    def __init__(self, signature: tuple, first_method, first_layer, slot_count: int):
        from sglang.srt.layers.quantization.v4_marlin_moe import (
            V4MarlinPreparedWeights,
            allocate_v4_mxfp4_marlin,
        )

        self.signature = signature
        self.device = first_layer.w13_weight.device
        self.num_experts = first_method.global_num_experts
        self.hidden_size = int(first_layer.w13_weight.shape[2]) * 2
        self.intermediate_size = int(first_layer.w2_weight.shape[2]) * 2
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self.V4MarlinPreparedWeights = V4MarlinPreparedWeights
        self.slots = [
            _Mxfp4PrefillSlot(
                index,
                allocate_v4_mxfp4_marlin(
                    num_experts=self.num_experts,
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    device=self.device,
                ),
            )
            for index in range(slot_count)
        ]

        expert_shapes = {
            "w13_weight": tuple(first_layer.w13_weight.shape[1:]),
            "w13_weight_scale_inv": tuple(
                first_layer._kt_mxfp4_raw_w13_scale_inv.shape[1:]
            ),
            "w2_weight": tuple(first_layer.w2_weight.shape[1:]),
            "w2_weight_scale_inv": tuple(
                first_layer._kt_mxfp4_raw_w2_scale_inv.shape[1:]
            ),
        }
        raw_dtypes = {
            "w13_weight": first_layer.w13_weight.dtype,
            "w13_weight_scale_inv": torch.float8_e8m0fnu,
            "w2_weight": first_layer.w2_weight.dtype,
            "w2_weight_scale_inv": torch.float8_e8m0fnu,
        }
        host_slots = first_method.kt_config.mxfp4_prefill_host_staging_experts
        # One stream owns this reusable raw window. Batching preparation cuts
        # four repack/swizzle launches per expert down to four per window while
        # keeping raw storage far below a complete DSV4 layer image.
        self.raw_batch_experts = min(host_slots, 16)
        self.raw_staging = {
            name: torch.empty(
                (self.raw_batch_experts, *shape),
                dtype=raw_dtypes[name],
                device=self.device,
            )
            for name, shape in expert_shapes.items()
        }

        host_dtypes = dict(raw_dtypes)
        host_dtypes["w13_weight_scale_inv"] = torch.bfloat16
        host_dtypes["w2_weight_scale_inv"] = torch.bfloat16
        try:
            self.host_staging = {
                name: torch.empty(
                    (host_slots, *shape),
                    dtype=host_dtypes[name],
                    device="cpu",
                    pin_memory=True,
                )
                for name, shape in expert_shapes.items()
            }
            self.gpu_scale_staging = {
                "w13_weight_scale_inv": torch.empty(
                    (self.raw_batch_experts, *expert_shapes["w13_weight_scale_inv"]),
                    dtype=torch.float8_e8m0fnu,
                    device="cpu",
                    pin_memory=True,
                ),
                "w2_weight_scale_inv": torch.empty(
                    (self.raw_batch_experts, *expert_shapes["w2_weight_scale_inv"]),
                    dtype=torch.float8_e8m0fnu,
                    device="cpu",
                    pin_memory=True,
                ),
            }
            self.host_is_pinned = True
        except RuntimeError:
            gc.collect()
            self.host_staging = {
                name: torch.empty(
                    (host_slots, *shape), dtype=host_dtypes[name], device="cpu"
                )
                for name, shape in expert_shapes.items()
            }
            self.gpu_scale_staging = {
                name: torch.empty(
                    (self.raw_batch_experts, *expert_shapes[name]),
                    dtype=torch.float8_e8m0fnu,
                    device="cpu",
                )
                for name in ("w13_weight_scale_inv", "w2_weight_scale_inv")
            }
            self.host_is_pinned = False
            logger.warning(
                "KT MXFP4 prefill could not allocate pinned host staging; "
                "falling back to synchronous pageable transfers"
            )
        self.host_slots = host_slots
        self.host_free_events = [torch.cuda.Event() for _ in range(host_slots)]
        self.host_slot_used = [False] * host_slots
        self.gpu_scale_free_events = [
            torch.cuda.Event() for _ in range(self.raw_batch_experts)
        ]
        self.gpu_scale_slot_used = [False] * self.raw_batch_experts
        self.current_slot_index: Optional[int] = None
        self.epoch = -1
        self.last_position: Optional[int] = None

    @property
    def registry(self):
        return _MXFP4_PREFILL_LAYER_REGISTRY[self.signature]

    @property
    def layer_order(self) -> list[int]:
        return sorted(self.registry)

    def _prepared_range_view(self, slot: _Mxfp4PrefillSlot, start: int, count: int):
        prepared = slot.prepared
        return self.V4MarlinPreparedWeights(
            w13=prepared.w13[start : start + count],
            w13_scale=prepared.w13_scale[start : start + count],
            w2=prepared.w2[start : start + count],
            w2_scale=prepared.w2_scale[start : start + count],
            hidden_size=prepared.hidden_size,
            intermediate_size=prepared.intermediate_size,
            num_experts=count,
        )

    def _submit_cpu_expert(self, method, logical_id: int, host_slot: int) -> None:
        if method.wrapper is None:
            raise RuntimeError("KT MXFP4 prefill has no CPU weight writer")
        cpu_id = int(method.logical_to_cpu_index[logical_id])
        if cpu_id < 0:
            raise RuntimeError(f"logical expert {logical_id} is not CPU resident")
        pointers = {
            name: [int(self.host_staging[name][host_slot].data_ptr())]
            for name in self._RAW_NAMES
        }
        method.wrapper.submit_write_weight_scale_to_buffer(
            1,
            cpu_id,
            pointers["w13_weight"],
            pointers["w13_weight_scale_inv"],
            pointers["w2_weight"],
            pointers["w2_weight_scale_inv"],
        )
        method.wrapper.sync_write_weight_scale_to_buffer()

    def _stage_gpu_expert(self, method, layer, logical_id: int, raw_row: int) -> None:
        gpu_id = int(method.logical_to_gpu_index[logical_id])
        if gpu_id < 0:
            raise RuntimeError(f"logical expert {logical_id} is not GPU resident")
        intermediate = self.intermediate_size
        dst_w13 = self.raw_staging["w13_weight"][raw_row]
        src_w13 = layer.w13_weight[gpu_id]
        # FlashInfer loads compact resident W13 as [up; gate]; Marlin and the
        # KT writer use [gate; up].
        dst_w13[:intermediate].copy_(src_w13[intermediate:], non_blocking=True)
        dst_w13[intermediate:].copy_(src_w13[:intermediate], non_blocking=True)
        dst_s13 = self.raw_staging["w13_weight_scale_inv"][raw_row]
        src_s13 = self.gpu_scale_staging["w13_weight_scale_inv"][raw_row]
        dst_s13[:intermediate].copy_(
            src_s13[intermediate:], non_blocking=self.host_is_pinned
        )
        dst_s13[intermediate:].copy_(
            src_s13[:intermediate], non_blocking=self.host_is_pinned
        )
        self.raw_staging["w2_weight"][raw_row].copy_(
            layer.w2_weight[gpu_id], non_blocking=True
        )
        self.raw_staging["w2_weight_scale_inv"][raw_row].copy_(
            self.gpu_scale_staging["w2_weight_scale_inv"][raw_row],
            non_blocking=self.host_is_pinned,
        )
        self.gpu_scale_free_events[raw_row].record(self.transfer_stream)
        self.gpu_scale_slot_used[raw_row] = True

    def _stage_gpu_scales_on_host(
        self, method, layer, logical_id: int, raw_row: int
    ) -> None:
        gpu_id = int(method.logical_to_gpu_index[logical_id])
        if self.gpu_scale_slot_used[raw_row]:
            self.gpu_scale_free_events[raw_row].synchronize()
        self.gpu_scale_staging["w13_weight_scale_inv"][raw_row].copy_(
            layer._kt_mxfp4_raw_w13_scale_inv[gpu_id]
        )
        self.gpu_scale_staging["w2_weight_scale_inv"][raw_row].copy_(
            layer._kt_mxfp4_raw_w2_scale_inv[gpu_id]
        )

    def _stage_cpu_expert(self, host_slot: int, raw_row: int) -> None:
        for name in self._RAW_NAMES:
            self.raw_staging[name][raw_row].copy_(
                self.host_staging[name][host_slot],
                non_blocking=self.host_is_pinned,
            )
        self.host_free_events[host_slot].record(self.transfer_stream)
        self.host_slot_used[host_slot] = True

    def _prepare_raw_batch(
        self, slot: _Mxfp4PrefillSlot, start: int, count: int
    ) -> None:
        from sglang.srt.layers.quantization.v4_marlin_moe import (
            prepare_v4_mxfp4_marlin,
        )

        prepare_v4_mxfp4_marlin(
            self.raw_staging["w13_weight"][:count],
            self.raw_staging["w13_weight_scale_inv"][:count],
            self.raw_staging["w2_weight"][:count],
            self.raw_staging["w2_weight_scale_inv"][:count],
            out=self._prepared_range_view(slot, start, count),
        )

    def _load_slot(self, slot: _Mxfp4PrefillSlot, layer_idx: int, method, layer):
        slot.state = "LOADING"
        slot.layer_idx = layer_idx
        slot.epoch = self.epoch
        with torch.cuda.stream(self.transfer_stream):
            if slot.has_consumed_event:
                self.transfer_stream.wait_event(slot.consumed_event)

        cpu_position = 0
        for start in range(0, self.num_experts, self.raw_batch_experts):
            count = min(self.raw_batch_experts, self.num_experts - start)
            for raw_row, logical_id in enumerate(range(start, start + count)):
                if bool(method.gpu_experts_mask[logical_id]):
                    self._stage_gpu_scales_on_host(method, layer, logical_id, raw_row)
                    with torch.cuda.stream(self.transfer_stream):
                        self._stage_gpu_expert(method, layer, logical_id, raw_row)
                    continue

                host_slot = cpu_position % self.host_slots
                if self.host_slot_used[host_slot]:
                    self.host_free_events[host_slot].synchronize()
                self._submit_cpu_expert(method, logical_id, host_slot)
                with torch.cuda.stream(self.transfer_stream):
                    self._stage_cpu_expert(host_slot, raw_row)
                cpu_position += 1

            with torch.cuda.stream(self.transfer_stream):
                self._prepare_raw_batch(slot, start, count)

        with torch.cuda.stream(self.transfer_stream):
            slot.ready_event.record(self.transfer_stream)
        slot.state = "READY"

    def _advance_round(self, layer_idx: int) -> None:
        position = self.layer_order.index(layer_idx)
        if self.last_position is None or position <= self.last_position:
            self.epoch += 1
        self.last_position = position

    def _acquire(self, layer_idx: int, method, layer):
        self._advance_round(layer_idx)
        for slot in self.slots:
            if (
                slot.state == "READY"
                and slot.layer_idx == layer_idx
                and slot.epoch == self.epoch
            ):
                return slot, True
        index = (
            0
            if self.current_slot_index is None
            else (self.current_slot_index + 1) % len(self.slots)
        )
        slot = self.slots[index]
        self._load_slot(slot, layer_idx, method, layer)
        return slot, False

    def _prefetch_successor(self, current: _Mxfp4PrefillSlot) -> None:
        order = self.layer_order
        position = order.index(current.layer_idx)
        if position + 1 >= len(order):
            return
        successor = order[position + 1]
        method, layer = self.registry[successor]
        target = self.slots[(current.index + 1) % len(self.slots)]
        if (
            target.state == "READY"
            and target.layer_idx == successor
            and target.epoch == self.epoch
        ):
            return
        self._load_slot(target, successor, method, layer)

    def apply(self, method, layer, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.quantization.v4_marlin_moe import apply_v4_marlin_moe

        layer_idx = method.kt_config.layer_idx
        slot, prefetch_hit = self._acquire(layer_idx, method, layer)
        main_stream = torch.cuda.current_stream(self.device)
        main_stream.wait_event(slot.ready_event)
        for tensor in (
            slot.prepared.w13,
            slot.prepared.w13_scale,
            slot.prepared.w2,
            slot.prepared.w2_scale,
        ):
            tensor.record_stream(main_stream)

        topk_weights, topk_ids, _ = dispatch_output.topk_output
        output = apply_v4_marlin_moe(
            hidden_states=dispatch_output.hidden_states,
            prepared=slot.prepared,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            # DSV4 applies routed_scaling_factor after the experts because the
            # KT wrapper is not marked as a fused-RSF backend.
            routed_scaling_factor=1.0,
            swiglu_limit=method.moe_runner_config.swiglu_limit,
        )
        slot.consumed_event.record(main_stream)
        slot.has_consumed_event = True
        slot.state = "IN_USE"
        self.current_slot_index = slot.index
        self._prefetch_successor(slot)
        if method.tp_rank == 0 and method.kt_config.layer_idx == self.layer_order[0]:
            logger.info(
                "KT MXFP4 layerwise prefill epoch=%d slot=%d %s tokens=%d",
                self.epoch,
                slot.index,
                "prefetch-hit" if prefetch_hit else "prime",
                dispatch_output.hidden_states.shape[0],
            )
        return StandardCombineInput(hidden_states=output)


def _mxfp4_pipeline_signature(method, layer) -> tuple:
    return (
        str(layer.w13_weight.device),
        method.kt_config.weight_path,
        method.kt_config.num_layers,
        method.global_num_experts,
        tuple(layer.w13_weight.shape[1:]),
        tuple(layer.w2_weight.shape[1:]),
    )


def _register_mxfp4_prefill_layer(method, layer) -> None:
    if method.kt_config.gpu_prefill_token_threshold <= 0:
        return
    if method.kt_config.method.upper() != "MXFP4":
        return
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(
        layer.w13_weight.device
    ) != (12, 0):
        raise RuntimeError("KT native-MXFP4 layerwise prefill currently requires SM120")
    if method.gpu_method.__class__.__name__ != "Mxfp4FlashinferCutlassMoEMethod":
        raise RuntimeError(
            "KT native-MXFP4 layerwise prefill requires the FlashInfer MXFP4 "
            "GPU method on SM120"
        )
    signature = _mxfp4_pipeline_signature(method, layer)
    method._mxfp4_pipeline_signature = signature
    registry = _MXFP4_PREFILL_LAYER_REGISTRY.setdefault(signature, {})
    registry[method.kt_config.layer_idx] = (method, layer)


def finalize_mxfp4_layerwise_prefill() -> None:
    """Reserve prepared slots before SGLang sizes the KV cache."""
    for signature, registry in list(_MXFP4_PREFILL_LAYER_REGISTRY.items()):
        if not registry or signature in _MXFP4_LAYERWISE_MANAGERS:
            continue
        first_idx = min(registry)
        method, layer = registry[first_idx]
        policy = method.kt_config.mxfp4_prefill_slots
        requested = 2 if policy in ("auto", "2") else 1
        manager = None
        last_error_message: Optional[str] = None
        for slot_count in range(requested, 0, -1):
            if policy != "auto" and slot_count != requested:
                break
            try:
                manager = _Mxfp4LayerwisePrefillManager(
                    signature, method, layer, slot_count
                )
                break
            except torch.cuda.OutOfMemoryError as exc:
                # Do not retain the exception traceback: it owns the partially
                # constructed manager and would keep an already-allocated slot
                # alive while the auto policy attempts its one-slot fallback.
                last_error_message = str(exc)
                manager = None
                gc.collect()
                torch.cuda.empty_cache()
                logger.warning(
                    "KT MXFP4 prefill could not allocate %d prepared slot(s); %s",
                    slot_count,
                    "trying one slot" if policy == "auto" and slot_count == 2 else "",
                )

        if manager is None:
            reason = "prepared Marlin slot allocation failed"
            if policy == "auto":
                _MXFP4_LAYERWISE_DISABLED_REASONS[signature] = reason
                for _, registered_layer in registry.values():
                    for name in (
                        "_kt_mxfp4_raw_w13_scale_inv",
                        "_kt_mxfp4_raw_w2_scale_inv",
                    ):
                        if hasattr(registered_layer, name):
                            delattr(registered_layer, name)
                gc.collect()
                logger.warning(
                    "KT MXFP4 layerwise prefill disabled; using hybrid MoE: %s",
                    reason,
                )
                continue
            if last_error_message:
                reason = f"{reason}: {last_error_message}"
            raise RuntimeError(reason)

        _MXFP4_LAYERWISE_MANAGERS[signature] = manager
        prepared_bytes = sum(
            tensor.numel() * tensor.element_size()
            for slot in manager.slots
            for tensor in (
                slot.prepared.w13,
                slot.prepared.w13_scale,
                slot.prepared.w2,
                slot.prepared.w2_scale,
            )
        )
        raw_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in manager.raw_staging.values()
        )
        preserved_scale_bytes = sum(
            getattr(registered_layer, name).numel()
            * getattr(registered_layer, name).element_size()
            for _, registered_layer in registry.values()
            for name in (
                "_kt_mxfp4_raw_w13_scale_inv",
                "_kt_mxfp4_raw_w2_scale_inv",
            )
        )
        logger.info(
            "KT MXFP4 layerwise prefill initialized %d prepared slot(s), "
            "%d %s host stages, raw batch=%d experts, GPU prepared/raw/total="
            "%.2f/%.2f/%.2f GiB, preserved host scales=%.2f GiB",
            len(manager.slots),
            manager.host_slots,
            "pinned" if manager.host_is_pinned else "pageable",
            manager.raw_batch_experts,
            prepared_bytes / 1024**3,
            raw_bytes / 1024**3,
            (prepared_bytes + raw_bytes) / 1024**3,
            preserved_scale_bytes / 1024**3,
        )
