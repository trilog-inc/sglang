import copy
import dataclasses
import logging
import os
import time
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_kt_ep_disabled_context,
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.draft_device import (
    draft_cuda_device_context,
    resolve_speculative_draft_device,
)
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    detect_nan,
    draft_tp_context,
    fast_topk,
    generate_token_bitmask,
    get_last_loc_large_page_size_large_top_k,
    load_token_map,
    select_top_k_tokens,
)
from sglang.srt.utils import (
    MultiprocessingSerializer,
    empty_context,
    get_available_gpu_memory,
    is_cuda,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_npu = is_npu()

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)


class EAGLEWorker(TpModelWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.enable_nan_detection = server_args.enable_nan_detection
        self.target_gpu_id = gpu_id
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_device = (
            torch.device("cuda", gpu_id) if is_cuda() else server_args.device
        )
        self.draft_gpu_id = (
            resolve_speculative_draft_device(server_args.speculative_draft_device)
            if server_args.speculative_draft_device is not None
            else gpu_id
        )
        if (
            server_args.speculative_draft_device is not None
            and self.draft_gpu_id == gpu_id
        ):
            raise ValueError(
                "--speculative-draft-device resolves to the target CUDA device "
                f"{gpu_id}; omit it for same-GPU drafting or select another GPU."
            )
        self._remote_draft = self.draft_gpu_id != gpu_id
        self.draft_device = (
            torch.device("cuda", self.draft_gpu_id)
            if is_cuda()
            else torch.device(server_args.device)
        )
        self._remote_req_owner: dict[int, int] = {}
        self._remote_req_synced_len: dict[int, int] = {}
        self._remote_req_owner_serial = 0
        if self._remote_draft:
            capability = torch.cuda.get_device_capability(self.draft_gpu_id)
            if capability != (8, 9):
                raise ValueError(
                    "GLM-5-Next remote MTP is currently validated for an RTX "
                    f"4090-class SM89 GPU, got SM{capability[0]}{capability[1]}."
                )
            peer_access = torch.cuda.can_device_access_peer(
                gpu_id, self.draft_gpu_id
            )
            logger.info(
                "GLM-5-Next heterogeneous MTP enabled: target=cuda:%d, "
                "draft=cuda:%d (%s), peer_access=%s.",
                gpu_id,
                self.draft_gpu_id,
                torch.cuda.get_device_name(self.draft_gpu_id),
                peer_access,
            )
            if not peer_access:
                logger.warning(
                    "CUDA peer access is unavailable between target and draft; "
                    "PyTorch will use its host-staged copy path."
                )
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        target_req_to_token_pool, target_token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self.req_to_token_pool = target_req_to_token_pool
        self.token_to_kv_pool_allocator = target_token_to_kv_pool_allocator

        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with (
            ctx
        ), self._draft_cuda_context(), speculative_moe_backend_context(), speculative_moe_a2a_backend_context(), speculative_kt_ep_disabled_context():
            super().__init__(
                server_args=server_args,
                gpu_id=self.draft_gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=(None if self._remote_draft else self.req_to_token_pool),
                token_to_kv_pool_allocator=(
                    None if self._remote_draft else self.token_to_kv_pool_allocator
                ),
            )

        # TpModelWorker represents the draft runner and therefore records its
        # GPU.  EAGLEWorker itself still coordinates target-side allocation and
        # verification, so retain an explicit target identity here.
        self.gpu_id = self.target_gpu_id
        self.device = server_args.device
        self._preserve_heterogeneous_flashinfer_arches()

        if self._remote_draft:
            # EAGLE's scheduling and verification logic must continue to use
            # the target allocator.  The draft ModelRunner owns a same-sized
            # physical KV pool on the 4090 and writes the same logical cache
            # locations there.
            self.draft_req_to_token_pool = self.draft_model_runner.req_to_token_pool
            self.draft_token_to_kv_pool_allocator = (
                self.draft_model_runner.token_to_kv_pool_allocator
            )
            self.req_to_token_pool = target_req_to_token_pool
            self.token_to_kv_pool_allocator = target_token_to_kv_pool_allocator
            if (
                self.draft_model_runner.max_total_num_tokens
                < self.target_worker.model_runner.max_total_num_tokens
            ):
                raise RuntimeError(
                    "The remote MTP KV pool is smaller than the target KV pool; "
                    "reduce --max-total-tokens or increase available memory on "
                    "the draft GPU."
                )
            target_mapping_shape = target_req_to_token_pool.req_to_token.shape
            draft_mapping_shape = self.draft_req_to_token_pool.req_to_token.shape
            if any(
                draft_size < target_size
                for draft_size, target_size in zip(
                    draft_mapping_shape, target_mapping_shape
                )
            ):
                raise RuntimeError(
                    "The remote MTP request mapping is smaller than the target "
                    f"mapping ({tuple(draft_mapping_shape)} < "
                    f"{tuple(target_mapping_shape)}); reduce the target request "
                    "or context limits."
                )
        else:
            self.draft_req_to_token_pool = self.req_to_token_pool
            self.draft_token_to_kv_pool_allocator = (
                self.token_to_kv_pool_allocator
            )

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_model_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_model_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_model_runner.model.hot_token_id.to(
                    embed.device
                )

        elif self._remote_draft:
            self._copy_embed_and_head_to_remote_draft(embed, head)
        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_model_runner.model.set_embed_and_head(embed, head)

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            self.eagle_use_aux_hidden_state = True
            eagle_config = getattr(
                self.draft_model_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        with self._draft_cuda_context(), self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

    def _draft_cuda_context(self):
        return (
            draft_cuda_device_context(self.draft_gpu_id)
            if self._remote_draft
            else nullcontext()
        )

    def _preserve_heterogeneous_flashinfer_arches(self) -> None:
        """Keep lazy FlashInfer builds valid for both GPUs in this process."""

        if (
            not self._remote_draft
            or "FLASHINFER_CUDA_ARCH_LIST" not in os.environ
        ):
            return

        def arch_for(device_id: int) -> str:
            major, minor = torch.cuda.get_device_capability(device_id)
            return f"{major}.{minor}{'a' if major >= 9 else ''}"

        arches = os.environ["FLASHINFER_CUDA_ARCH_LIST"].split()
        arches.extend((arch_for(self.target_gpu_id), arch_for(self.draft_gpu_id)))
        os.environ["FLASHINFER_CUDA_ARCH_LIST"] = " ".join(dict.fromkeys(arches))

    def _copy_embed_and_head_to_remote_draft(self, embed, head) -> None:
        """Populate the already-allocated draft vocabulary modules on RTX 4090."""

        draft_embed, draft_head = (
            self.draft_model_runner.model.get_embed_and_head()
        )
        with self._draft_cuda_context(), torch.no_grad():
            draft_embed.copy_(embed, non_blocking=True)
            draft_head.copy_(head, non_blocking=True)
            torch.cuda.current_stream(self.draft_gpu_id).synchronize()

    @staticmethod
    def _copy_spec_to_device(spec_info, device):
        if spec_info is None:
            return None
        copied = copy.copy(spec_info)
        if not dataclasses.is_dataclass(copied):
            return copied
        for field in dataclasses.fields(copied):
            value = getattr(copied, field.name)
            if isinstance(value, torch.Tensor) and not field.name.endswith("_cpu"):
                setattr(
                    copied,
                    field.name,
                    value.to(device=device, non_blocking=True),
                )
        return copied

    def _model_worker_batch_to_draft(self, model_worker_batch):
        if not self._remote_draft:
            return model_worker_batch
        copied = copy.copy(model_worker_batch)
        for field in dataclasses.fields(copied):
            value = getattr(copied, field.name)
            if isinstance(value, torch.Tensor) and not field.name.endswith("_cpu"):
                setattr(
                    copied,
                    field.name,
                    value.to(device=self.draft_device, non_blocking=True),
                )
        copied.spec_info = self._copy_spec_to_device(
            model_worker_batch.spec_info, self.draft_device
        )
        return copied

    def _remote_owner_for_request(self, req, row: int) -> int:
        """Return a collision-free owner generation for a request-pool row."""

        if req is None:
            return row
        owner = getattr(req, "_glm5_next_remote_mtp_owner", None)
        if owner is None:
            self._remote_req_owner_serial += 1
            owner = self._remote_req_owner_serial
            setattr(req, "_glm5_next_remote_mtp_owner", owner)
        return owner

    def _sync_remote_req_to_token(self, model_worker_batch) -> None:
        """Mirror active target request rows into the independent draft pool."""

        if not self._remote_draft or model_worker_batch.req_pool_indices.numel() == 0:
            return
        if model_worker_batch.reqs is not None:
            rows = [req.req_pool_idx for req in model_worker_batch.reqs]
        else:
            rows = model_worker_batch.req_pool_indices.detach().cpu().tolist()
        if model_worker_batch.seq_lens_cpu is not None:
            seq_lens = model_worker_batch.seq_lens_cpu.tolist()
        else:
            seq_lens = model_worker_batch.seq_lens.detach().cpu().tolist()
        reqs = model_worker_batch.reqs or [None] * len(rows)
        target_mapping = self.req_to_token_pool.req_to_token
        draft_mapping = self.draft_model_runner.req_to_token_pool.req_to_token
        margin = self.page_size + self.speculative_num_steps + 2

        target_ready = torch.cuda.Event()
        with torch.cuda.device(self.gpu_id):
            torch.cuda.current_stream(self.gpu_id).record_event(target_ready)
        with self._draft_cuda_context():
            torch.cuda.current_stream(self.draft_gpu_id).wait_event(target_ready)
            for row, seq_len, req in zip(rows, seq_lens, reqs):
                row = int(row)
                owner = self._remote_owner_for_request(req, row)
                end = min(
                    int(seq_len) + margin,
                    int(target_mapping.shape[1]),
                )
                if self._remote_req_owner.get(row) != owner:
                    start = 0
                else:
                    start = max(
                        0,
                        min(self._remote_req_synced_len.get(row, 0), end) - margin,
                    )
                if start < end:
                    draft_mapping[row, start:end].copy_(
                        target_mapping[row, start:end], non_blocking=True
                    )
                self._remote_req_owner[row] = owner
                self._remote_req_synced_len[row] = end

    def _copy_draft_capture_to_target(self, target_spec, draft_spec) -> None:
        for name in ("topk_p", "topk_index", "hidden_states"):
            value = getattr(draft_spec, name, None)
            setattr(
                target_spec,
                name,
                None
                if value is None
                else value.to(device=self.target_device, non_blocking=True),
            )

    def _handoff_draft_to_target(self) -> None:
        """Make target work wait for asynchronous draft-to-target copies."""

        if not self._remote_draft:
            return
        ready = torch.cuda.Event()
        torch.cuda.current_stream(self.draft_gpu_id).record_event(ready)
        with torch.cuda.device(self.gpu_id):
            torch.cuda.current_stream(self.gpu_id).wait_event(ready)

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners
        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        if self._remote_draft:
            # The existing graph runners assume target and draft buffers share
            # one CUDA device.  Keep target graphs enabled while running the
            # comparatively small MTP model eagerly on the 4090.
            logger.info(
                "Draft CUDA graphs are disabled for heterogeneous MTP; "
                "target-model CUDA graphs remain enabled."
            )
            return

        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

        # Capture extend
        if self.draft_extend_attn_backend and not _is_npu:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = EAGLEDraftExtendCudaGraphRunner(
                self
            )
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

    @property
    def draft_model_runner(self):
        return self.model_runner

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            logits_output, next_token_ids, seq_lens_cpu = self.forward_target_extend(
                batch
            )
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context(), speculative_kt_ep_disabled_context():
                self.forward_draft_extend(
                    batch,
                    logits_output.hidden_states,
                    next_token_ids,
                    seq_lens_cpu,
                    logits_output.mm_input_embeds,
                )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )
        else:
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context(), speculative_kt_ep_disabled_context():
                spec_info = self.draft(batch)
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                self.verify(batch, spec_info)
            )

            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context(), speculative_kt_ep_disabled_context():
                # NOTE: We should use `check_forward_draft_extend_after_decode`
                # when DP attention is enabled, but it is slow. Skip it for now.
                if (
                    self.server_args.enable_dp_attention
                    or batch.spec_info.verified_id.shape[0] > 0
                ):
                    # decode is not finished
                    self.forward_draft_extend_after_decode(batch)

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )

    def check_forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        local_need_forward = batch.spec_info.verified_id.shape[0] > 0
        if not self.server_args.enable_dp_attention:
            return local_need_forward

        global_need_forward = torch.tensor(
            [
                (local_need_forward),
            ],
            dtype=torch.int64,
        )
        torch.distributed.all_reduce(
            global_need_forward, group=get_tp_group().cpu_group
        )
        global_need_forward_cnt = global_need_forward[0].item()
        need_forward = global_need_forward_cnt > 0
        return need_forward

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, int, Optional[torch.Tensor]]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        logits_output, next_token_ids = (
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        return (
            logits_output,
            next_token_ids,
            model_worker_batch.seq_lens_cpu,
        )

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        batch.maybe_evict_swa()
        for req in batch.reqs:
            req.decode_batch_idx += 1

        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Accumulate penalty
        if batch.sampling_info.penalizer_orchestrator.is_required:
            # This is a relaxed version of penalties for speculative decoding.
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )

        # Allocate cache locations
        # Layout of the out_cache_loc
        # [       topk 0         ] [       topk 1         ]
        # [iter=0, iter=1, iter=2] [iter=0, iter=1, iter=2]
        if self.page_size == 1:
            alloc_len_per_decode = self.speculative_num_steps * self.topk
            # TODO: We only need self.speculative_num_steps - 1 * topk cache loc
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
                batch.tree_cache,
                num_seqs * alloc_len_per_decode,
                backup_state=True,
            )
        else:
            if self.topk == 1:
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                seq_lens_cpu = batch.seq_lens_cpu + self.speculative_num_steps
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:
                # In this case, the last partial page needs to be duplicated.
                # KV cache layout in batch.req_to_token_pool.req_to_token:
                #
                # | -------- | -- xxxx .. | -- xxxx .. | -- xxxx .. |
                #    prefix     top-k = 0    tok-k = 1    top-k = 2
                #
                #  "-" means prefix tokens
                #  "x" means speculative draft tokens
                #  "." means padded tokens

                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.num_new_pages_per_topk,
                    self.extend_lens,
                    last_page_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                    self.topk,
                    self.page_size,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                last_page_lens_cpu = prefix_lens_cpu % self.page_size
                num_new_pages_per_topk = (
                    last_page_lens_cpu + self.speculative_num_steps + self.page_size - 1
                ) // self.page_size
                seq_lens_cpu = (
                    prefix_lens_cpu // self.page_size * self.page_size
                    + num_new_pages_per_topk * (self.page_size * self.topk)
                )
                extend_num_tokens = torch.sum((seq_lens_cpu - prefix_lens_cpu)).item()

            out_cache_loc, token_to_kv_pool_state_backup = (
                alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )

        if self.page_size > 1 and self.topk > 1:
            last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
            duplicate_cache_len = torch.sum(last_page_lens_cpu).item() * (self.topk - 1)
            target_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
            source_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
        else:
            # When source_cache_loc is not needed, simply skip
            duplicate_cache_len = 0
            source_cache_loc, target_cache_loc, last_page_lens_cumsum = None, None, None

        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens,
            self.num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
            self.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps + self.page_size),
        )

        if self.page_size > 1 and self.topk > 1:
            if duplicate_cache_len > 0:
                self.draft_model_runner.token_to_kv_pool.move_kv_cache(
                    target_cache_loc, source_cache_loc
                )
            # Remove padded slots
            # TODO: We only need self.speculative_num_steps - 1 cache loc
            out_cache_loc = out_cache_loc[
                : num_seqs * self.topk * self.speculative_num_steps
            ]

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        batch.spec_info = EagleDraftInput.create_idle_input(
            device=self.device,
            hidden_size=self.model_config.spec_hidden_size,
            dtype=self.model_config.dtype,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def draft(self, batch: ScheduleBatch):
        # Parse args
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
        else:
            self._draft_preprocess_decode(batch)

        target_spec_info = batch.spec_info
        assert isinstance(target_spec_info, EagleDraftInput)

        target_spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        target_spec_info.num_tokens_per_req = self.topk
        target_spec_info.num_tokens_for_logprob_per_req = self.topk
        batch.return_hidden_states = False

        # Get forward batch
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        self._sync_remote_req_to_token(model_worker_batch)

        with self._draft_cuda_context():
            model_worker_batch = self._model_worker_batch_to_draft(
                model_worker_batch
            )
            forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.draft_model_runner
            )
            spec_info = forward_batch.spec_info
            assert isinstance(spec_info, EagleDraftInput)
            can_cuda_graph = (
                self.cuda_graph_runner
                and self.cuda_graph_runner.can_run(forward_batch)
            )
            if can_cuda_graph:
                parent_list, top_scores_index, draft_tokens = (
                    self.cuda_graph_runner.replay(forward_batch)
                )
            else:
                forward_batch.can_run_dp_cuda_graph = False
                if (
                    not forward_batch.forward_mode.is_idle()
                    and self.speculative_num_steps > 1
                ):
                    # Skip attention backend init for idle mode or 1-step draft
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                # Run forward steps
                parent_list, top_scores_index, draft_tokens = self.draft_forward(
                    forward_batch
                )

            if batch.forward_mode.is_idle():
                verify_input = EagleVerifyInput.create_idle_input(
                    self.topk,
                    self.speculative_num_steps,
                    self.speculative_num_draft_tokens,
                )
            else:
                (
                    tree_mask,
                    position,
                    retrive_index,
                    retrive_next_token,
                    retrive_next_sibling,
                    draft_tokens,
                ) = build_tree_kernel_efficient(
                    spec_info.verified_id,
                    parent_list,
                    top_scores_index,
                    draft_tokens,
                    forward_batch.seq_lens,
                    batch.seq_lens_sum,
                    self.topk,
                    self.speculative_num_steps,
                    self.speculative_num_draft_tokens,
                )

                verify_input = EagleVerifyInput(
                    draft_token=draft_tokens,
                    custom_mask=tree_mask,
                    positions=position,
                    retrive_index=retrive_index,
                    retrive_next_token=retrive_next_token,
                    retrive_next_sibling=retrive_next_sibling,
                    retrive_cum_len=None,
                    spec_steps=self.speculative_num_steps,
                    topk=self.topk,
                    draft_token_num=self.server_args.speculative_num_draft_tokens,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    seq_lens_sum=forward_batch.seq_lens_sum,
                    seq_lens_cpu=forward_batch.seq_lens_cpu,
                )

            if self._remote_draft:
                verify_input = self._copy_spec_to_device(
                    verify_input, self.target_device
                )
                self._handoff_draft_to_target()
            return verify_input

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
        # TODO: We only need self.speculative_num_steps - 1 cache loc
        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        # Reuse NSA/DSA topk_indices from the first draft forward step for
        # subsequent steps, analogous to skip_topk in deepseek_v2.py layers.
        # Only safe with topk == 1: select_top_k_tokens reorders candidate rows
        # each step, which would desync the cached indices from their rows.
        index_share_for_mtp_iteration = (
            getattr(self.model_config.hf_config, "index_share_for_mtp_iteration", False)
            and self.topk == 1
        )
        if index_share_for_mtp_iteration:
            forward_batch.reuse_mtp_topk_indices = True
            forward_batch.topk_indices = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            # This is a temporary fix for the case that the user is using standalone
            # speculative decoding and the draft model architecture is gpt-oss. gpt-oss
            # rope kernel needs cache_loc to be contiguous.
            if (
                self.server_args.speculative_algorithm == "STANDALONE"
                and self.model_config.hf_config.architectures[0] == "GptOssForCausalLM"
            ):
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            if self.server_args.enable_nan_detection:
                detect_nan(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        if index_share_for_mtp_iteration:
            forward_batch.topk_indices = None
            forward_batch.reuse_mtp_topk_indices = False
        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        return parent_list, top_scores_index, draft_tokens

    def clear_cache_pool(self):
        if not self._remote_draft:
            # Allocator and request rows are shared with the target worker.
            return
        with self._draft_cuda_context():
            self.draft_req_to_token_pool.clear()
            self.draft_token_to_kv_pool_allocator.clear()
        self._remote_req_owner.clear()
        self._remote_req_synced_len.clear()

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        seq_lens_pre_verify = batch.seq_lens.clone()
        spec_info.prepare_for_verify(batch, self.page_size)
        spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrive_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrive_next_token.shape
            ).cpu()

        # Forward
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            # Overlap the CPU operations for bitmask generation with the forward pass.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert spec_info.grammar is not None
                vocab_mask = vocab_mask.to(spec_info.retrive_next_token.device)
                # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        if self.enable_nan_detection:
            detect_nan(logits_output)

        spec_info.hidden_states = logits_output.hidden_states
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        # Post process based on verified outputs.
        # Pick indices that we care (accepted)
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]

        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
            or self.target_worker.model_runner.hybrid_lightning_config is not None
        ):
            self._mamba_verify_update(
                batch, res, logits_output, spec_info, seq_lens_pre_verify
            )

        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        # Prepare the batch for the next draft forwards.
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    def _mamba_verify_update(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
        spec_info: EagleVerifyInput,
        seq_lens_pre_verify: torch.Tensor,
    ):
        accepted_length = (
            torch.tensor(
                res.accept_length_per_req_cpu,
                device=logits_output.hidden_states.device,
                dtype=torch.int64,
            )
            + 1
        )
        cumulative_accepted_lengths = torch.cumsum(accepted_length, dim=0)
        # prepend 0 to the cumulative_accepted_lengths
        accepted_indices_start = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=cumulative_accepted_lengths.dtype,
                    device=cumulative_accepted_lengths.device,
                ),
                cumulative_accepted_lengths[:-1],
            ]
        )
        accepted_indices_offset = torch.arange(
            0,
            len(batch.seq_lens) * batch.spec_info.draft_token_num,
            step=batch.spec_info.draft_token_num,
            dtype=accepted_indices_start.dtype,
            device=accepted_indices_start.device,
        )

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        # res.accepted_indices.shape[0] > 0 skips DP attn idle batch
        if spec_info.topk > 1 and res.accepted_indices.shape[0] > 0:
            # accepted_indices=[0,2,3,4,5,7,9,10,11], accepted_length=[4, 3, 2], cumulative_accepted_lengths=[4, 7, 9]
            # first_token_indices_per_req=prepend(0, accepted_indices[cumulative_accepted_lengths[:-1]]) = [0, 5, 10]
            # last_token_indices_per_req=accepted_indices[cumulative_accepted_lengths - 1] = [4, 9, 11] (last token ID of each req)
            # max_relative_indices_per_req = [4,4,1]; those are the per-req spec-decoding step offsets that contain the correct mamba caches
            # first_token_indices_per_req = res.accepted_indices[accepted_indices_start]
            accepted_steps = (
                res.accepted_indices[cumulative_accepted_lengths - 1]
                - accepted_indices_offset
            )
        else:
            accepted_steps = accepted_length - 1

        if batch.mamba_track_indices is not None:
            # If after verify, the request's seq_lens has crossed a mamba track interval,
            # we need to update the mamba state for the request at the crossing point.
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            mamba_steps_to_track = torch.where(
                to_track_mask,
                res.accepted_indices[to_track_ith + accepted_indices_start]
                - accepted_indices_offset,
                -1,
            )
        else:
            mamba_steps_to_track = None

        self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        mm_input_embeds: Optional[torch.Tensor] = None,
    ):
        """Run draft model extend. This API modifies the states of the batch.

        Args:
            batch: The batch to run.
            hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        batch.return_hidden_states = False
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=seq_lens_cpu
        )
        target_spec_info = batch.spec_info
        self._sync_remote_req_to_token(model_worker_batch)
        with self._draft_cuda_context():
            model_worker_batch = self._model_worker_batch_to_draft(
                model_worker_batch
            )
            forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.draft_model_runner
            )
            forward_batch.return_logprob = False
            if mm_input_embeds is not None:
                forward_batch.mm_input_embeds = mm_input_embeds.to(
                    self.draft_device, non_blocking=True
                )
            logits_output = self.draft_model_runner.forward(
                forward_batch
            ).logits_output
            if self.enable_nan_detection:
                detect_nan(logits_output)
            assert isinstance(forward_batch.spec_info, EagleDraftInput)
            if not self._remote_draft:
                assert forward_batch.spec_info is target_spec_info
            self.capture_for_decode(logits_output, forward_batch.spec_info)
            if self._remote_draft:
                self._copy_draft_capture_to_target(
                    target_spec_info, forward_batch.spec_info
                )
                self._handoff_draft_to_target()

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        assert isinstance(batch.spec_info, EagleDraftInput)
        # Backup fields that will be modified in-place
        seq_lens_backup = batch.seq_lens.clone()
        seq_lens_cpu_backup = batch.seq_lens_cpu.clone()
        req_pool_indices_backup = batch.req_pool_indices
        accept_length_backup = batch.spec_info.accept_length
        return_logprob_backup = batch.return_logprob

        input_is_idle = batch.forward_mode.is_idle()

        if not input_is_idle and batch.spec_info.verified_id.numel() == 0:
            batch = batch.copy()
            batch.prepare_for_idle()
            hidden_size = self.model_config.spec_hidden_size
            if (
                self.speculative_algorithm.is_eagle3()
                and self.eagle_use_aux_hidden_state
            ):
                hidden_size = self.model_config.hidden_size * 3
            batch.spec_info = EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=hidden_size,
                dtype=self.model_config.dtype,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )

        batch.spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        batch.spec_info.num_tokens_for_logprob_per_req = 1
        batch.spec_info.prepare_extend_after_decode(
            batch,
            self.speculative_num_steps,
        )
        batch.forward_mode = (
            ForwardMode.DRAFT_EXTEND
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )

        batch.return_hidden_states = False
        target_spec_info = batch.spec_info
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        self._sync_remote_req_to_token(model_worker_batch)
        with self._draft_cuda_context():
            model_worker_batch = self._model_worker_batch_to_draft(
                model_worker_batch
            )
            forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.draft_model_runner
            )
            if forward_batch.seq_lens_cpu is not None:
                forward_batch.seq_lens_sum = (
                    forward_batch.seq_lens_cpu.sum().item()
                )
            else:
                forward_batch.seq_lens_sum = forward_batch.seq_lens.sum().item()

            # Run
            can_cuda_graph = (
                self.cuda_graph_runner_for_draft_extend
                and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
            )
            if can_cuda_graph:
                logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                    forward_batch
                )
                (
                    forward_batch.spec_info.topk_p,
                    forward_batch.spec_info.topk_index,
                ) = (logits_output.topk_p, logits_output.topk_index)
                forward_batch.spec_info.hidden_states = logits_output.hidden_states
            else:
                forward_batch.can_run_dp_cuda_graph = False
                if not forward_batch.forward_mode.is_idle():
                    self.draft_model_runner.attn_backend.init_forward_metadata(
                        forward_batch
                    )
                logits_output = self.draft_model_runner.forward(
                    forward_batch, skip_attn_backend_init=True
                ).logits_output
                self.capture_for_decode(logits_output, forward_batch.spec_info)

            if self.enable_nan_detection:
                detect_nan(logits_output)
            if self._remote_draft:
                self._copy_draft_capture_to_target(
                    target_spec_info, forward_batch.spec_info
                )
                self._handoff_draft_to_target()

        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        batch.forward_mode = (
            ForwardMode.DECODE if not input_is_idle else ForwardMode.IDLE
        )
        batch.seq_lens = seq_lens_backup
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.req_pool_indices = req_pool_indices_backup
        batch.spec_info.accept_length = accept_length_backup
        batch.return_logprob = return_logprob_backup

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        draft_input.hidden_states = logits_output.hidden_states

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        with self._draft_cuda_context():
            success, message = self.model_runner.update_weights_from_tensor(
                named_tensors=named_tensors,
                load_format=recv_req.load_format,
            )
        if not success:
            return success, message

        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message


@torch.compile(dynamic=True, disable=_is_npu)
def get_last_loc_large_page_size_top_k_1(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens,
    speculative_num_steps: int,
):
    prefix_lens = seq_lens
    seq_lens = prefix_lens + speculative_num_steps
    last_loc = get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )
    return prefix_lens, seq_lens, last_loc
