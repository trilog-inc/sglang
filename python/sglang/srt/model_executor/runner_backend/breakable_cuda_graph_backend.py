# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""BreakableCudaGraphBackend — segment-captured graphs with eager break
markers (eager_on_graph decorators on attention / mamba layers).
No torch.compile.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    DedupedCudaGraphMixin,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


class BreakableCudaGraphBackend(DedupedCudaGraphMixin, BaseCudaGraphBackend):
    """Segmented capture: graphs break at attention / mamba boundaries;
    attention metadata is recomputed at replay outside captured segments.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
        debug_eager: bool = False,
    ) -> None:
        self._model_runner = cuda_graph_runner.model_runner
        self._graphs: Dict[Any, BreakableCUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._capture_inputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._debug_eager = debug_eager
        self._shared_output_buffer: Optional[Any] = None
        self._use_shared_output_buffer: Optional[bool] = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            raise NotImplementedError(
                "Breakable CUDA graph is not compatible with memory saver mode"
            )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        self._shared_output_buffer = None
        self._use_shared_output_buffer = None
        self.begin_cuda_graph_capture()
        try:
            with self.replay_session():
                yield
        finally:
            try:
                self.end_cuda_graph_capture()
            finally:
                self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        warmup_out = None
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            warmup_out = forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        # Drain work still in flight from the final warmup (and its hook)
        # before entering capture.  This is especially important for models
        # whose warmup triggers asynchronous JIT compilation or side-stream
        # state preparation.  Align TP ranks only after the device is drained
        # so no rank starts capturing while another is finishing warmup.
        self._device_module.synchronize()
        self._tp_group.barrier()

        graph = BreakableCUDAGraph(debug_name=f"shape={shape_key!r}")
        captured_fn = (
            eager_on_graph(True, suspend_nested_capture=True)(forward_fn)
            if self._debug_eager
            else forward_fn
        )
        size = shape_key.size
        supports_shared_output = self._supports_shared_output(warmup_out)
        if self._use_shared_output_buffer is None:
            self._use_shared_output_buffer = supports_shared_output
        elif self._use_shared_output_buffer != supports_shared_output:
            raise ValueError(
                "BCG output structure changed between capture sizes: "
                f"shared-buffer support changed to {supports_shared_output} "
                f"for {type(warmup_out)}"
            )

        if supports_shared_output and self._shared_output_buffer is None:
            self._shared_output_buffer = self._alloc_full_buffer(warmup_out, size)
        with BreakableCUDAGraphCapture(
            cuda_graph=graph,
            pool=self._pool,
            stream=self._capture_stream,
            barrier_fn=self._tp_group.barrier,
        ):
            out = captured_fn()
            if self._supports_shared_output(out) != supports_shared_output:
                raise ValueError(
                    "BCG output structure changed between warmup and capture: "
                    f"{type(warmup_out)} vs {type(out)}"
                )
            if supports_shared_output:
                out_rows = self._output_rows(out, size)
                self._copy_output_to_buffer(out, self._shared_output_buffer, out_rows)

        if supports_shared_output:
            stored = self._slice_output(self._shared_output_buffer, out_rows)
        else:
            # Structured model outputs (notably LogitsProcessorOutput during
            # speculative target verification) already own graph-stable tensor
            # storage.  Keep the captured object directly, as FullCudaGraphBackend
            # does.  Sending it through the shared tensor-tree buffer would both
            # mistake request batch size for verify-token rows and add a large
            # logits copy to every replay.
            stored = out
        self._graphs[shape_key] = graph
        self._outputs[shape_key] = stored
        # CUDA graphs retain tensor addresses, not Python tensor lifetimes.
        self._capture_inputs[shape_key] = capture_inputs

    @classmethod
    def _supports_shared_output(cls, output: Any) -> bool:
        """Whether ``output`` can use the recursive shared tensor-tree buffer."""
        if (
            output is None
            or torch.is_tensor(output)
            or isinstance(output, PPProxyTensors)
        ):
            return True
        if isinstance(output, (list, tuple)):
            return all(cls._supports_shared_output(item) for item in output)
        return False

    def _output_rows(self, output: Any, cap: int) -> int:
        """Leading-dim row count actually produced by the body, clamped to ``cap``.

        A body that shards or prunes its output along dim 0 returns fewer than
        ``cap`` rows; everything else returns exactly ``cap``.
        """
        if torch.is_tensor(output):
            return min(cap, output.shape[0])
        if isinstance(output, PPProxyTensors):
            rows = [t.shape[0] for t in output.tensors.values()]
            return min([cap, *rows])
        if isinstance(output, (list, tuple)) and output:
            return min(self._output_rows(o, cap) for o in output if o is not None)
        return cap

    def _alloc_full_buffer(self, output: Any, size: int) -> Any:
        """A same-structure buffer as ``output`` but with ``size`` leading rows."""
        if output is None:
            return None
        if torch.is_tensor(output):
            return output.new_empty((size, *output.shape[1:]))
        if isinstance(output, PPProxyTensors):
            return PPProxyTensors(
                {
                    key: t.new_empty((size, *t.shape[1:]))
                    for key, t in output.tensors.items()
                }
            )
        if isinstance(output, tuple):
            return tuple(self._alloc_full_buffer(o, size) for o in output)
        if isinstance(output, list):
            return [self._alloc_full_buffer(o, size) for o in output]
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    def _slice_output(self, output: Any, num_tokens: int) -> Any:
        if output is None:
            return None
        if torch.is_tensor(output):
            return output[:num_tokens]
        if isinstance(output, PPProxyTensors):
            return output[:num_tokens]
        if isinstance(output, tuple):
            return tuple(self._slice_output(item, num_tokens) for item in output)
        if isinstance(output, list):
            return [self._slice_output(item, num_tokens) for item in output]
        raise TypeError(f"Unsupported BCG output type: {type(output)}")

    def _copy_output_to_buffer(
        self, output: Any, output_buffer: Any, num_tokens: int
    ) -> None:
        if output is None or output_buffer is None:
            if output is None and output_buffer is None:
                return
            raise ValueError(
                "BCG output structure changed between capture sizes: "
                f"{type(output)} vs {type(output_buffer)}"
            )
        if torch.is_tensor(output) and torch.is_tensor(output_buffer):
            output_buffer[:num_tokens].copy_(output[:num_tokens])
            return
        if isinstance(output, PPProxyTensors) and isinstance(
            output_buffer, PPProxyTensors
        ):
            if output.tensors.keys() != output_buffer.tensors.keys():
                raise ValueError(
                    "BCG output proxy structure changed between capture sizes: "
                    f"{output.tensors.keys()} != {output_buffer.tensors.keys()}"
                )
            for key, tensor in output.tensors.items():
                self._copy_output_to_buffer(
                    tensor, output_buffer.tensors[key], num_tokens
                )
            return
        if isinstance(output, (list, tuple)) and isinstance(
            output_buffer, type(output)
        ):
            if len(output) != len(output_buffer):
                raise ValueError(
                    "BCG output sequence structure changed between capture sizes: "
                    f"{len(output)} != {len(output_buffer)}"
                )
            for item, buffer in zip(output, output_buffer):
                self._copy_output_to_buffer(item, buffer, num_tokens)
            return
        raise TypeError(
            "Unsupported BCG output buffer pair: "
            f"{type(output)} vs {type(output_buffer)}"
        )

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        with enable_breakable_cuda_graph():
            yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        # Diagnostic escape hatch: --debug-cuda-graph already makes the model
        # forward an eager graph break.  Optionally replace the capture-time
        # ForwardBatch fields with their live replay values as well, allowing
        # graph-stable input rails to be isolated from replay-prepared attention
        # metadata without changing the production capture path.
        if self._debug_eager and get_bool_env_var(
            "SGLANG_BCG_DEBUG_USE_LIVE_FORWARD_BATCH"
        ):
            captured_forward_batch = self._capture_inputs.get(shape_key)
            if captured_forward_batch is not None:
                capture_input_snapshots = getattr(
                    self, "_debug_capture_input_snapshots", None
                )
                if capture_input_snapshots is None:
                    capture_input_snapshots = {}
                    self._debug_capture_input_snapshots = capture_input_snapshots
                captured_fields = capture_input_snapshots.setdefault(
                    shape_key, captured_forward_batch.__dict__.copy()
                )
                captured_forward_batch.__dict__.clear()
                captured_forward_batch.__dict__.update(captured_fields)
                # These fields are runner-owned graph controls/buffers rather
                # than request data.  A live ForwardBatch legitimately leaves
                # several of them unset; preserve their capture-time values so
                # run_once still executes with the graph runner's contract.
                graph_owned_fields = (
                    "capture_hidden_mode",
                    "dp_padding_mode",
                    "global_dp_buffer_len",
                    "global_forward_mode",
                    "global_num_tokens_cpu",
                    "global_num_tokens_for_logprob_gpu",
                    "global_num_tokens_gpu",
                    "next_token_logits_buffer",
                )
                graph_owned = {
                    name: getattr(captured_forward_batch, name, None)
                    for name in graph_owned_fields
                }
                fields_file = os.getenv(
                    "SGLANG_BCG_DEBUG_LIVE_FORWARD_BATCH_FIELDS_FILE"
                )
                if fields_file:
                    try:
                        with open(fields_file) as f:
                            live_fields = {
                                field.strip()
                                for field in f.read().split(",")
                                if field.strip()
                            }
                    except OSError:
                        live_fields = set()
                    for name in live_fields:
                        if name in static_forward_batch.__dict__:
                            setattr(
                                captured_forward_batch,
                                name,
                                getattr(static_forward_batch, name),
                            )
                else:
                    captured_forward_batch.__dict__.update(
                        static_forward_batch.__dict__
                    )
                captured_forward_batch.__dict__.update(graph_owned)

        # Python eager breaks close over the capture-time ForwardBatch object,
        # not the live object passed to replay.  Keep the real row count live:
        # DSV4 uses it to slice low-ratio source projections and attention.
        # Leaving the capture-time None here makes those breaks process padded
        # graph rows as real tokens and corrupts target-verify output.
        captured_forward_batch = self._capture_inputs.get(shape_key)
        if (
            captured_forward_batch is not None
            and static_forward_batch is not None
            and hasattr(static_forward_batch, "num_token_non_padded_cpu")
        ):
            captured_forward_batch.num_token_non_padded_cpu = (
                static_forward_batch.num_token_non_padded_cpu
            )
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self.close()
        self._graphs.clear()
        self._outputs.clear()
        self._capture_inputs.clear()
        self._pool = None
        self._shared_output_buffer = None
        self._use_shared_output_buffer = None
