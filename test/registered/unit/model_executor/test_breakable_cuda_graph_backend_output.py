"""CPU unit tests for breakable CUDA graph structured outputs."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import mock_open, patch

import torch

import sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend as bcg_module
import sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph as bcg_core
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBreakableCudaGraphStructuredOutput(CustomTestCase):
    def test_replay_refreshes_live_non_padded_token_count(self):
        backend = object.__new__(bcg_module.BreakableCudaGraphBackend)
        backend._debug_eager = False
        backend._outputs = {ShapeKey(size=1): "output"}
        captured_batch = SimpleNamespace(num_token_non_padded_cpu=None)
        live_batch = SimpleNamespace(num_token_non_padded_cpu=4)
        replay_counts = []
        backend._capture_inputs = {ShapeKey(size=1): captured_batch}
        backend._graphs = {
            ShapeKey(size=1): SimpleNamespace(
                replay=lambda: replay_counts.append(
                    captured_batch.num_token_non_padded_cpu
                )
            )
        }

        output = backend.replay(ShapeKey(size=1), live_batch)

        self.assertEqual(output, "output")
        self.assertEqual(replay_counts, [4])

    def test_debug_eager_can_replay_with_live_forward_batch(self):
        backend = object.__new__(bcg_module.BreakableCudaGraphBackend)
        backend._debug_eager = True
        backend._outputs = {ShapeKey(size=1): "output"}
        captured_batch = SimpleNamespace(
            marker="captured", dp_padding_mode="capture-control"
        )
        live_batch = SimpleNamespace(marker="live", dp_padding_mode=None)
        replay_markers = []
        backend._capture_inputs = {ShapeKey(size=1): captured_batch}
        backend._graphs = {
            ShapeKey(size=1): SimpleNamespace(
                replay=lambda: replay_markers.append(
                    (captured_batch.marker, captured_batch.dp_padding_mode)
                )
            )
        }

        with patch.object(
            bcg_module,
            "get_bool_env_var",
            side_effect=lambda name: name
            == "SGLANG_BCG_DEBUG_USE_LIVE_FORWARD_BATCH",
        ):
            output = backend.replay(ShapeKey(size=1), live_batch)

        self.assertEqual(output, "output")
        self.assertEqual(replay_markers, [("live", "capture-control")])

    def test_debug_eager_can_select_live_forward_batch_fields(self):
        backend = object.__new__(bcg_module.BreakableCudaGraphBackend)
        backend._debug_eager = True
        backend._outputs = {ShapeKey(size=1): "output"}
        captured_batch = SimpleNamespace(
            marker="captured",
            untouched="captured",
            dp_padding_mode="capture-control",
        )
        live_batch = SimpleNamespace(
            marker="live",
            untouched="live",
            dp_padding_mode=None,
        )
        replay_values = []
        backend._capture_inputs = {ShapeKey(size=1): captured_batch}
        backend._graphs = {
            ShapeKey(size=1): SimpleNamespace(
                replay=lambda: replay_values.append(
                    (
                        captured_batch.marker,
                        captured_batch.untouched,
                        captured_batch.dp_padding_mode,
                    )
                )
            )
        }

        with (
            patch.object(
                bcg_module,
                "get_bool_env_var",
                side_effect=lambda name: name
                == "SGLANG_BCG_DEBUG_USE_LIVE_FORWARD_BATCH",
            ),
            patch.dict(
                "os.environ",
                {
                    "SGLANG_BCG_DEBUG_LIVE_FORWARD_BATCH_FIELDS_FILE": (
                        "/tmp/live-fields"
                    )
                },
            ),
            patch("builtins.open", mock_open(read_data="marker")),
        ):
            output = backend.replay(ShapeKey(size=1), live_batch)

        self.assertEqual(output, "output")
        self.assertEqual(
            replay_values,
            [("live", "captured", "capture-control")],
        )

        with (
            patch.object(
                bcg_module,
                "get_bool_env_var",
                side_effect=lambda name: name
                == "SGLANG_BCG_DEBUG_USE_LIVE_FORWARD_BATCH",
            ),
            patch.dict(
                "os.environ",
                {
                    "SGLANG_BCG_DEBUG_LIVE_FORWARD_BATCH_FIELDS_FILE": (
                        "/tmp/live-fields"
                    )
                },
            ),
            patch("builtins.open", mock_open(read_data="untouched")),
        ):
            backend.replay(ShapeKey(size=1), live_batch)

        self.assertEqual(
            replay_values,
            [
                ("live", "captured", "capture-control"),
                ("captured", "live", "capture-control"),
            ],
        )

    def test_capture_drains_final_warmup_before_graph_construction(self):
        call_log = []
        backend = object.__new__(bcg_module.BreakableCudaGraphBackend)
        backend._device_module = SimpleNamespace(
            synchronize=lambda: call_log.append("synchronize")
        )
        backend._tp_group = SimpleNamespace(
            barrier=lambda: call_log.append("barrier")
        )
        backend._debug_eager = False
        backend._pool = None
        backend._capture_stream = None
        backend._shared_output_buffer = None
        backend._use_shared_output_buffer = None
        backend._graphs = {}
        backend._outputs = {}
        backend._capture_inputs = {}

        def forward_fn():
            call_log.append("forward")
            return None

        def post_warmup_hook():
            call_log.append("post_warmup_hook")

        class _Graph:
            def __init__(self, *args, **kwargs):
                call_log.append("graph_create")

        with (
            patch.object(bcg_module, "BreakableCUDAGraph", _Graph),
            patch.object(
                bcg_module,
                "BreakableCUDAGraphCapture",
                return_value=contextlib.nullcontext(),
            ),
        ):
            backend.capture_one(
                ShapeKey(size=4),
                forward_fn,
                post_warmup_hook=post_warmup_hook,
            )

        graph_idx = call_log.index("graph_create")
        last_hook_idx = max(
            i for i, value in enumerate(call_log) if value == "post_warmup_hook"
        )
        sync_idx = call_log.index("synchronize", last_hook_idx + 1, graph_idx)
        barrier_idx = call_log.index("barrier", sync_idx + 1, graph_idx)

        self.assertEqual(call_log.count("forward"), 3)
        self.assertEqual(call_log.count("post_warmup_hook"), 2)
        self.assertLess(last_hook_idx, sync_idx)
        self.assertLess(sync_idx, barrier_idx)
        self.assertLess(barrier_idx, graph_idx)

    def test_debug_eager_suspends_nested_graph_breaks(self):
        class _Graph:
            def __init__(self):
                self._break_fns = []
                self._break_labels = []

        class _Capture:
            def __init__(self):
                self.cuda_graph = _Graph()
                self._barrier_fn = None
                self.end_count = 0
                self.begin_count = 0

            def _end_current_segment(self):
                self.end_count += 1

            def _begin_new_segment(self):
                self.begin_count += 1

        capture = _Capture()

        @bcg_core.eager_on_graph(True)
        def nested(value):
            return value + 1

        @bcg_core.eager_on_graph(True, suspend_nested_capture=True)
        def whole_forward(value):
            return nested(value) * 2

        token = bcg_core._current_capture_var.set(capture)
        try:
            self.assertEqual(whole_forward(3), 8)
        finally:
            bcg_core._current_capture_var.reset(token)

        self.assertEqual(capture.end_count, 1)
        self.assertEqual(capture.begin_count, 1)
        self.assertEqual(len(capture.cuda_graph._break_fns), 1)

    def test_logits_output_is_retained_without_request_row_truncation(self):
        backend = object.__new__(bcg_module.BreakableCudaGraphBackend)
        backend._device_module = SimpleNamespace(synchronize=lambda: None)
        backend._tp_group = SimpleNamespace(barrier=lambda: None)
        backend._debug_eager = False
        backend._pool = None
        backend._capture_stream = None
        backend._shared_output_buffer = None
        backend._use_shared_output_buffer = None
        backend._graphs = {}
        backend._outputs = {}
        backend._capture_inputs = {}

        # ShapeKey.size is request batch size. DSpark verifies six tokens per
        # request, so each tensor field has six times as many leading rows.
        def make_output(rows: int, fill: float) -> LogitsProcessorOutput:
            return LogitsProcessorOutput(
                next_token_logits=torch.full((rows, 7), fill),
                hidden_states=torch.full((rows, 5), fill),
            )

        class _Graph:
            def __init__(self, *args, **kwargs):
                pass

            def replay(self):
                pass

        with (
            patch.object(bcg_module, "BreakableCUDAGraph", _Graph),
            patch.object(
                bcg_module,
                "BreakableCUDAGraphCapture",
                return_value=contextlib.nullcontext(),
            ),
        ):
            captured_outputs = {}
            for size in (4, 2):
                rows = size * 6
                outputs = [
                    make_output(rows, 1),
                    make_output(rows, 2),
                    make_output(rows, 3),
                ]
                captured_outputs[size] = outputs[-1]
                backend.capture_one(ShapeKey(size=size), lambda: outputs.pop(0))

        for size in (4, 2):
            stored = backend.replay(ShapeKey(size=size), static_forward_batch=None)
            self.assertIs(stored, captured_outputs[size])
            self.assertEqual(stored.next_token_logits.shape, (size * 6, 7))
            self.assertEqual(stored.hidden_states.shape, (size * 6, 5))
        self.assertIsNone(backend._shared_output_buffer)
        self.assertFalse(backend._use_shared_output_buffer)


if __name__ == "__main__":
    unittest.main()
