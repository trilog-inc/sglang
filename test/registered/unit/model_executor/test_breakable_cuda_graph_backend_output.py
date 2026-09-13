"""CPU unit tests for breakable CUDA graph structured outputs."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend as bcg_module
import sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph as bcg_core
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBreakableCudaGraphStructuredOutput(CustomTestCase):
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
