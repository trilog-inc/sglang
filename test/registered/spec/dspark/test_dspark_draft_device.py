import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.layers.quantization.fp8_utils as fp8_utils
from sglang.srt.layers.quantization.fp8_utils import (
    Fp8GemmRunnerBackend,
    fp8_gemm_runner_backend_context,
    get_fp8_gemm_runner_backend,
)
from sglang.srt.speculative.draft_worker_common import (
    resolve_speculative_draft_device,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestResolveSpeculativeDraftDevice(CustomTestCase):
    def test_logical_indices(self):
        with patch("torch.cuda.device_count", return_value=3):
            self.assertEqual(resolve_speculative_draft_device("2"), 2)
            self.assertEqual(resolve_speculative_draft_device("cuda:1"), 1)
            with self.assertRaisesRegex(ValueError, "only 3 CUDA devices"):
                resolve_speculative_draft_device("3")

    def test_gpu_uuid(self):
        uuids = [
            "GPU-56fd682e-006d-2a13-a642-31748746dba6",
            "GPU-b6bb9e3a-d439-ef36-dcf5-40eeb5870765",
        ]
        with (
            patch("torch.cuda.device_count", return_value=2),
            patch(
                "torch.cuda.get_device_properties",
                side_effect=lambda index: SimpleNamespace(uuid=uuids[index]),
            ),
        ):
            self.assertEqual(resolve_speculative_draft_device(uuids[1]), 1)

    def test_fp8_backend_override_is_scoped_to_draft_context(self):
        with patch.object(
            fp8_utils,
            "FP8_GEMM_RUNNER_BACKEND",
            Fp8GemmRunnerBackend.CUTLASS,
        ):
            self.assertEqual(
                get_fp8_gemm_runner_backend(), Fp8GemmRunnerBackend.CUTLASS
            )
            with fp8_gemm_runner_backend_context("triton"):
                self.assertEqual(
                    get_fp8_gemm_runner_backend(), Fp8GemmRunnerBackend.TRITON
                )
            self.assertEqual(
                get_fp8_gemm_runner_backend(), Fp8GemmRunnerBackend.CUTLASS
            )


if __name__ == "__main__":
    unittest.main()
