import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _server_args(**overrides):
    args = object.__new__(ServerArgs)
    args.nsa_prefill_backend = None
    args.nsa_decode_backend = None
    args.kv_cache_dtype = "fp8_e4m3"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestGlmSm120NsaBackendPolicy(unittest.TestCase):
    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_glm_sm120_fp8_defaults_to_trtllm_until_dependency_release(
        self, _mock_is_hip
    ):
        args = _server_args()

        args._set_default_nsa_backends(
            kv_cache_dtype="fp8_e4m3",
            major=12,
            model_arch="GlmMoeDsaForCausalLM",
        )

        self.assertEqual(args.nsa_prefill_backend, "trtllm")
        self.assertEqual(args.nsa_decode_backend, "trtllm")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_explicit_flashinfer_defaults_unspecified_side_to_flashinfer(
        self, _mock_is_hip
    ):
        args = _server_args(nsa_decode_backend="flashinfer_sparse_mla")

        args._set_default_nsa_backends(
            kv_cache_dtype="fp8_e4m3",
            major=12,
            model_arch="GlmMoeDsaForCausalLM",
        )

        self.assertEqual(args.nsa_prefill_backend, "flashinfer_sparse_mla")
        self.assertEqual(args.nsa_decode_backend, "flashinfer_sparse_mla")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_rejects_unsupported_backend_for_glm_sm120_fp8(self, _mock_is_hip):
        args = _server_args(nsa_prefill_backend="fa3")

        with self.assertRaisesRegex(ValueError, "supports only"):
            args._set_default_nsa_backends(
                kv_cache_dtype="fp8_e4m3",
                major=12,
                model_arch="GlmMoeDsaForCausalLM",
            )

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_rejects_mixed_trtllm_and_flashinfer_layouts(self, _mock_is_hip):
        args = _server_args(
            nsa_prefill_backend="trtllm",
            nsa_decode_backend="flashinfer_sparse_mla",
        )

        with self.assertRaisesRegex(ValueError, "same NSA backend"):
            args._set_default_nsa_backends(
                kv_cache_dtype="fp8_e4m3",
                major=12,
                model_arch="GlmMoeDsaForCausalLM",
            )

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_rejects_flashinfer_backend_for_non_glm_model(self, _mock_is_hip):
        args = _server_args(
            nsa_prefill_backend="flashinfer_sparse_mla",
            nsa_decode_backend="flashinfer_sparse_mla",
        )

        with self.assertRaisesRegex(ValueError, "GlmMoeDsaForCausalLM"):
            args._set_default_nsa_backends(
                kv_cache_dtype="fp8_e4m3",
                major=12,
                model_arch="DeepseekV3ForCausalLM",
            )


class _Runner(ModelRunnerKVCacheMixin):
    pass


class TestNsaFp8KvSizing(unittest.TestCase):
    @patch(
        "sglang.srt.model_executor.model_runner_kv_cache_mixin.is_deepseek_compressed",
        return_value=False,
    )
    @patch(
        "sglang.srt.model_executor.model_runner_kv_cache_mixin.is_deepseek_nsa",
        return_value=True,
    )
    @patch(
        "sglang.srt.model_executor.model_runner_kv_cache_mixin.get_nsa_index_head_dim",
        return_value=0,
    )
    def test_flashinfer_sparse_mla_uses_packed_656_byte_cache_dim(
        self, _mock_index_dim, _mock_is_nsa, _mock_is_compressed
    ):
        runner = _Runner()
        runner.use_mla_backend = True
        runner.kv_cache_dtype = torch.float8_e4m3fn
        runner.server_args = SimpleNamespace(
            nsa_prefill_backend="flashinfer_sparse_mla",
            nsa_decode_backend="flashinfer_sparse_mla",
        )
        runner.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(),
            kv_lora_rank=512,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
        )

        self.assertEqual(runner.calculate_mla_kv_cache_dim(), 656)
        self.assertEqual(runner.get_cell_size_per_token(num_layers=1), 656)


if __name__ == "__main__":
    unittest.main()
