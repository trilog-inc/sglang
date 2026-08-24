"""Tests for the pre-SM89 DSV4 cache dequantization dispatch."""

import unittest
from unittest.mock import MagicMock, patch

import torch

import sglang.kernels.ops.attention.dsv4.dequant_k_cache as dequant
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDsv4DequantSm8x(unittest.TestCase):
    @patch.object(dequant.torch.cuda, "get_device_capability", return_value=(8, 6))
    def test_sm86_rejects_native_fp8_triton_pointer(self, capability):
        with patch.object(dequant.torch.version, "hip", None):
            device = torch.device("cuda:1")
            self.assertFalse(dequant._can_use_native_fp8_triton(device))
        capability.assert_called_once_with(device)

    @patch.object(dequant.torch.cuda, "get_device_capability", return_value=(12, 0))
    def test_sm120_keeps_native_fp8_kernel(self, capability):
        with patch.object(dequant.torch.version, "hip", None):
            device = torch.device("cuda:0")
            self.assertTrue(dequant._can_use_native_fp8_triton(device))
        capability.assert_called_once_with(device)

    def test_sm86_dispatch_passes_only_byte_and_bf16_inputs(self):
        quant_cache = torch.zeros((1, 584), dtype=torch.uint8)
        page_table = torch.zeros(1, dtype=torch.int32)
        native_kernel = MagicMock()
        u8_kernel = MagicMock()
        u8_launch = MagicMock()
        u8_kernel.__getitem__.return_value = u8_launch

        dequant._get_fp8_bf16_lut.cache_clear()
        with (
            patch.object(dequant, "_can_use_native_fp8_triton", return_value=False),
            patch.object(dequant, "_dequantize_k_cache_paged_kernel", native_kernel),
            patch.object(dequant, "_dequantize_k_cache_paged_u8_kernel", u8_kernel),
        ):
            out = dequant.dequantize_k_cache_paged(quant_cache, page_table, page_size=1)

        self.assertEqual(out.shape, (1, 1, dequant.DIM_NOPE + dequant.DIM_ROPE))
        native_kernel.__getitem__.assert_not_called()
        u8_kernel.__getitem__.assert_called_once_with((1,))
        u8_launch.assert_called_once()
        args = u8_launch.call_args.args
        self.assertEqual(args[1].dtype, torch.bfloat16)
        self.assertEqual(args[2].dtype, torch.uint8)
        self.assertEqual(args[3].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
