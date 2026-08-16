import unittest
from unittest.mock import patch

from sglang.srt.layers.quantization.fp8_utils import can_auto_enable_marlin_fp8
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFp8MarlinCurrentDevice(unittest.TestCase):
    @patch(
        "sglang.srt.layers.quantization.fp8_utils.get_device_capability",
        return_value=(8, 6),
    )
    @patch(
        "sglang.srt.layers.quantization.fp8_utils.torch.cuda.current_device",
        return_value=1,
    )
    def test_sm86_draft_uses_its_active_device(self, current_device, capability):
        self.assertTrue(can_auto_enable_marlin_fp8())
        current_device.assert_called_once_with()
        capability.assert_called_once_with(1)

    @patch(
        "sglang.srt.layers.quantization.fp8_utils.get_device_capability",
        return_value=(12, 0),
    )
    @patch(
        "sglang.srt.layers.quantization.fp8_utils.torch.cuda.current_device",
        return_value=0,
    )
    def test_sm120_target_keeps_native_fp8(self, current_device, capability):
        self.assertFalse(can_auto_enable_marlin_fp8())
        current_device.assert_called_once_with()
        capability.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
