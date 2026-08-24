"""Unit tests for model-specific server warmup inputs."""

import base64
import struct
import unittest

from sglang.srt.entrypoints.http_server import (
    KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64,
    KIMI_VLM_WARMUP_PNG_PICTURE_BASE64,
    MINIMUM_PNG_PICTURE_BASE64,
    _get_server_warmup_max_new_tokens,
    _get_vlm_warmup_image_base64,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestVlmWarmupImage(CustomTestCase):
    def test_kimi_k2_uses_representative_vision_image(self):
        image_base64 = _get_vlm_warmup_image_base64(
            {"architectures": ["KimiK25ForConditionalGeneration"]}
        )
        self.assertEqual(image_base64, KIMI_VLM_WARMUP_PNG_PICTURE_BASE64)

        png = base64.b64decode(KIMI_VLM_WARMUP_PNG_PICTURE_BASE64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (512, 512))

    def test_kimi_k3_uses_native_patch_grid_image(self):
        for model_info in (
            {"architectures": ["KimiK3ForConditionalGeneration"]},
            {"architectures": None, "model_type": "kimi_k3"},
        ):
            with self.subTest(model_info=model_info):
                self.assertEqual(
                    _get_vlm_warmup_image_base64(model_info),
                    KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64,
                )

        png = base64.b64decode(KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (448, 448))

    def test_other_vlms_keep_minimal_startup_image(self):
        self.assertEqual(
            _get_vlm_warmup_image_base64(
                {"architectures": ["Qwen3VLForConditionalGeneration"]}
            ),
            MINIMUM_PNG_PICTURE_BASE64,
        )
        self.assertEqual(
            _get_vlm_warmup_image_base64({"architectures": None}),
            MINIMUM_PNG_PICTURE_BASE64,
        )


class TestServerWarmupTokens(CustomTestCase):
    def test_generation_token_count_can_be_reduced(self):
        with envs.SGLANG_WARMUP_MAX_NEW_TOKENS.override(2):
            self.assertEqual(_get_server_warmup_max_new_tokens(True), 2)

    def test_non_generation_warmup_remains_one(self):
        with envs.SGLANG_WARMUP_MAX_NEW_TOKENS.override(7):
            self.assertEqual(_get_server_warmup_max_new_tokens(False), 1)

    def test_generation_token_count_must_be_positive(self):
        with envs.SGLANG_WARMUP_MAX_NEW_TOKENS.override(0):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                _get_server_warmup_max_new_tokens(True)


if __name__ == "__main__":
    unittest.main()
