"""Unit tests for the DSV4 DSpark SM120 sparse-attention width policy."""

import os
import unittest
from unittest.mock import patch

import sglang.srt.layers.attention.deepseek_v4_backend as dsv4_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDsv4DsparkSm120Policy(CustomTestCase):
    def test_flashinfer_uses_an_instantiated_dsv4_topk(self):
        with (
            patch.object(dsv4_backend, "_is_sm120", True),
            patch.dict(
                os.environ,
                {"SGLANG_SM120_FLASHMLA_BACKEND": "flashinfer"},
            ),
        ):
            # A six-token DSpark block needs 134 logical entries, naturally
            # aligned to 192. FlashInfer has no 192-wide DSV4 specialization.
            self.assertEqual(
                dsv4_backend._dspark_swa_page_index_alignment(block_size=6),
                512,
            )
            self.assertEqual(
                dsv4_backend._dspark_swa_page_index_alignment(block_size=384),
                512,
            )
            self.assertEqual(
                dsv4_backend._dspark_swa_page_index_alignment(block_size=385),
                1024,
            )
            with self.assertRaisesRegex(ValueError, "exceeds.*envelope"):
                dsv4_backend._dspark_swa_page_index_alignment(block_size=897)

    def test_other_backends_keep_compact_64_alignment(self):
        with (
            patch.object(dsv4_backend, "_is_sm120", True),
            patch.dict(
                os.environ,
                {"SGLANG_SM120_FLASHMLA_BACKEND": "triton"},
            ),
        ):
            self.assertEqual(
                dsv4_backend._dspark_swa_page_index_alignment(block_size=6),
                64,
            )


if __name__ == "__main__":
    unittest.main()
