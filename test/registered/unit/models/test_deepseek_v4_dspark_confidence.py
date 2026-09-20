"""Unit tests for DeepSeek-V4 DSpark confidence shape handling."""

import unittest

import torch
from torch import nn

from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.models.dspark import DSparkConfidenceHead
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepseekV4DsparkConfidence(CustomTestCase):
    def test_runtime_gamma_can_be_smaller_than_checkpoint_gamma(self):
        model = DeepseekV4ForCausalLMDSpark.__new__(
            DeepseekV4ForCausalLMDSpark
        )
        nn.Module.__init__(model)
        model.gamma = 5
        model.confidence_head = DSparkConfidenceHead(
            hidden_size=8,
            markov_rank=0,
            with_markov=False,
            bias=False,
        )

        confidence = model.compute_confidence(
            anchor_tokens=torch.tensor([1, 2]),
            sampled_tokens=torch.tensor([[3, 4, 5], [6, 7, 8]]),
            x_post_hc=torch.randn(2 * 3, 8),
        )

        self.assertEqual(confidence.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
