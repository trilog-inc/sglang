import unittest

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    _mask_logits_with_candidate_block_mask_,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    select_candidate_block_mask,
    select_candidate_blocks,
)


class TestDSV4CandidateBlockMask(unittest.TestCase):
    def test_compact_mask_matches_dense_mask(self):
        torch.manual_seed(123)
        for rows, width, block_size, topk_blocks in (
            (1, 1, 8, 2048),
            (3, 65, 8, 4),
            (7, 257, 8, 8),
            (4, 1024, 32, 5),
        ):
            logits = torch.randn(rows, width)
            lengths = torch.linspace(1, width, rows, dtype=torch.int64)[:, None]
            columns = torch.arange(width)
            logits.masked_fill_(columns[None, :] >= lengths, -torch.inf)

            compact = select_candidate_block_mask(
                logits, lengths, topk_blocks=topk_blocks, block_size=block_size
            )
            dense = select_candidate_blocks(
                logits, lengths, topk_blocks=topk_blocks, block_size=block_size
            )
            actual = _mask_logits_with_candidate_block_mask_(
                logits.clone(), compact, block_size
            )
            expected = logits.masked_fill(~dense, -torch.inf)

            self.assertEqual(
                compact.shape[1], (width + block_size - 1) // block_size
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
