"""CPU-only coverage for DeepSeek-V4-Vision span scheduling helpers."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[5]
VISIBLE_WINDOW_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "attention"
    / "dsv4"
    / "visible_window.py"
)


def _load_visible_window_module():
    spec = importlib.util.spec_from_file_location(
        "dsv4_visible_window_test_target", VISIBLE_WINDOW_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


visible_window = _load_visible_window_module()


def _mm_input(*spans, dsv4=True):
    item = SimpleNamespace(
        offsets=list(spans),
        model_specific_data={"types": [2], "perm": [0]} if dsv4 else {},
        is_image=lambda: True,
    )
    return SimpleNamespace(mm_items=[item])


class TestDsv4VisibleWindow(unittest.TestCase):
    def test_full_span_gets_bidirectional_window_and_padding(self):
        result = visible_window.compute_visible_window_overrides(
            mm_inputs=[_mm_input((10, 19))],
            extend_prefix_lens=[8],
            extend_seq_lens=[15],
            swa_window=4,
            padded_num_tokens=17,
        )

        self.assertIsNotNone(result)
        starts, lengths = result
        self.assertEqual(len(starts), 17)
        # The item offset starts at a compress-pad token. Position 10 keeps
        # its plain causal window; IMAGE_START is position 11 for this span.
        self.assertEqual((starts[2], lengths[2]), (7, 4))
        self.assertEqual((starts[3], lengths[3]), (8, 12))
        # The final image token retains the complete image span.
        self.assertEqual((starts[11], lengths[11]), (11, 9))
        self.assertEqual((starts[-2:], lengths[-2:]), ([0, 0], [1, 1]))

    def test_shallow_radix_hit_keeps_span_visible(self):
        result = visible_window.compute_visible_window_overrides(
            mm_inputs=[_mm_input((10, 19))],
            extend_prefix_lens=[13],
            extend_seq_lens=[7],
            swa_window=4,
            padded_num_tokens=7,
        )

        self.assertIsNotNone(result)
        starts, lengths = result
        self.assertEqual((starts[0], lengths[0]), (10, 10))
        self.assertEqual((starts[-1], lengths[-1]), (11, 9))

    def test_shallow_radix_hit_with_truncated_tail_degrades_safely(self):
        result = visible_window.compute_visible_window_overrides(
            mm_inputs=[_mm_input((10, 19))],
            extend_prefix_lens=[13],
            extend_seq_lens=[2],
            swa_window=4,
            padded_num_tokens=2,
        )

        self.assertIsNone(result)

    def test_deep_radix_hit_is_cut_back_to_span_start(self):
        mm_input = _mm_input((10, 19))

        self.assertIsNone(
            visible_window.image_span_cut_point(mm_input, position=13, swa_window=4)
        )
        self.assertEqual(
            visible_window.image_span_cut_point(mm_input, position=14, swa_window=4),
            10,
        )
        self.assertFalse(
            visible_window.has_visible_window_span(
                [mm_input], prefix_lens=[15], extend_lens=[5], swa_window=4
            )
        )

    def test_chunk_boundary_advances_to_span_end(self):
        mm_input = _mm_input((10, 19))

        self.assertEqual(
            visible_window.image_span_aligned_extend_end(mm_input, 11), 20
        )
        self.assertEqual(
            visible_window.image_span_aligned_extend_end(mm_input, 10), 10
        )
        self.assertEqual(
            visible_window.image_span_aligned_extend_end(mm_input, 20), 20
        )

    def test_non_dsv4_multimodal_items_are_unchanged(self):
        mm_input = _mm_input((10, 19), dsv4=False)

        self.assertEqual(
            visible_window.image_span_aligned_extend_end(mm_input, 11), 11
        )
        self.assertIsNone(
            visible_window.image_span_cut_point(mm_input, position=18, swa_window=4)
        )
        self.assertFalse(
            visible_window.has_visible_window_span(
                [mm_input], prefix_lens=[8], extend_lens=[15], swa_window=4
            )
        )


if __name__ == "__main__":
    unittest.main()
