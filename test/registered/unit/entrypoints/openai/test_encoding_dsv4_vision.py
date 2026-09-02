"""Dependency-free coverage for DeepSeek-V4 vision message encoding."""

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
ENCODING_PATH = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "entrypoints"
    / "openai"
    / "encoding_dsv4.py"
)


def _load_encoding_module():
    spec = importlib.util.spec_from_file_location(
        "dsv4_vision_encoding_test_target", ENCODING_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


encoding_dsv4 = _load_encoding_module()


class TestDsv4VisionEncoding(unittest.TestCase):
    def test_tool_arguments_accept_json_string_or_normalized_dict(self):
        arguments = {"query": "weather", "limit": 3, "units": ["C", "F"]}

        from_json = encoding_dsv4.encode_arguments_to_dsml(
            {"arguments": '{"query":"weather","limit":3,"units":["C","F"]}'}
        )
        from_dict = encoding_dsv4.encode_arguments_to_dsml(
            {"arguments": arguments}
        )

        self.assertEqual(from_json, from_dict)
        self.assertIn('name="query" string="true">weather<', from_dict)
        self.assertIn('name="limit" string="false">3<', from_dict)
        self.assertIn('name="units" string="false">["C", "F"]<', from_dict)

    def test_tool_arguments_reject_invalid_json(self):
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            encoding_dsv4.encode_arguments_to_dsml({"arguments": "{bad json"})

    def test_tool_arguments_require_an_object(self):
        with self.assertRaisesRegex(ValueError, "JSON object/dict"):
            encoding_dsv4.encode_arguments_to_dsml({"arguments": "[1, 2, 3]"})

    def test_text_only_api_still_returns_a_string(self):
        prompt = encoding_dsv4.encode_messages(
            [{"role": "user", "content": "hello"}], thinking_mode="chat"
        )

        self.assertIsInstance(prompt, str)
        self.assertIn("<｜User｜>hello", prompt)

    def test_image_url_and_input_image_preserve_prompt_order(self):
        prompt, media = encoding_dsv4.encode_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAA"},
                        },
                        {"type": "text", "text": "second"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.invalid/image.png",
                        },
                    ],
                }
            ],
            thinking_mode="chat",
            return_multi_modal_data=True,
        )

        self.assertEqual(prompt.count(encoding_dsv4.IMAGE_PLACEHOLDER), 2)
        self.assertEqual(
            [record["url"] for record in media["images"]],
            [
                "data:image/png;base64,AAA",
                "https://example.invalid/image.png",
            ],
        )

    def test_text_only_history_may_contain_a_literal_placeholder(self):
        prompt = encoding_dsv4.encode_messages(
            [
                {
                    "role": "assistant",
                    "content": f"prior {encoding_dsv4.IMAGE_PLACEHOLDER}",
                },
                {"role": "user", "content": "continue without an image"},
            ],
            thinking_mode="chat",
        )

        self.assertIn(encoding_dsv4.IMAGE_PLACEHOLDER, prompt)

    def test_multimodal_literal_placeholder_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "image special token"):
            encoding_dsv4.encode_messages(
                [
                    {
                        "role": "user",
                        "content": f"bad {encoding_dsv4.IMAGE_PLACEHOLDER}",
                    }
                ],
                thinking_mode="chat",
                return_multi_modal_data=True,
            )


if __name__ == "__main__":
    unittest.main()
