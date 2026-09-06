"""Exercise the chat client and metrics without importing the serving runtime."""

import ast
import json
import traceback
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_client():
    source = Path(__file__).parents[3] / "python/sglang/bench_serving.py"
    selected = {
        "RequestFuncInput",
        "RequestFuncOutput",
        "BenchmarkMetrics",
        "async_request_openai_chat_completions",
        "calculate_metrics",
    }
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in selected]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    namespace = {
        "dataclass": dataclass,
        "field": field,
        "np": np,
        "json": json,
        "traceback": traceback,
        "get_request_headers": lambda: {},
        "remove_prefix": lambda text, prefix: text.removeprefix(prefix),
        "args": SimpleNamespace(disable_stream=False, disable_ignore_eos=False),
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return SimpleNamespace(**namespace), namespace


class TestChatReasoning(unittest.IsolatedAsyncioTestCase):
    async def run_request(self, deltas, *, message=None):
        client, namespace = load_client()
        clock = SimpleNamespace(now=0.0)
        namespace["time"] = SimpleNamespace(perf_counter=lambda: clock.now)
        namespace["args"].disable_stream = message is not None

        class Response:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def json(self):
                clock.now = 5.0
                return {
                    "choices": [{"message": message}],
                    "usage": {"completion_tokens": 3},
                }

            @property
            def content(self):
                async def stream():
                    yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}'
                    for index, delta in enumerate(deltas, start=1):
                        clock.now = float(index)
                        yield (
                            "data: " + json.dumps({"choices": [{"delta": delta}]})
                        ).encode()
                    clock.now = 4.0
                    yield b'data: {"choices":[],"usage":{"completion_tokens":3}}'
                    clock.now = 5.0
                    yield b"data: [DONE]"

                return stream()

        class Session(Response):
            def post(self, **kwargs):
                return Response()

        namespace["_create_bench_client_session"] = Session
        request = client.RequestFuncInput(
            prompt="hello",
            api_url="http://localhost/v1/chat/completions",
            prompt_len=1,
            output_len=10,
            model="test",
            lora_name=None,
            image_data=None,
            extra_request_body={},
        )
        output = await client.async_request_openai_chat_completions(request)
        self.assertTrue(output.success, output.error)
        self.assertEqual(output.output_len, 3)
        return client, output

    async def test_reasoning_starts_timing_and_counts_toward_metrics(self):
        client, output = await self.run_request(
            [
                {"reasoning_content": "think ", "content": None},
                {"reasoning_content": "more"},
                {"content": "answer"},
            ]
        )
        self.assertEqual(output.reasoning_text, "think more")
        self.assertEqual(output.generated_text, "answer")
        self.assertEqual(output.ttft, 1.0)
        self.assertEqual(output.itl, [1.0, 1.0])
        self.assertEqual(output.text_chunks, ["more", "answer"])
        tokenizer = SimpleNamespace(encode=lambda text, **kw: text.split())
        metrics, _ = client.calculate_metrics(
            None, [output], 5.0, tokenizer, "sglang-oai-chat"
        )
        self.assertEqual(metrics.total_output_retokenized, 3)
        self.assertEqual(metrics.mean_ttft_ms, 1000)
        self.assertEqual(metrics.mean_tpot_ms, 2000)

    async def test_reasoning_only_completion(self):
        _, output = await self.run_request([{"reasoning_content": "thinking"}])
        self.assertEqual(output.generated_text, "")
        self.assertEqual(output.reasoning_text, "thinking")
        self.assertEqual(output.ttft, 1.0)

    async def test_content_only_completion(self):
        _, output = await self.run_request([{"content": "one"}, {"content": " two"}])
        self.assertEqual(output.generated_text, "one two")
        self.assertEqual(output.reasoning_text, "")
        self.assertEqual(output.itl, [1.0])

    async def test_both_channels_in_one_event_have_one_interval(self):
        _, output = await self.run_request(
            [
                {"reasoning_content": "one"},
                {"reasoning_content": " two", "content": "answer"},
            ]
        )
        self.assertEqual(output.itl, [1.0])
        self.assertEqual(output.text_chunks, [" twoanswer"])
        self.assertEqual(output.reasoning_text, "one two")
        self.assertEqual(output.generated_text, "answer")

    async def test_non_streaming_preserves_both_channels(self):
        _, output = await self.run_request(
            [], message={"reasoning_content": "thinking", "content": "answer"}
        )
        self.assertEqual(output.generated_text, "answer")
        self.assertEqual(output.reasoning_text, "thinking")
        self.assertEqual(output.ttft, 5.0)

    async def test_non_streaming_reasoning_only(self):
        _, output = await self.run_request(
            [], message={"reasoning_content": "thinking", "content": None}
        )
        self.assertEqual(output.generated_text, "")
        self.assertEqual(output.reasoning_text, "thinking")


if __name__ == "__main__":
    unittest.main()
