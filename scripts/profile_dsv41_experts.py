#!/usr/bin/env python3
"""Capture a compact, broad DeepSeek-V4.1 expert-frequency profile.

The server must be launched with ``--expert-distribution-recorder-mode stat``.
This utility discards warmup routes, samples several representative domains,
and collapses the recorder's step history into a small [layers, experts] file
accepted by ``--kt-expert-frequency-file``.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


WHITE_GIF = (
    "data:image/gif;base64,"
    "R0lGODlhAQABAIAAAP///////yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
)

TEXT_PROMPTS = [
    "Explain tensor parallelism versus expert parallelism for a heterogeneous "
    "single-node mixture-of-experts inference server, including bottlenecks.",
    "Write a Python interval-merging function, explain half-open boundary cases, "
    "and include tests and a complexity analysis.",
    "Design a rollback-safe PostgreSQL migration for changing a busy table's "
    "integer status column to text without a long exclusive lock.",
    "Compare CRISPR base editing and prime editing for a pathogenic point "
    "mutation, including mechanisms, byproducts, and delivery constraints.",
    "Explain the economic effects of a central bank reducing its balance sheet "
    "while holding its policy interest rate constant.",
    "Prove that there are infinitely many prime numbers, then contrast Euclid's "
    "proof with a proof using Fermat numbers.",
    "Translate 'The deployment succeeded, but monitoring found a latency "
    "regression' into French, Japanese, and Spanish and explain nuances.",
    "Write a restrained science-fiction scene in which an engineer discovers "
    "that a navigation error is actually a message.",
    "Compare the causes and institutional consequences of the 1929 crash and "
    "the 2008 financial crisis without treating them as identical.",
    "Given a product experiment with conversion counts 1042/10000 and "
    "1108/10000, explain how to estimate uncertainty and decide whether to ship.",
    "Create a concise incident-response plan for intermittent packet loss across "
    "two data centers, including evidence collection and rollback criteria.",
]


def _post_json(url: str, payload: Any | None = None, timeout: float = 600) -> Any:
    body = b"" if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.decode(errors="replace")


def _recorder_files() -> set[Path]:
    return set(Path("/tmp").glob("expert_distribution_recorder_*.pt"))


def _profile_payload(model: str, content: Any, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="deepseek-v41-flash")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("/tmp/dsv41_profile.pt"))
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    before = _recorder_files()

    # Drop warmup routes and begin a clean capture. Stop may report that an
    # inactive recorder was already stopped; dump is still useful as a reset.
    _post_json(f"{base_url}/stop_expert_distribution_record")
    _post_json(f"{base_url}/dump_expert_distribution_record")
    before |= _recorder_files()
    _post_json(f"{base_url}/start_expert_distribution_record")

    requests: list[tuple[str, Any]] = [
        (f"text-{index:02d}", prompt) for index, prompt in enumerate(TEXT_PROMPTS, 1)
    ]
    requests.append(
        (
            "vision-01",
            [
                {"type": "image_url", "image_url": {"url": WHITE_GIF}},
                {
                    "type": "text",
                    "text": "Describe what is visible and state what cannot be inferred.",
                },
            ],
        )
    )

    total_completion_tokens = 0
    total_wall = 0.0
    try:
        for name, content in requests:
            started = time.perf_counter()
            result = _post_json(
                f"{base_url}/v1/chat/completions",
                _profile_payload(args.model, content, args.max_tokens),
            )
            elapsed = time.perf_counter() - started
            usage = result.get("usage") or {}
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_completion_tokens += completion_tokens
            total_wall += elapsed
            print(
                f"{name}: prompt={usage.get('prompt_tokens')} "
                f"completion={completion_tokens} wall={elapsed:.3f}s "
                f"completion_rate={completion_tokens / elapsed:.2f} tok/s",
                flush=True,
            )
    finally:
        _post_json(f"{base_url}/stop_expert_distribution_record")
        _post_json(f"{base_url}/dump_expert_distribution_record")

    created = _recorder_files() - before
    if not created:
        raise RuntimeError("Recorder dump did not create a new /tmp profile")
    source = max(created, key=lambda path: path.stat().st_mtime_ns)

    import torch

    loaded = torch.load(source, map_location="cpu", weights_only=True)
    logical_count = loaded["logical_count"]
    if logical_count.dim() == 3:
        logical_count = logical_count.sum(dim=0)
    compact = {
        "logical_count": logical_count,
        "source": "broad-dsv41-profile-v1",
        "requests": len(requests),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, args.output)
    print(
        f"saved {args.output} ({args.output.stat().st_size} bytes) from {source}; "
        f"requests={len(requests)} completion_tokens={total_completion_tokens} "
        f"wall={total_wall:.3f}s",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {detail}") from exc
