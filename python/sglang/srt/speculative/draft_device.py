"""CUDA-device helpers for a heterogeneous speculative draft worker."""

from __future__ import annotations

from contextlib import contextmanager

import torch


def _normalize_cuda_uuid(value: str) -> str:
    return value.strip().lower().removeprefix("gpu-").replace("-", "")


def resolve_speculative_draft_device(device: str) -> int:
    """Resolve a logical CUDA index, ``cuda:N``, or visible CUDA UUID."""

    raw = str(device).strip()
    has_cuda_prefix = raw.lower().startswith("cuda:")
    index_text = raw[5:] if has_cuda_prefix else raw
    # A UUID may happen to contain only decimal digits after punctuation is
    # removed.  Bare logical indices are necessarily short in practice;
    # ``cuda:N`` remains the unambiguous form for any index.
    if index_text.isdecimal() and (has_cuda_prefix or len(index_text) <= 3):
        index = int(index_text)
    else:
        wanted = _normalize_cuda_uuid(raw)
        index = -1
        for candidate in range(torch.cuda.device_count()):
            uuid = getattr(torch.cuda.get_device_properties(candidate), "uuid", None)
            if uuid is not None and _normalize_cuda_uuid(str(uuid)) == wanted:
                index = candidate
                break
        if index < 0:
            raise ValueError(
                "Could not resolve --speculative-draft-device "
                f"{device!r} to a visible CUDA device."
            )

    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            "--speculative-draft-device resolved to logical CUDA index "
            f"{index}, but only {torch.cuda.device_count()} CUDA devices are visible."
        )
    return index


@contextmanager
def draft_cuda_device_context(gpu_id: int):
    """Route implicit ``device='cuda'`` allocations to the draft GPU."""

    with torch.cuda.device(int(gpu_id)):
        yield


__all__ = ["draft_cuda_device_context", "resolve_speculative_draft_device"]
