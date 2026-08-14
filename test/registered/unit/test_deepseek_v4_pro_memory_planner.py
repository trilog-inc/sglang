from __future__ import annotations

import importlib.util
import json
import struct
import sys
import unittest
from pathlib import Path


_SCRIPT = (
    Path(__file__).parents[3] / "scripts" / "deepseek_v4_pro_memory_planner.py"
)
_SPEC = importlib.util.spec_from_file_location("deepseek_v4_pro_memory_planner", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
planner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = planner
_SPEC.loader.exec_module(planner)


def _write_safetensor(path: Path, tensors: dict[str, tuple[str, list[int], int]]) -> None:
    offset = 0
    header = {}
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def test_scanner_reads_metadata_without_safetensors_dependency(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 2}), encoding="utf-8"
    )
    _write_safetensor(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "layers.0.ffn.experts.0.w1.weight": ("U8", [2, 2], 4),
            "layers.0.ffn.experts.0.w1.scale": ("F8_E8M0", [2, 1], 2),
            "layers.0.attention.q_proj.weight": ("F8_E4M3", [2, 2], 4),
            "layers.0.ffn.shared_experts.w1.weight": ("F8_E4M3", [2, 2], 4),
            "layers.2.ffn.experts.0.w1.weight": ("U8", [2, 2], 4),
            "model.embed_tokens.weight": ("BF16", [2, 2], 8),
        },
    )

    audit = planner.scan_checkpoint(tmp_path)

    assert audit.payload_bytes == 26
    assert audit.category_bytes == {
        "attention": 4,
        "embeddings": 8,
        "mtp": 4,
        "routed_experts": 6,
        "shared_experts": 4,
    }
    assert audit.expert_bytes == {(0, 0): 6}
    assert not audit.warnings


def test_scanner_rejects_non_native_routed_scale(tmp_path: Path) -> None:
    _write_safetensor(
        tmp_path / "model.safetensors",
        {"layers.0.ffn.experts.0.w1.scale": ("F32", [1], 4)},
    )

    audit = planner.scan_checkpoint(tmp_path)

    assert "not native one-byte UE8M0" in audit.warnings[0]


def test_placement_applies_reserves_and_assigns_remaining_experts_to_host(
    tmp_path: Path,
) -> None:
    tensors = {"model.embed_tokens.weight": ("U8", [10], 10)}
    for layer in range(2):
        for expert in range(4):
            tensors[f"layers.{layer}.ffn.experts.{expert}.w1.weight"] = (
                "U8",
                [100],
                100,
            )
    _write_safetensor(tmp_path / "model.safetensors", tensors)
    audit = planner.scan_checkpoint(tmp_path)
    devices = [
        planner.DevicePlan("primary", 1_000, 100, 10, 1, "flashinfer_mxfp4"),
        planner.DevicePlan("helper", 1_000, 100, 0, 1, "marlin_mxfp4"),
    ]

    usages, diagnostics = planner.build_placement(
        audit,
        host_total_bytes=1_000,
        host_reserve_bytes=100,
        host_runtime_bytes=0,
        devices=devices,
        allocator_overhead_fraction=0,
    )

    assert not diagnostics
    assert usages[0].routed_expert_bytes == 400
    assert usages[0].expert_count == 4
    assert usages[1].routed_expert_bytes == 200
    assert usages[1].layer_experts == {0: [0], 1: [0]}
    assert usages[1].non_expert_bytes == 10
    assert usages[2].routed_expert_bytes == 200
    assert all(usage.passes for usage in usages)


def test_placement_rejects_more_gpu_experts_than_checkpoint(tmp_path: Path) -> None:
    _write_safetensor(
        tmp_path / "model.safetensors",
        {"layers.0.ffn.experts.0.w1.weight": ("U8", [8], 8)},
    )
    audit = planner.scan_checkpoint(tmp_path)

    with unittest.TestCase().assertRaisesRegex(ValueError, "require 2 experts"):
        planner.build_placement(
            audit,
            host_total_bytes=100,
            host_reserve_bytes=1,
            host_runtime_bytes=0,
            devices=[planner.DevicePlan("primary", 100, 1, 0, 2, "mxfp4")],
            allocator_overhead_fraction=0,
        )


def test_native_pro_expert_bank_size_matches_strategy() -> None:
    hidden_size = 7_168
    intermediate_size = 3_072
    routed_layers = 61
    experts_per_layer = 384

    bytes_per_expert = (
        3 * hidden_size * intermediate_size // 2
        + 3 * hidden_size * intermediate_size // 32
    )
    resident_bytes = bytes_per_expert * routed_layers * experts_per_layer

    assert bytes_per_expert == 35_094_528
    assert 765.5 < resident_bytes / planner.GIB < 765.7


def test_go_gate_rejects_metadata_warnings(tmp_path: Path) -> None:
    _write_safetensor(
        tmp_path / "model.safetensors",
        {"layers.0.ffn.experts.0.w1.scale": ("F32", [1], 4)},
    )
    audit = planner.scan_checkpoint(tmp_path)
    usages, diagnostics = planner.build_placement(
        audit,
        host_total_bytes=100,
        host_reserve_bytes=1,
        host_runtime_bytes=0,
        devices=[planner.DevicePlan("primary", 100, 1, 0, 0, "mxfp4")],
        allocator_overhead_fraction=0,
    )

    assert all(usage.passes for usage in usages)
    assert not diagnostics
    assert planner._json_result(audit, usages, diagnostics)["go"] is False
