#!/usr/bin/env python3
"""Metadata-only capacity planner for DeepSeek-V4-Pro.

The planner reads only the JSON headers of safetensor shards.  It never maps or
loads tensor payloads, which makes it safe to run before attempting to load the
865+ GB checkpoint.  Routed experts are assigned statically, in logical expert
order, to the configured GPU tiers; the remainder stay in native packed MXFP4
form in host memory.

This is intentionally a capacity gate, not a performance model.  Runtime
workspaces and KV cache are represented by explicit reserve/runtime arguments
instead of optimistic guesses hidden in the calculation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


GIB = 1 << 30
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_EXPERT_RE = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    shard: str
    category: str
    layer: int | None = None
    expert: int | None = None


@dataclass(frozen=True)
class DevicePlan:
    name: str
    total_bytes: int
    reserve_bytes: int
    runtime_bytes: int
    experts_per_layer: int
    backend: str
    mtp_fraction: float = 0.0


@dataclass
class TierUsage:
    name: str
    representation: str
    capacity_bytes: int
    reserve_bytes: int
    runtime_bytes: int
    weight_bytes: int = 0
    routed_expert_bytes: int = 0
    non_expert_bytes: int = 0
    mtp_bytes: int = 0
    expert_count: int = 0
    layer_experts: dict[int, list[int]] = field(default_factory=dict)

    @property
    def used_bytes(self) -> int:
        return self.weight_bytes + self.runtime_bytes

    @property
    def headroom_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    @property
    def passes(self) -> bool:
        return self.headroom_bytes >= self.reserve_bytes


@dataclass(frozen=True)
class AuditResult:
    tensors: tuple[TensorMetadata, ...]
    category_bytes: Mapping[str, int]
    layer_bytes: Mapping[int, int]
    expert_bytes: Mapping[tuple[int, int], int]
    payload_bytes: int
    shard_file_bytes: int
    warnings: tuple[str, ...]


def _gib(value: float) -> int:
    if value < 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("GiB values must be finite and non-negative")
    return int(value * GIB)


def _layer_and_expert(name: str) -> tuple[int | None, int | None]:
    layer_match = _LAYER_RE.search(name)
    expert_match = _EXPERT_RE.search(name)
    return (
        int(layer_match.group(1)) if layer_match else None,
        int(expert_match.group(1)) if expert_match else None,
    )


def categorize_tensor(
    name: str,
    *,
    num_hidden_layers: int | None = None,
) -> tuple[str, int | None, int | None]:
    """Return the strategy audit category and optional layer/expert IDs."""

    lower = name.lower()
    layer, expert = _layer_and_expert(lower)
    if (
        "mtp" in lower
        or "nextn" in lower
        or (
            layer is not None
            and num_hidden_layers is not None
            and layer >= num_hidden_layers
        )
    ):
        return "mtp", layer, expert
    if "shared_expert" in lower or "shared_experts" in lower:
        return "shared_experts", layer, None
    if expert is not None:
        return "routed_experts", layer, expert
    if "indexer" in lower:
        return "indexer", layer, None
    if any(token in lower for token in ("self_attn", ".attention.", ".attn.", ".mla.")):
        return "attention", layer, None
    if any(
        token in lower
        for token in ("embed_tokens", "word_embeddings", "tok_embeddings")
    ):
        return "embeddings", layer, None
    if any(token in lower for token in ("lm_head", "output_layer", "output.weight")):
        return "heads", layer, None
    return "miscellaneous", layer, None


def _read_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return {}
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {config_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return value


def _read_safetensor_header(path: Path) -> tuple[dict, int]:
    file_size = path.stat().st_size
    if file_size < 8:
        raise ValueError(f"{path} is too small to be a safetensor shard")
    with path.open("rb", buffering=0) as handle:
        prefix = handle.read(8)
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size == 0 or header_size > file_size - 8:
            raise ValueError(
                f"{path} has invalid safetensor header length {header_size} "
                f"for a {file_size}-byte file"
            )
        header = handle.read(header_size)
    try:
        value = json.loads(header)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} has an invalid safetensor JSON header: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} safetensor header must be a JSON object")
    return value, file_size - 8 - header_size


def scan_checkpoint(model_dir: Path) -> AuditResult:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"No *.safetensors shards found in {model_dir}")

    config = _read_config(model_dir)
    num_hidden_layers_value = config.get("num_hidden_layers")
    num_hidden_layers = (
        int(num_hidden_layers_value)
        if isinstance(num_hidden_layers_value, (int, float))
        else None
    )
    tensors: list[TensorMetadata] = []
    category_bytes: defaultdict[str, int] = defaultdict(int)
    layer_bytes: defaultdict[int, int] = defaultdict(int)
    expert_bytes: defaultdict[tuple[int, int], int] = defaultdict(int)
    warnings: list[str] = []
    seen_names: set[str] = set()
    shard_file_bytes = 0

    for shard in shards:
        shard_file_bytes += shard.stat().st_size
        header, data_region_bytes = _read_safetensor_header(shard)
        for name, raw_info in header.items():
            if name == "__metadata__":
                continue
            if name in seen_names:
                raise ValueError(f"Tensor {name!r} occurs in more than one shard")
            seen_names.add(name)
            if not isinstance(raw_info, dict):
                raise ValueError(
                    f"Tensor metadata for {name!r} in {shard} is not an object"
                )
            offsets = raw_info.get("data_offsets")
            shape = raw_info.get("shape")
            dtype = raw_info.get("dtype")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(item, int) for item in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
                or offsets[1] > data_region_bytes
            ):
                raise ValueError(f"Invalid data_offsets for tensor {name!r} in {shard}")
            if not isinstance(shape, list) or not all(
                isinstance(item, int) and item >= 0 for item in shape
            ):
                raise ValueError(f"Invalid shape for tensor {name!r} in {shard}")
            if not isinstance(dtype, str):
                raise ValueError(f"Invalid dtype for tensor {name!r} in {shard}")
            nbytes = offsets[1] - offsets[0]
            category, layer, expert = categorize_tensor(
                name, num_hidden_layers=num_hidden_layers
            )
            item = TensorMetadata(
                name=name,
                dtype=dtype,
                shape=tuple(shape),
                nbytes=nbytes,
                shard=shard.name,
                category=category,
                layer=layer,
                expert=expert,
            )
            tensors.append(item)
            category_bytes[category] += nbytes
            if layer is not None:
                layer_bytes[layer] += nbytes
            if category == "routed_experts":
                if layer is None or expert is None:
                    warnings.append(f"Could not resolve layer/expert IDs for {name}")
                else:
                    expert_bytes[(layer, expert)] += nbytes
                native_ue8m0_dtype = dtype in {"U8", "I8"} or dtype.startswith(
                    "F8_E8M0"
                )
                if "scale" in name.lower() and not native_ue8m0_dtype:
                    warnings.append(
                        f"Routed-expert scale {name} uses {dtype}, not native one-byte UE8M0"
                    )

    if not expert_bytes:
        warnings.append("No routed-expert tensors were recognized")
    else:
        expected_experts = config.get("n_routed_experts")
        if isinstance(expected_experts, int) and expected_experts > 0:
            experts_by_layer: defaultdict[int, set[int]] = defaultdict(set)
            for layer, expert in expert_bytes:
                experts_by_layer[layer].add(expert)
            for layer, expert_ids in sorted(experts_by_layer.items()):
                if len(expert_ids) != expected_experts:
                    warnings.append(
                        f"Layer {layer} contains {len(expert_ids)} routed experts; "
                        f"config.json declares {expected_experts}"
                    )

        hidden_size = config.get("hidden_size")
        intermediate_size = config.get("moe_intermediate_size")
        if (
            isinstance(hidden_size, int)
            and hidden_size > 0
            and isinstance(intermediate_size, int)
            and intermediate_size > 0
        ):
            expected_native_bytes = (
                3 * hidden_size * intermediate_size // 2
                + 3 * hidden_size * intermediate_size // 32
            )
            mismatches = [
                (layer, expert, nbytes)
                for (layer, expert), nbytes in expert_bytes.items()
                if nbytes != expected_native_bytes
            ]
            if mismatches:
                layer, expert, actual = mismatches[0]
                warnings.append(
                    f"Routed expert ({layer}, {expert}) occupies {actual} bytes; "
                    f"the config-derived native E2M1+UE8M0 estimate is {expected_native_bytes} "
                    f"({len(mismatches)} mismatches total)"
                )
    payload_bytes = sum(item.nbytes for item in tensors)
    return AuditResult(
        tensors=tuple(tensors),
        category_bytes=dict(sorted(category_bytes.items())),
        layer_bytes=dict(sorted(layer_bytes.items())),
        expert_bytes=dict(sorted(expert_bytes.items())),
        payload_bytes=payload_bytes,
        shard_file_bytes=shard_file_bytes,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def build_placement(
    audit: AuditResult,
    *,
    host_total_bytes: int,
    host_reserve_bytes: int,
    host_runtime_bytes: int,
    devices: Sequence[DevicePlan],
    allocator_overhead_fraction: float,
    excluded_categories: frozenset[str] = frozenset(),
) -> tuple[list[TierUsage], list[str]]:
    if not devices:
        raise ValueError("At least one GPU device must be configured")
    if allocator_overhead_fraction < 0:
        raise ValueError("Allocator overhead cannot be negative")
    if "routed_experts" in excluded_categories:
        raise ValueError("Routed experts cannot be excluded from placement")

    mtp_fraction_total = sum(device.mtp_fraction for device in devices)
    if any(device.mtp_fraction < 0 for device in devices):
        raise ValueError("GPU MTP fractions cannot be negative")
    if mtp_fraction_total and not math.isclose(
        mtp_fraction_total, 1.0, rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError("Nonzero GPU MTP fractions must sum to 1")
    if mtp_fraction_total and "mtp" in excluded_categories:
        raise ValueError("MTP cannot be both excluded and assigned to GPUs")

    overhead = 1.0 + allocator_overhead_fraction
    usages = [
        TierUsage(
            name="host",
            representation="native_mxfp4_e2m1_ue8m0",
            capacity_bytes=host_total_bytes,
            reserve_bytes=host_reserve_bytes,
            runtime_bytes=host_runtime_bytes,
        )
    ]
    usages.extend(
        TierUsage(
            name=device.name,
            representation=(
                "primary_dense_plus_" + device.backend if index == 0 else device.backend
            ),
            capacity_bytes=device.total_bytes,
            reserve_bytes=device.reserve_bytes,
            runtime_bytes=device.runtime_bytes,
        )
        for index, device in enumerate(devices)
    )

    routed_expert_bytes = audit.category_bytes.get("routed_experts", 0)
    mtp_bytes = audit.category_bytes.get("mtp", 0)
    other_excluded_bytes = sum(
        audit.category_bytes.get(category, 0)
        for category in excluded_categories
        if category != "mtp"
    )
    primary_non_expert_raw_bytes = (
        audit.payload_bytes - routed_expert_bytes - mtp_bytes - other_excluded_bytes
    )
    primary_non_expert = math.ceil(primary_non_expert_raw_bytes * overhead)
    usages[1].non_expert_bytes = primary_non_expert
    usages[1].weight_bytes += primary_non_expert

    if "mtp" not in excluded_categories:
        placed_mtp_bytes = math.ceil(mtp_bytes * overhead)
        if mtp_fraction_total:
            shard_indices = [
                index for index, device in enumerate(devices) if device.mtp_fraction > 0
            ]
            remaining_mtp_bytes = placed_mtp_bytes
            for position, device_index in enumerate(shard_indices):
                if position == len(shard_indices) - 1:
                    shard_bytes = remaining_mtp_bytes
                else:
                    shard_bytes = math.floor(
                        placed_mtp_bytes * devices[device_index].mtp_fraction
                    )
                    remaining_mtp_bytes -= shard_bytes
                usage = usages[device_index + 1]
                usage.mtp_bytes += shard_bytes
                usage.weight_bytes += shard_bytes
                usage.representation += "+mtp_shard"
        else:
            usages[1].mtp_bytes = placed_mtp_bytes
            usages[1].weight_bytes += placed_mtp_bytes

    by_layer: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for (layer, expert), nbytes in audit.expert_bytes.items():
        by_layer[layer].append((expert, nbytes))

    diagnostics: list[str] = []
    expected_layers: set[int] | None = None
    for layer, experts in sorted(by_layer.items()):
        experts.sort()
        layer_expert_ids = {expert for expert, _ in experts}
        if expected_layers is None:
            expected_layers = layer_expert_ids
        elif layer_expert_ids != expected_layers:
            diagnostics.append(
                f"Layer {layer} has {len(layer_expert_ids)} routed experts; "
                f"the first routed layer has {len(expected_layers)}"
            )
        offset = 0
        for device_index, device in enumerate(devices):
            end = offset + device.experts_per_layer
            if end > len(experts):
                raise ValueError(
                    f"GPU expert counts require {end} experts in layer {layer}, "
                    f"but metadata contains {len(experts)}"
                )
            selected = experts[offset:end]
            raw_bytes = sum(nbytes for _, nbytes in selected)
            placed_bytes = math.ceil(raw_bytes * overhead)
            usage = usages[device_index + 1]
            usage.routed_expert_bytes += placed_bytes
            usage.weight_bytes += placed_bytes
            usage.expert_count += len(selected)
            usage.layer_experts[layer] = [expert for expert, _ in selected]
            offset = end
        host_experts = experts[offset:]
        host_raw_bytes = sum(nbytes for _, nbytes in host_experts)
        host_placed_bytes = math.ceil(host_raw_bytes * overhead)
        usages[0].routed_expert_bytes += host_placed_bytes
        usages[0].weight_bytes += host_placed_bytes
        usages[0].expert_count += len(host_experts)
        usages[0].layer_experts[layer] = [expert for expert, _ in host_experts]

    if not by_layer:
        diagnostics.append("Placement contains no recognized routed-expert layers")
    return usages, diagnostics


def _parse_csv_numbers(value: str, *, cast, option: str) -> list:
    try:
        result = [cast(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid {option}: {value!r}") from exc
    if not result:
        raise argparse.ArgumentTypeError(f"{option} cannot be empty")
    return result


def _same_length(parser: argparse.ArgumentParser, **values: Sequence) -> None:
    lengths = {key: len(value) for key, value in values.items()}
    if len(set(lengths.values())) != 1:
        parser.error(
            "GPU list arguments must have the same number of entries: "
            + ", ".join(f"{key}={length}" for key, length in lengths.items())
        )


def _human_bytes(value: int) -> str:
    return f"{value / GIB:,.2f} GiB"


def _json_result(
    audit: AuditResult,
    usages: Sequence[TierUsage],
    diagnostics: Sequence[str],
    excluded_category_bytes: Mapping[str, int] | None = None,
) -> dict:
    excluded_category_bytes = excluded_category_bytes or {}
    return {
        "checkpoint": {
            "tensor_payload_bytes": audit.payload_bytes,
            "placement_payload_bytes": audit.payload_bytes
            - sum(excluded_category_bytes.values()),
            "safetensor_file_bytes": audit.shard_file_bytes,
            "tensor_count": len(audit.tensors),
            "category_bytes": dict(audit.category_bytes),
            "layer_bytes": {
                str(key): value for key, value in audit.layer_bytes.items()
            },
            "routed_expert_instances": len(audit.expert_bytes),
            "routed_expert_bytes_min": min(audit.expert_bytes.values(), default=0),
            "routed_expert_bytes_max": max(audit.expert_bytes.values(), default=0),
        },
        "placement_exclusions": dict(excluded_category_bytes),
        "placement": [
            {
                **asdict(usage),
                "used_bytes": usage.used_bytes,
                "headroom_bytes": usage.headroom_bytes,
                "passes": usage.passes,
            }
            for usage in usages
        ],
        "warnings": list(audit.warnings),
        "diagnostics": list(diagnostics),
        "go": (
            bool(usages)
            and all(usage.passes for usage in usages)
            and not audit.warnings
            and not diagnostics
        ),
    }


def _print_report(
    audit: AuditResult,
    usages: Sequence[TierUsage],
    diagnostics: Sequence[str],
    *,
    verbose_layers: bool,
    excluded_category_bytes: Mapping[str, int] | None = None,
) -> None:
    excluded_category_bytes = excluded_category_bytes or {}
    print("DeepSeek-V4-Pro metadata audit")
    print(f"  tensor payload:   {_human_bytes(audit.payload_bytes)}")
    print(f"  shard file bytes: {_human_bytes(audit.shard_file_bytes)}")
    print(f"  tensors:          {len(audit.tensors):,}")
    print(f"  routed experts:   {len(audit.expert_bytes):,}")
    print("\nTensor categories")
    for category, nbytes in audit.category_bytes.items():
        print(f"  {category:20s} {_human_bytes(nbytes):>14s}")

    if excluded_category_bytes:
        print("\nExcluded from placement")
        for category, nbytes in excluded_category_bytes.items():
            print(f"  {category:20s} {_human_bytes(nbytes):>14s}")
        placement_payload = audit.payload_bytes - sum(excluded_category_bytes.values())
        print(f"  {'placement payload':20s} {_human_bytes(placement_payload):>14s}")

    if verbose_layers:
        print("\nPer-layer native payload")
        for layer, nbytes in audit.layer_bytes.items():
            print(f"  layer {layer:3d} {_human_bytes(nbytes):>14s}")

        placement_layers = sorted(
            {layer for usage in usages for layer in usage.layer_experts}
        )
        if placement_layers:
            print("\nPer-layer routed-expert placement")
            for layer in placement_layers:
                counts = ", ".join(
                    f"{usage.name}={len(usage.layer_experts.get(layer, []))}"
                    for usage in usages
                )
                print(f"  layer {layer:3d}: {counts}")

    print("\nProposed steady-state placement")
    print(
        "  tier                 experts      weights     runtime     headroom     reserve   result"
    )
    for usage in usages:
        print(
            f"  {usage.name:20.20s} {usage.expert_count:7,d} "
            f"{_human_bytes(usage.weight_bytes):>11s} "
            f"{_human_bytes(usage.runtime_bytes):>11s} "
            f"{_human_bytes(usage.headroom_bytes):>12s} "
            f"{_human_bytes(usage.reserve_bytes):>11s}   "
            f"{'PASS' if usage.passes else 'FAIL'}"
        )
        print(f"    representation: {usage.representation}")
        if usage.mtp_bytes:
            print(f"    MTP shard: {_human_bytes(usage.mtp_bytes)}")

    messages = [*audit.warnings, *diagnostics]
    if messages:
        print("\nDiagnostics")
        for message in messages:
            print(f"  - {message}")
    go = (
        bool(usages)
        and all(usage.passes for usage in usages)
        and not audit.warnings
        and not diagnostics
    )
    print("\nGO" if go else "\nNO-GO")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="Directory containing config.json and safetensor shards",
    )
    parser.add_argument("--host-ram-gib", type=float, default=768.0)
    parser.add_argument("--host-reserve-gib", type=float, default=64.0)
    parser.add_argument("--host-runtime-gib", type=float, default=0.0)
    parser.add_argument(
        "--gpu-names",
        default="rtx-pro-6000,rtx-4090,rtx-3090-1,rtx-3090-2",
    )
    parser.add_argument("--gpu-capacities-gib", default="96,24,24,24")
    parser.add_argument("--gpu-reserves-gib", default="12,3,3,3")
    parser.add_argument(
        "--gpu-runtime-gib",
        default="5,0,0,0",
        help="Non-weight runtime use, including the primary KV cache",
    )
    parser.add_argument("--gpu-experts-per-layer", default="18,9,9,9")
    parser.add_argument(
        "--gpu-backends",
        default="flashinfer_mxfp4,marlin_mxfp4,marlin_mxfp4,marlin_mxfp4",
    )
    parser.add_argument(
        "--gpu-mtp-fractions",
        help=(
            "Comma-separated fraction of MTP/NextN bytes assigned to each GPU. "
            "Use all zeros for primary-GPU placement or fractions summing to 1 "
            "for an explicit shard plan, for example 0,0,0.5,0.5."
        ),
    )
    parser.add_argument(
        "--allocator-overhead-percent",
        type=float,
        default=0.5,
        help="Representation/alignment overhead applied to all weight placements",
    )
    parser.add_argument(
        "--exclude-mtp",
        action="store_true",
        help=(
            "Exclude MTP/NextN tensors from placement while retaining them in "
            "the checkpoint audit. Use only for target-only inference, whose "
            "model loader skips those draft weights."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    parser.add_argument(
        "--json-output", type=Path, help="Also write the JSON report to this path"
    )
    parser.add_argument("--verbose-layers", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.allocator_overhead_percent < 0:
        parser.error("--allocator-overhead-percent cannot be negative")

    names = _parse_csv_numbers(args.gpu_names, cast=str, option="--gpu-names")
    capacities = _parse_csv_numbers(
        args.gpu_capacities_gib, cast=float, option="--gpu-capacities-gib"
    )
    reserves = _parse_csv_numbers(
        args.gpu_reserves_gib, cast=float, option="--gpu-reserves-gib"
    )
    runtimes = _parse_csv_numbers(
        args.gpu_runtime_gib, cast=float, option="--gpu-runtime-gib"
    )
    expert_counts = _parse_csv_numbers(
        args.gpu_experts_per_layer,
        cast=int,
        option="--gpu-experts-per-layer",
    )
    backends = _parse_csv_numbers(args.gpu_backends, cast=str, option="--gpu-backends")
    mtp_fractions = (
        _parse_csv_numbers(
            args.gpu_mtp_fractions,
            cast=float,
            option="--gpu-mtp-fractions",
        )
        if args.gpu_mtp_fractions is not None
        else [0.0] * len(names)
    )
    _same_length(
        parser,
        names=names,
        capacities=capacities,
        reserves=reserves,
        runtimes=runtimes,
        expert_counts=expert_counts,
        backends=backends,
        mtp_fractions=mtp_fractions,
    )
    if any(
        value < 0
        for value in [
            *capacities,
            *reserves,
            *runtimes,
            *expert_counts,
            *mtp_fractions,
        ]
    ):
        parser.error(
            "GPU capacities, reserves, runtime use, expert counts, and MTP "
            "fractions cannot be negative"
        )
    mtp_fraction_total = sum(mtp_fractions)
    if mtp_fraction_total and not math.isclose(
        mtp_fraction_total, 1.0, rel_tol=0, abs_tol=1e-9
    ):
        parser.error("Nonzero --gpu-mtp-fractions values must sum to 1")
    if args.exclude_mtp and mtp_fraction_total:
        parser.error("--exclude-mtp cannot be combined with GPU MTP fractions")

    devices = [
        DevicePlan(
            name=name,
            total_bytes=_gib(capacity),
            reserve_bytes=_gib(reserve),
            runtime_bytes=_gib(runtime),
            experts_per_layer=expert_count,
            backend=backend,
            mtp_fraction=mtp_fraction,
        )
        for name, capacity, reserve, runtime, expert_count, backend, mtp_fraction in zip(
            names,
            capacities,
            reserves,
            runtimes,
            expert_counts,
            backends,
            mtp_fractions,
        )
    ]

    try:
        audit = scan_checkpoint(args.model)
        excluded_categories = frozenset({"mtp"}) if args.exclude_mtp else frozenset()
        excluded_category_bytes = {
            category: audit.category_bytes.get(category, 0)
            for category in sorted(excluded_categories)
        }
        usages, diagnostics = build_placement(
            audit,
            host_total_bytes=_gib(args.host_ram_gib),
            host_reserve_bytes=_gib(args.host_reserve_gib),
            host_runtime_bytes=_gib(args.host_runtime_gib),
            devices=devices,
            allocator_overhead_fraction=args.allocator_overhead_percent / 100.0,
            excluded_categories=excluded_categories,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    result = _json_result(
        audit,
        usages,
        diagnostics,
        excluded_category_bytes=excluded_category_bytes,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_report(
            audit,
            usages,
            diagnostics,
            verbose_layers=args.verbose_layers,
            excluded_category_bytes=excluded_category_bytes,
        )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if result["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
