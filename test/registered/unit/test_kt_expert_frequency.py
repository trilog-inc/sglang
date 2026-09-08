from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import kt_ep_wrapper
from sglang.srt.layers.moe.kt_ep_wrapper import _load_activation_frequency


def _server_args(
    profile,
    hf_config,
    num_gpu_experts,
    strategy="frequency",
    max_tokens=None,
    cpu_costs=None,
):
    return SimpleNamespace(
        get_hf_config=lambda: hf_config,
        kt_gpu_experts_ratio=None,
        kt_num_gpu_experts=num_gpu_experts,
        kt_expert_placement_strategy=strategy,
        kt_expert_frequency_file=str(profile),
        kt_expert_frequency_max_tokens=max_tokens,
        kt_expert_frequency_cpu_costs=cpu_costs,
        init_expert_location="trivial",
    )


def _run_placement(monkeypatch, server_args):
    monkeypatch.setattr(kt_ep_wrapper, "_KT_GPU_EXPERTS_MASKS", None)
    monkeypatch.setattr(kt_ep_wrapper, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(kt_ep_wrapper.dist, "is_initialized", lambda: False)
    return kt_ep_wrapper._init_kt_gpu_experts_masks(server_args)


def test_load_activation_frequency_sums_recorder_buffer(tmp_path):
    samples = torch.tensor(
        [
            [[10, 0, 0], [0, 2, 1]],
            [[0, 9, 0], [4, 0, 0]],
        ]
    )
    profile = tmp_path / "buffered_distribution.pt"
    torch.save({"logical_count": samples}, profile)

    actual = _load_activation_frequency(str(profile), 2, 3)

    torch.testing.assert_close(
        actual, torch.tensor([[10, 9, 0], [4, 2, 1]], dtype=torch.float64)
    )


def test_load_activation_frequency_filters_prefill_samples(tmp_path):
    samples = torch.tensor(
        [
            [[1, 1, 0], [0, 1, 1]],
            [[4, 3, 1], [2, 2, 4]],
            [[0, 1, 1], [1, 0, 1]],
        ]
    )
    profile = tmp_path / "buffered_distribution.pt"
    torch.save({"logical_count": samples}, profile)

    actual = _load_activation_frequency(
        str(profile),
        2,
        3,
        max_tokens=1,
        num_experts_per_tok=2,
        moe_layers=[0, 1],
    )

    torch.testing.assert_close(
        actual, torch.tensor([[1, 2, 1], [1, 1, 2]], dtype=torch.float64)
    )


def test_load_activation_frequency_filter_requires_buffered_profile(tmp_path):
    profile = tmp_path / "aggregated_distribution.pt"
    torch.save({"logical_count": torch.ones(2, 3)}, profile)

    with pytest.raises(ValueError, match="three-dimensional logical_count"):
        _load_activation_frequency(
            str(profile),
            2,
            3,
            max_tokens=1,
            num_experts_per_tok=2,
            moe_layers=[0, 1],
        )


def test_load_activation_frequency_rejects_negative_counts(tmp_path):
    profile = tmp_path / "negative_distribution.pt"
    torch.save(torch.tensor([[1, -1]]), profile)

    with pytest.raises(ValueError, match="negative counts"):
        _load_activation_frequency(str(profile), 1, 2)


def test_frequency_placement_selects_top_experts_per_layer(tmp_path, monkeypatch):
    scores = torch.tensor(
        [
            [1, 9, 3, 8],
            [6, 2, 7, 0],
            [4, 5, 1, 10],
        ]
    )
    profile = tmp_path / "expert_distribution_recorder.pt"
    torch.save({"logical_count": scores}, profile)
    hf_config = SimpleNamespace(
        num_hidden_layers=3,
        num_hash_layers=0,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        n_routed_experts=4,
    )

    masks = _run_placement(
        monkeypatch, _server_args(profile, hf_config, num_gpu_experts=2)
    )

    assert masks.tolist() == [
        [False, True, False, True],
        [True, False, True, False],
        [False, True, False, True],
    ]
    assert masks.sum(dim=1).tolist() == [2, 2, 2]


def test_frequency_ratio_preserves_total_budget_across_layers(tmp_path, monkeypatch):
    profile = tmp_path / "ratio_distribution.pt"
    torch.save(torch.arange(12).reshape(3, 4), profile)
    hf_config = SimpleNamespace(
        num_hidden_layers=3,
        num_hash_layers=0,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        n_routed_experts=4,
    )
    server_args = _server_args(profile, hf_config, num_gpu_experts=None)
    server_args.kt_gpu_experts_ratio = 1 / 3

    masks = _run_placement(monkeypatch, server_args)

    assert masks.sum().item() == 4
    assert masks.sum(dim=1).tolist() == [2, 1, 1]


def test_global_frequency_placement_moves_budget_between_layers(tmp_path, monkeypatch):
    scores = torch.tensor(
        [
            [800, 100, 50, 50],
            [25, 25, 25, 25],
            [700, 100, 100, 100],
        ]
    )
    profile = tmp_path / "expert_distribution_recorder.pt"
    torch.save({"logical_count": scores}, profile)
    hf_config = SimpleNamespace(
        num_hidden_layers=3,
        num_hash_layers=0,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        n_routed_experts=4,
    )

    masks = _run_placement(
        monkeypatch,
        _server_args(
            profile,
            hf_config,
            num_gpu_experts=2,
            strategy="frequency-global",
        ),
    )

    assert masks.tolist() == [
        [True, False, False, False],
        [True, True, True, True],
        [True, False, False, False],
    ]
    assert masks.sum().item() == 6


def test_latency_aware_placement_uses_route_cooccurrence(tmp_path, monkeypatch):
    samples = torch.tensor(
        [
            [[1, 1, 0, 0], [1, 1, 0, 0]],
            [[1, 1, 0, 0], [1, 1, 0, 0]],
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            [[0, 0, 1, 1], [0, 0, 1, 1]],
        ]
    )
    profile = tmp_path / "buffered_distribution.pt"
    torch.save({"logical_count": samples}, profile)
    hf_config = SimpleNamespace(
        num_hidden_layers=2,
        num_hash_layers=0,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
    )

    masks = _run_placement(
        monkeypatch,
        _server_args(
            profile,
            hf_config,
            num_gpu_experts=1,
            strategy="frequency-global-latency",
            max_tokens=1,
            cpu_costs=[0.0, 1.0, 10.0],
        ),
    )

    assert masks.tolist() == [
        [True, False, True, False],
        [False, False, False, False],
    ]
    assert masks.sum().item() == 2


def test_latency_aware_placement_requires_complete_cost_curve(
    tmp_path, monkeypatch
):
    profile = tmp_path / "buffered_distribution.pt"
    torch.save({"logical_count": torch.tensor([[[1, 1, 0]]])}, profile)
    hf_config = SimpleNamespace(
        num_hidden_layers=1,
        num_hash_layers=0,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        n_routed_experts=3,
        num_experts_per_tok=2,
    )

    with pytest.raises(ValueError, match="CPU top-k 0 through 2"):
        _run_placement(
            monkeypatch,
            _server_args(
                profile,
                hf_config,
                num_gpu_experts=1,
                strategy="frequency-global-latency",
                max_tokens=1,
                cpu_costs=[0.0, 1.0],
            ),
        )


def test_frequency_placement_rejects_incomplete_profile(tmp_path, monkeypatch):
    profile = tmp_path / "incomplete_distribution.pt"
    torch.save({"logical_count": torch.tensor([[1, 2], [0, 0]])}, profile)
    hf_config = SimpleNamespace(
        num_hidden_layers=2,
        num_hash_layers=0,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        n_routed_experts=2,
    )

    with pytest.raises(ValueError, match="no recorded routes"):
        _run_placement(monkeypatch, _server_args(profile, hf_config, num_gpu_experts=1))
