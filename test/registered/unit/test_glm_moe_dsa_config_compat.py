import transformers.configuration_utils as hf_configuration_utils

from sglang.srt.utils.hf_transformers_utils import (
    _ensure_deepseek_sparse_attention_layer_type_compat,
)


def test_deepseek_sparse_attention_layer_type_compat_is_idempotent(monkeypatch):
    monkeypatch.setattr(
        hf_configuration_utils,
        "ALLOWED_LAYER_TYPES",
        ("full_attention", "sparse"),
    )

    _ensure_deepseek_sparse_attention_layer_type_compat()
    _ensure_deepseek_sparse_attention_layer_type_compat()

    assert hf_configuration_utils.ALLOWED_LAYER_TYPES == (
        "full_attention",
        "sparse",
        "deepseek_sparse_attention",
    )
