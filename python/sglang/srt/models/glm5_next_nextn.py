"""GLM-5-Next appended MTP block for EAGLE/NextN decoding."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from sglang.srt.configs.glm5_next import Glm5NextConfig, Glm5NextTextConfig
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.glm5_next import (
    Glm5NextDecoderLayer,
    Glm5NextForConditionalGeneration,
    normalize_glm5_next_weight_name,
)
from sglang.srt.models.glm5_next_norm import Glm5NextRMSNorm
from sglang.srt.models.transformers import maybe_prefix
from sglang.srt.utils.common import BumpAllocator


class Glm5NextModelNextN(nn.Module):
    """One checkpoint-appended DSA/MoE block with shared embeddings/head."""

    def __init__(
        self,
        config: Glm5NextTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.enorm = Glm5NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = Glm5NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        self.alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.decoder = Glm5NextDecoderLayer(
            config=config,
            layer_idx=0,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "decoder"),
            alt_stream=self.alt_stream,
            is_nextn=True,
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = Glm5NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        if hidden_states.shape[0] != 0:
            target_hidden_states = forward_batch.spec_info.hidden_states
            if target_hidden_states.shape[-1] != self.config.hidden_size:
                raise RuntimeError(
                    "GLM-5-Next MTP target hidden width must be "
                    f"{self.config.hidden_size}, got "
                    f"{target_hidden_states.shape[-1]}"
                )
            hidden_states = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(target_hidden_states),
                    ),
                    dim=-1,
                )
            )

        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        with get_global_expert_distribution_recorder().disable_this_region():
            hidden_states, residual = self.decoder(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=None,
                zero_allocator=zero_allocator,
            )

        if residual is not None:
            raise RuntimeError("final GLM-5-Next MTP residual state must be None")
        if hidden_states.shape[0] != 0:
            hidden_states = self.shared_head.norm(hidden_states)
        return hidden_states


class Glm5NextForConditionalGenerationNextN(Glm5NextForConditionalGeneration):
    """Text-only draft model backed by GLM's appended checkpoint layer."""

    def __init__(
        self,
        config: Glm5NextConfig | Glm5NextTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        text_config = getattr(config, "text_config", config)
        self.config = text_config
        self.quant_config = quant_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pp_group = get_pp_group()
        self.num_fused_shared_experts = 0
        self.multimodal_enabled = False

        self.model = Glm5NextModelNextN(
            text_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model.shared_head.head"),
        )
        self.logits_processor = LogitsProcessor(text_config)
        get_attn_tp_context().init_context(text_config.q_lora_rank, True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], is_nextn: bool = True
    ) -> None:
        del is_nextn

        def normalized_weights():
            for name, tensor in weights:
                normalized_name = normalize_glm5_next_weight_name(name)
                if normalized_name is not None:
                    yield normalized_name, tensor

        DeepseekV2WeightLoaderMixin.do_load_weights(
            self, normalized_weights(), is_nextn=True
        )


EntryClass = [Glm5NextForConditionalGenerationNextN]
