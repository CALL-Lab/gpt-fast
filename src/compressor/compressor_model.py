from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import math

from src.gptfast.model import Transformer, ModelArgs, \
    TransformerBlock, Attention, FeedForward, RMSNorm,\
    TransformerOutput, TransformerBlockOutput, AttentionOutput, KVCache, \
    find_multiple, precompute_freqs_cis,\
    apply_rotary_emb, scaled_dot_product_attention
from src.gptfast.tokenizer import TiktokenWrapper, SentencePieceWrapper

"""
This is the compressor model.
"""
class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None

    # this need to be invoked after the weights is initialized or loaded
    def post_init(self) -> None:
        head_dim = self.config.dim // self.config.n_head
        self.config.max_seq_length = find_multiple(self.config.max_seq_length, 8)
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(self.config.max_batch_size, self.config.max_seq_length, self.config.n_local_heads, head_dim, dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype)
        self.causal_mask = torch.tril(torch.ones(self.config.max_seq_length, self.config.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> TransformerOutput:
        assert self.freqs_cis is not None, "`post_init()` must be involked first"
        mask = self.causal_mask[input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        hidden_states: Optional[Tuple[Tensor]] = None
        attentions: Optional[Tuple[Tensor]] = None
        if self.config.output_hidden_states:
            hidden_states = tuple()
        if self.config.output_attentions:
            attentions = tuple()

        # attention layers
        for i, layer in enumerate(self.layers):
            layer_output: TransformerBlockOutput = layer(x, input_pos, freqs_cis, mask)
            x = layer_output.hidden_state
            if self.config.output_hidden_states:
                hidden_states += (layer_output.hidden_state, )
            if self.config.output_attentions:
                attentions += (layer_output.attention, )

        # normalization layer
        x = self.norm(x)
        if self.config.output_hidden_states:
            hidden_states += (x, )

        logits = self.output(x)
        return TransformerOutput(
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions
        )

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))

    # load a pretrained model
    @classmethod
    def from_pretrained(cls, config: ModelArgs, checkpoint_file, device):
        # this prevents memory allocation on model creation
        with torch.device('meta'):
            model = cls(config)
        # weights is directly loaded into vram by mmap the weight file
        checkpoint = torch.load(checkpoint_file, mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)
        model = model.to(device)
        with torch.device(device):
            model.post_init()
        return model

 