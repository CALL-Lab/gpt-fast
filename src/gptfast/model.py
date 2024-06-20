# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import math


# The smallest multiple of k that is greater than or equal to n
def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

# The output of a transformer model
# including the logits and intermediate variables
@dataclass
class TransformerOutput:
    logits: Optional[Tensor]
    hidden_states: Optional[Tuple[Tensor]]
    attentions: Optional[Tuple[Tensor]]

# The output of a single layer in a transformer model
@dataclass
class TransformerBlockOutput:
    hidden_state: Tensor
    attention: Optional[Tensor]

# The output of attention
@dataclass
class AttentionOutput:
    attention_output: Tensor
    attention_coefficients: Optional[Tensor]

# Hyperparams of a LLM
@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    max_batch_size: int = 1
    max_seq_length: int = 128
    output_logits:bool = True
    output_hidden_states:bool = False
    output_attentions:bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config in str(name).upper() or config in str(name)]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match

        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "Mistral-7B": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),
    "Llama-3-8B": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config: ModelArgs = config

        # layers
        self.tok_embeddings: nn.Embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers: nn.ModuleList = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm: nn.Module = RMSNorm(config.dim, eps=config.norm_eps)
        self.output: nn.Module = nn.Linear(config.dim, config.vocab_size, bias=False)

        # positional embedding cache
        self.freqs_cis: Optional[Tensor] = None

    # this need to be invoked after the weights is initialized or loaded
    def post_init(self, max_seq_length:Optional[int]=None, max_batch_size:Optional[int]=None) -> None:
        # You can change the maximum sequence length and batch size by rerunning `post_init`
        # This will reset the positional embedding cache and KV cache
        if max_seq_length is not None:
            self.config.max_seq_length = max_seq_length
        if max_batch_size is not None:
            self.config.max_batch_size = max_batch_size

        # Initialize the KV cache
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

        # Initialize the positional embedding cache
        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype)
        self.causal_mask = torch.tril(torch.ones(self.config.max_seq_length, self.config.max_seq_length, dtype=torch.bool))

    def forward(self, tokens: Tensor, input_pos: Optional[Tensor] = None) -> TransformerOutput:
        assert self.freqs_cis is not None, "`post_init()` must be involked first"
        mask = self.causal_mask[input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        # if `tokens` is a 2D tensor, we treat it as a batch of token ids
        # if `tokens` is a 3D tensor, we treat it as a batch of word vectors
        assert len(tokens.shape)==2 or (len(tokens.shape)==3 and tokens.shape[-1]==self.config.dim), "invalid input shape"
        # if `tokens` is token ids, we need to convert it into word vectors first
        if len(tokens.shape) == 2:
            x = self.tok_embeddings(tokens)
        elif len(tokens.shape) == 3:
            x = tokens

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

        # output layer
        logits: Optional[Tensor] = None
        if self.config.output_logits:
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


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> TransformerBlockOutput:
        attention_output: AttentionOutput = self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        h = x + attention_output.attention_output
        out = h + self.feed_forward(self.ffn_norm(h))
        return TransformerBlockOutput(
            hidden_state=out,
            attention=attention_output.attention_coefficients
        )


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.config: ModelArgs = config
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> AttentionOutput:
        bsz, seqlen, _ = x.shape

        kv_size = self.config.n_local_heads * self.config.head_dim
        q, k, v = self.wqkv(x).split([self.config.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.config.n_head, self.config.head_dim)
        k = k.view(bsz, seqlen, self.config.n_local_heads, self.config.head_dim)
        v = v.view(bsz, seqlen, self.config.n_local_heads, self.config.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.config.n_head // self.config.n_local_heads, dim=1)
        v = v.repeat_interleave(self.config.n_head // self.config.n_local_heads, dim=1)
        y, attn_weight = scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.config.dim)

        y = self.wo(y)
        return AttentionOutput(
            attention_output=y,
            attention_coefficients=attn_weight if self.config.output_attentions else None
        )


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

# An alternative implementation of `torch.nn.functions.scaled_dot_product_attention`
# that outputs the attention weights.
def scaled_dot_product_attention(query:Tensor, key:Tensor, value:Tensor, attn_mask:Optional[Tensor]=None, dropout_p:float=0.0, is_causal:bool=False, scale:float=None) -> Tuple[Tensor]:
    device = query.device
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    attn_output = attn_weight @ value
    return attn_output, attn_weight