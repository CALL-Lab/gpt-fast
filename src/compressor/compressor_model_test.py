# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import math
import sys
from pathlib import Path

wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
from src.gptfast.model import TransformerOutput, TransformerBlockOutput, AttentionOutput,  \
    ModelArgs, TransformerBlock, Attention, FeedForward, RMSNorm, KVCache,\
    find_multiple, precompute_freqs_cis, apply_rotary_emb, scaled_dot_product_attention



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
    "compressor": dict(block_size=1024, n_layer=8, n_head=8, n_local_heads=4, dim=4096, intermediate_size=4096, vocab_size=128256, compressed_tokens_num = 1, compressor_attach_name = "Llama-3-8B"),
}


class compress_Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        attached_model_name = self.config.compressor_attach_name
        self.config.vocab_size, self.config.dim = transformer_configs[attached_model_name]['vocab_size'], \
                                                  transformer_configs[attached_model_name]['dim']
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.compressed_tokens_num * config.dim, bias=False)

        self.freqs_cis: Optional[Tensor] = None

    # this need to be invoked after the weights is initialized or loaded
    def post_init(self, embedding_model_dict_path: str = "Llama-3-8B/original/consolidated.00.pth", model_dict_key: str = 'tok_embeddings.weight') -> None:
        # load embedding model and fix embedding model
        print(f"Loading embedding layer to compressor from {embedding_model_dict_path}......")
        checkpoint: dict = torch.load(embedding_model_dict_path, mmap=True, weights_only=True)
        self.tok_embeddings.load_state_dict({"weight": checkpoint[model_dict_key]}, assign=True)
        for param in self.tok_embeddings.parameters(): 
            param.requires_grad = False
        # config recheck
        head_dim = self.config.dim // self.config.n_head
        self.config.max_seq_length = find_multiple(self.config.max_seq_length, 8)
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        # for b in self.layers:
        #     b.attention.kv_cache = KVCache(self.config.max_batch_size, self.config.max_seq_length, self.config.n_local_heads, head_dim, dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype)
        self.freqs_cis_batch = self.freqs_cis[0:self.config.max_seq_length]
        self.causal_mask = torch.tril(torch.ones(self.config.max_seq_length, self.config.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, seq_lens: Tensor = None, tok_level_pad_mask: bool = False, drop_out_p: float=0) -> TransformerOutput: # idx: [B,S] ; input_poses: List(arange(seq_length)) with batch length
        '''
        Args:
            idx (Tensor): The input tensor of shape [B, S] where B is the batch size and S is the sequence length.
            seq_lens (Tensor): The tensor  of shape [S] containing the lengths of the input sequences.
            tok_level_pad_mask (bool, optional): Whether to apply token-level padding mask, which does not affect the attention pad mask. Defaults to False.
        Returns:
            output(TransformerOutput): The output of the transformer model, including logits, hidden states, and attentions.
        '''
        assert self.freqs_cis is not None, "`post_init()` must be involked first."
        # assert idx.shape == input_poses.shape, "idx and input_poses should have the same shape."
        assert idx.size(1) ==  self.config.max_seq_length, "Input sequence length should be equal to max_seq_length in batch inference."
        masks = torch.zeros(idx.size(0), self.config.max_seq_length, self.config.max_seq_length, dtype=torch.bool).to(self.causal_mask.device)
        if tok_level_pad_mask:
            tok_masks = torch.zeros(idx.size(0), self.config.max_seq_length).to(self.causal_mask.device) # non-bool mask
        for seq_id in torch.arange(len(seq_lens)):
            mask = self.causal_mask.clone()
            mask[ : , 0: (self.config.max_seq_length - seq_lens[seq_id])] = False # Left padding mask
            masks[seq_id] = mask.type(torch.bool)
            if tok_level_pad_mask:
                tok_masks[seq_id][ - seq_lens[seq_id] : ] = 1 # seq token mask set true
        
        # masks = torch.stack(masks).type(torch.bool)
        freqs_cis = self.freqs_cis_batch

        # hidden_states: Optional[Tuple[Tensor]] = None
        # attentions: Optional[Tuple[Tensor]] = None
        # if self.config.output_hidden_states:
        #     hidden_states = tuple()
        # if self.config.output_attentions:
        #     attentions = tuple()

        x = self.tok_embeddings(idx)
        if tok_level_pad_mask:
            x = tok_masks.type(x.dtype).unsqueeze(-1) * x
        # attention layers
        for i, layer in enumerate(self.layers):
            layer_output: TransformerBlockOutput = layer(x, seq_lens, freqs_cis, masks, drop_out_p=drop_out_p)
            x = layer_output.hidden_state
            # if self.config.output_hidden_states:
            #     hidden_states += (layer_output.hidden_state, )
            # if self.config.output_attentions:
            #     attentions += (layer_output.attention, )

        # normalization layer
        x = self.norm(x)
        # if self.config.output_hidden_states:
        #     hidden_states += (x, )

        # logits: Optional[Tensor] = None
        # if self.config.output_logits:
        #     logits = self.output(x)

        # return TransformerOutput(
        #     logits=logits,
        #     hidden_states=hidden_states,
        #     attentions=attentions
        # )
        compressed_token = self.output(x)
        return compressed_token

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

    @classmethod
    def creat_compressor(cls, config: ModelArgs, device, 
                         embedding_model_dict_path="consolidated.00.pth", 
                         model_dict_key='tok_embeddings.weight'):
        # this prevents memory allocation on model creation
        with torch.device(device):
            model = cls(config)
            model.post_init(embedding_model_dict_path, model_dict_key)
        return model
    
    def __repr__(self):
        return f'compressor_Transformer attached to model {self.config.compressor_attach_name}'