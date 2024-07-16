from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import math

from src.gptfast.model import Transformer, \
    TransformerBlock, Attention, FeedForward, RMSNorm,\
    TransformerOutput, TransformerBlockOutput, AttentionOutput, KVCache, \
    find_multiple, precompute_freqs_cis,\
    apply_rotary_emb, scaled_dot_product_attention
from utils import _get_model_size, encode_tokens
from src.gptfast.tokenizer import TiktokenWrapper, SentencePieceWrapper

transformer_configs = {
    "basic-compressor-v1":dict(block_size=128, n_layer=8, n_head=4, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256),
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

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 128
    rope_base: float = 10000
    norm_eps: float = 1e-5
    max_batch_size: int = 1
    max_seq_length: int = 128
    output_logits: bool = True
    output_hidden_states: bool = False
    output_attentions: bool = False
    
    compressor_attach_name = "Llama-3-8B"
    compressor_architecture: str = 'seq2one-e' # choose from ['seq2one-v','seq2one-e']

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        assert self.dim % self.n_head == 0, "dim({}) has no aliquot with value n_head({})".format(self.dim, self.n_head)
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

"""
This is the compressor model.
"""
class compressor_Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        attached_model_name = self.config.compressor_attach_name
        self.config.vocab_size, self.config.dim = transformer_configs[attached_model_name]['vocab_size'], \
                                                  transformer_configs[attached_model_name]['dim']

        config = self.config
        
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(self.config.n_layer))
        self.norm = RMSNorm(self.config.dim, eps=self.config.norm_eps)
        if self.config.compressor_architecture == 'seq2one-v':
            self.output = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)
        elif self.config.compressor_architecture == 'seq2one-e':
            self.output = nn.Linear(self.config.dim, self.config.dim, bias=False)
        self.freqs_cis: Optional[Tensor] = None

    # this need to be invoked after the weights is initialized or loaded
    def post_init(self, embedding_model_dict_path: str = "Llama-3-8B/original/consolidated.00.pth", model_dict_key: str = 'tok_embeddings.weight') -> None:
        # load embedding model and fix embedding model
        print(f"Loading embedding layer to compressor from {embedding_model_dict_path}......")
        checkpoint: dict = torch.load(embedding_model_dict_path, mmap=True, weights_only=True)
        self.tok_embeddings.load_state_dict(checkpoint[model_dict_key], assign=True)
        for param in self.tok_embeddings.parameters(): 
            param.requires_grad = False
        # config recheck
        head_dim = self.config.dim // self.config.n_head
        self.config.max_seq_length = find_multiple(self.config.max_seq_length, 8)
        dtype = self.output.weight.dtype
        
        # For quantized layers, dtype is encoded in scales
        # if hasattr(self.output, "scales"):
        #     dtype = self.output.scales.dtype
        # elif hasattr(self.output, "scales_and_zeros"):
        #     dtype = self.output.scales_and_zeros.dtype
        # for b in self.layers:
        #     b.attention.kv_cache = KVCache(self.config.max_batch_size, self.config.max_seq_length, self.config.n_local_heads, head_dim, dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype)
        self.causal_mask = torch.tril(torch.ones(self.config.max_seq_length, self.config.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> TransformerOutput:
        assert self.freqs_cis is not None, "`post_init()` must be involked first"
        mask = self.causal_mask[input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)
        # layer level output
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
        if self.config.compressor_architecture == 'seq2one-v':
            # Compress with vocabulary sample and composition
            assert 0 == 1, "Seq2one-v forward not finished."
        elif self.config.compressor_architecture == 'seq2one-e':
            return logits
            

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
    
    def __repr__(self):
        return f'compressor_Transformer attached to model {self.config.compressor_attach_name}'


class Compressor(nn.Module):
    def __init__(self, attached_model_name: str = "Llama-3-8B") -> None:
        super().__init__()
        config = ModelArgs(n_layer=8, n_head=8,compressor_architecture='seq2one-e', compressor_attach_name=attached_model_name)
        compress_model = compressor_Transformer(config)
        compress_model.post_init(embedding_model_dict_path = "Llama-3-8B/original/consolidated.00.pth", 
                                 model_dict_key = 'tok_embeddings.weight')
        print(f"Initiate compressor_model finished, size {_get_model_size(compress_model)}")
    
    def compressor_train(self, compress_dataset):
        pass
    
    def compress_to_one_token(self, tokenizer, prompt):
        encoded = encode_tokens(tokenizer, prompt, bos=True, device=self.device)
        prompt_length = encoded.size(0)