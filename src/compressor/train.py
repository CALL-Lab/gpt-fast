from datasets import Dataset
from pathlib import Path
from copy import deepcopy
import json
import tqdm
import sys
import torch
import numpy as np
from typing import Optional, Tuple, List


wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
import src.gptfast.model as gptFast
from src.gptfast.model import TransformerOutput
import src.gptfast.tokenizer as gptFastTokenizer
from compressor_model_test import compress_Transformer, transformer_configs
from train_utils import remove_elements_by_indices_np

def remove_compressed_tokens(token_ids: List[int], compress_ids: List[int]) -> List[int]:
    return remove_elements_by_indices_np(np.array(token_ids), compress_ids)

def batch_padding(token_ids: List[List[int]], padding: int, max_len = None) -> List[List[int]]:
    if max_len is None:
        max_len = max(len(ids) for ids in token_ids)
    for i, ids in enumerate(token_ids):
        if len(ids) < max_len:
            token_ids[i] = [padding] * (max_len - len(ids)) + ids
    return token_ids

def get_data_batch(batch, device: str):
    braced_tokens = batch['brace_token_ids']
    extracted_hidden_states = batch['extracted_hidden_states']
    padded_compress_ids, seq_start_number = batch['tokens_ids'], batch['pads_number']
    unpadded_compress_ids = padded_compress_ids[seq_start_number:]
    reserved_tokens = remove_compressed_tokens(unpadded_compress_ids, batch['compress_ids'])
    
    return data_rows

padded_braced_tokens = batch_padding(braced_tokens, padding=0, max_len=96)


def compressor_batch_train(compress_model: compress_Transformer, llm_model: gptFast.Transformer, data_rows: torch.tensor, batch_num: int):
    

def train_loop(compress_model: compress_Transformer, llm_model: gptFast.Transformer, tokenizer: gptFastTokenizer.TiktokenWrapper, compress_datasets: list[Dataset]):
    compress_model.train()
    llm_model.eval()
    
    
    

def main():
    device = "cuda"
    model_args = gptFast.ModelArgs(**gptFast.transformer_configs["Llama-3-8B"], max_seq_length=32, output_hidden_states=True, output_attentions=True)
    # load tokenizer, llama3 uses tiktoken
    tokenizer = gptFastTokenizer.TiktokenWrapper("/home/yuhao/work/code_repo/gpt-fast/tokenizer.model")
    llm_model = gptFast.Transformer.from_pretrained(model_args, "/home/yuhao/work/code_repo/gpt-fast/consolidated.00.pth", device)
    compressor_args = gptFast.ModelArgs(**transformer_configs["compressor"], max_seq_length=32, output_hidden_states=True, output_attentions=True)
    compress_model = compress_Transformer.creat_instance(compressor_args, device)
    
    output_path = Path("dataset/stage2/")
    compress_datasets = []
    for i, data_file_path in enumerate(output_path.glob("*.parquet*")):
        ds = Dataset.from_parquet(str(data_file_path))
        if i == 0: print(ds.column_names)
        compress_datasets.append(ds)
    train_loop(compress_model, llm_model, tokenizer, compress_datasets: list)