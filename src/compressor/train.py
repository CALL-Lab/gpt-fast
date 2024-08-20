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


def train_loop(compress_model: compress_Transformer, llm_model: gptFast.Transformer, tokenizer: gptFastTokenizer.TiktokenWrapper, compress_datasets: list):
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