from datasets import Dataset
from pathlib import Path
import os
from copy import deepcopy
import json
import tqdm
from typing import Optional, Tuple, List
import numpy as np
from itertools import takewhile


def count_leading_zeros(int_list):
    return len(list(takewhile(lambda x: x == 0, int_list)))

def batch_padding(token_ids: List[List[int]], padding: int, max_len = None) -> List[List[int]]:
    if max_len is None:
        max_len = max(len(ids) for ids in token_ids)
    for i, ids in enumerate(token_ids):
        if len(ids) < max_len:
            token_ids[i] = [padding] * (max_len - len(ids)) + ids
    return token_ids

def row_padding(token_ids: List[int], padding: int, max_len = None) -> List[int]:
    if len(token_ids) < max_len:
        token_ids = [padding] * (max_len - len(token_ids)) + token_ids
    return token_ids

def reserved_token_row_padding(token_ids: List[int], padding: int, max_len = None, mode = 'CA') -> List[int]:
    assert mode in ['CA', 'CAC','AC'], "The mode should be 'CA', 'CAC' or 'AC'."
    assert len(token_ids) < max_len, "Dataset has been padded to the max length."
    if mode == 'CA':
        token_ids = [padding] * (max_len - len(token_ids)) + token_ids
    elif mode == 'CAC':
        assert max_len - len(token_ids) >= 2, "The to pad length should be greater than or equal to 2." 
        token_ids = [padding] * (max_len - len(token_ids) - 1) + token_ids + [padding]
    elif mode == 'AC':
        token_ids = token_ids + [padding] * (max_len - len(token_ids))
    return token_ids

def save_pads_number(row):
    pads_number = len(row['token_ids']) - len(json.loads(row['braced_token_ids'])) + len(json.loads(row['compress_ids']))
    assert pads_number >= 0, "The pads number should be greater than or equal to 0."
    row['pads_number'] = pads_number
    return row


def remove_elements_by_indices_np(arr: np.ndarray, indices: List[int]) -> np.ndarray:
    mask = np.ones(len(arr), dtype=bool)
    mask[indices] = False
    return arr[mask]

def remove_compressed_tokens(token_ids: List[int], compress_ids: List[int]) -> List[int]:
    return remove_elements_by_indices_np(np.array(token_ids), compress_ids)


def modify_dataset_for_train(row):
    row['pads_number'] = min(count_leading_zeros(row['token_ids']), count_leading_zeros(row['pred']))
    row['origin_seq_len'] = len(row['token_ids']) - row['pads_number']
    
    braced_tokens = json.loads(row['braced_token_ids']) # [json.loads(braced_ids) for braced_ids in row['braced_token_ids']]
    row['braced_token_ids'] = braced_tokens
    
    padded_braced_tokens = row_padding(braced_tokens, padding=0, max_len=96)
    row['padded_braced_tokens'] = padded_braced_tokens
    
    # extracted_hidden_states = batch['extracted_hidden_states']
    padded_compress_ids, seq_start_number = row['token_ids'], row['pads_number']
    unpadded_compress_ids = padded_compress_ids[seq_start_number:]
    reserved_tokens = remove_compressed_tokens(unpadded_compress_ids, json.loads(row['compress_ids']))
    row['reserved_tokens'] = reserved_tokens
    
    return row

def reserved_tokens_pad(row):
    row['padded_reserved_tokens'] = reserved_token_row_padding(row['reserved_tokens'], padding=0, max_len=96, mode='CA')
    return row

def seq_num_calc(row):
    row['origin_seq_len'] = len(row['token_ids']) - row['pads_number']
    return row

def seq_num_calc(row):
    # print('eval')
    row['eval_seq_len'] = len(row['token_ids']) - row['pads_number'] - len(json.loads(row['compress_ids'])) + 1
    row['braced_len'] = len(row['braced_token_ids'])
    return row

def post_process_compress_dataset(data_file_dir: Path, mode: str, output_dir: str = None):
    if mode == "pad_num_calc":
        for data_file in tqdm.tqdm(list(data_file_dir.glob("*.parquet_test10000"))):
            ds = Dataset.from_parquet(str(data_file))
            ds = ds.map(save_pads_number)
            print(ds.column_names)
            ds.to_parquet(str(data_file))
            ds.cleanup_cache_files()
            
    elif mode == "training_prepare":
        for data_file in tqdm.tqdm(list(data_file_dir.glob("*.parquet*"))):
            ds = Dataset.from_parquet(str(data_file))
            # ds = ds.map(modify_dataset_for_train)
            # ds = ds.map(reserved_tokens_pad)
            # ds = ds.map(seq_num_calc)
            ds = ds.map(seq_num_calc)
            print(ds.column_names)
            ds.to_parquet(os.path.join(output_dir, "train_"+str(data_file.name)))
            ds.cleanup_cache_files()
            # for test
            # r1 = ds[0]
            # print("1")

# stg3_dataset_path = Path("dataset/stage2/")
stg3_dataset_path = Path("dataset/to_train_stage_4")
output_dir = "dataset/to_train_stage_5/"
print(f"source_dir: {stg3_dataset_path}, output_dir: {output_dir}")

if __name__ == "__main__":
    post_process_compress_dataset(stg3_dataset_path, "training_prepare", output_dir)
    

