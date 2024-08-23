import numpy as np
from typing import Optional, Tuple, List
import torch
import sys
from pathlib import Path

wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
from src.gptfast.model import TransformerOutput

# # 示例数组
# my_array = np.array([1, 2, 3, 4, 5, 6])
# # 要删除的元素索引
# indices_to_remove = [1, 3, 5]
# # 删除元素
# filtered_array = remove_elements_by_indices_np(my_array, indices_to_remove)
# print(filtered_array)  # 输出: [1 3 5]
def remove_elements_by_indices_np(arr: np.ndarray, indices: List[int]) -> np.ndarray:
    mask = np.ones(len(arr), dtype=bool)
    mask[indices] = False
    return arr[mask]

def extract_hidden_states(outputs: TransformerOutput, compress_idxes_lst: List[List[int]], seq_len_lst: List[int], layer_idx=32, pad_mode='left'):
    '''
    compress_idx(list[list[int]]): [seq_id: [token_id: num_compressed_tokens]]
    seq_len_lst(list[int]): [seq: seq_len]
    '''
    assert len(compress_idxes_lst) == len(seq_len_lst), "The length of compress_idx_lst and seq_len_lst should be the same."
    assert pad_mode == 'left', "The pad_mode should be 'left'."
    
    batch_hidden_state = outputs.hidden_states[layer_idx]# shape is (batch_size, num_heads, seq_len, max_seq_len)
    bsz, max_seq_len, hidden_dim = batch_hidden_state.shape
    extracted_hidden_states = []
    for seq_id in range(len(compress_idxes_lst)):
        cur_uncompressed_idxs = torch.tensor([i for i in torch.arange(seq_len_lst[seq_id]) \
                                            if i not in compress_idxes_lst[seq_id]])
        cur_uncompressed_neg_ids = cur_uncompressed_idxs - seq_len_lst[seq_id]
        cur_attn_proj_hidden_state = batch_hidden_state[seq_id, cur_uncompressed_neg_ids, :]
        extracted_hidden_states.append(cur_attn_proj_hidden_state)
    
    assert len(extracted_hidden_states) == len(seq_len_lst), "The seqs number of extracted_hidden_states and seq_len_lst should be the same."
    return extracted_hidden_states

def extract_single_compressed_hidden_states(outputs: TransformerOutput, seq_len_lst: List[int], layer_idx=32, pad_mode='left'):
    '''
    compress_idx(list[list[int]]): [seq_id: [token_id: num_compressed_tokens]]
    seq_len_lst(list[int]): [seq: seq_len]
    '''
    assert pad_mode == 'left', "The pad_mode should be 'left'."
    
    batch_hidden_state = outputs.hidden_states[layer_idx] # shape is (batch_size, seq_len, dim)
    bsz, max_seq_len, hidden_dim = batch_hidden_state.shape
    extracted_hidden_states = []
    for seq_id in range(len(seq_len_lst)):
        cur_attn_proj_hidden_state = batch_hidden_state[seq_id, -seq_len_lst[seq_id]+1: , :]
        extracted_hidden_states.append(cur_attn_proj_hidden_state)
    
    assert len(extracted_hidden_states) == len(seq_len_lst), "The seqs number of extracted_hidden_states and seq_len_lst should be the same."
    return extracted_hidden_states