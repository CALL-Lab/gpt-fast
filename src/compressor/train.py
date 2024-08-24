from datasets import Dataset, DatasetDict, concatenate_datasets
from pathlib import Path
from copy import deepcopy
import json
import tqdm
import sys
import torch
from torch.nn import MSELoss
from torch import Tensor
import numpy as np
from typing import Optional, Tuple, List
from datetime import datetime
import os
import wandb as wb


wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))
import src.gptfast.model as gptFast
from src.gptfast.model import TransformerOutput
import src.gptfast.tokenizer as gptFastTokenizer
from compressor_model_test import compress_Transformer, transformer_configs
from train_utils import remove_elements_by_indices_np, extract_hidden_states, \
    extract_single_compressed_hidden_states


BATCH_SIZE = 5
# EPOCHS = 10
update_freq = 1
eval_freq = 20
LR = 5e-2
DROP_OUT_P = 0.0
datasets_num = 14
MAX_seq_len = 96
embedding_scale = 1e-3

device = "cuda"


PRIVATE_WB_KEY = "your_wb_key"


# Initiate W&B experiment tracker
os.environ["WANDB_API_KEY"] = PRIVATE_WB_KEY
wb.login(key=PRIVATE_WB_KEY)
wb_run = wb.init(
    # set the wandb project where this run will be logged
    project="compressor training",
    reinit=True,
    mode="online",
    # track hyperparameters and run metadata
    config={
        "train_mode": "compressor",
        "batch_size": BATCH_SIZE,
        # "epochs": EPOCHS,
        "update_freq": update_freq,
        "eval_freq": eval_freq,
        "lr": LR,
        "drop_out_p": DROP_OUT_P,
        "datasets_num": datasets_num,
        "max_seq_len": MAX_seq_len,
        "device": device,
    }
)


class simple_Dataset_dataloader():
    def __init__(self, datasetDict) -> None:
        self.datasetDict = datasetDict
        self.cur_train_row_id = 0
        self.cur_test_row_id = 0
        
    def dataset_get_batch(self, batch_size: int, split: str = 'train') -> Tuple[Optional[Dataset], bool]:
        '''
        return: batch, end_flag
        '''
        assert split in ['train', 'test'], "The split should be 'train' or 'test'."
        dataset = self.datasetDict[split]
        if split == 'train':
            if self.cur_train_row_id + batch_size > len(dataset):
                return None, True # end of dataset
            data_rows = dataset[self.cur_train_row_id: self.cur_train_row_id + batch_size]
            self.cur_train_row_id += batch_size
        elif split == 'test':
            if self.cur_test_row_id + batch_size > len(dataset):
                return None, True # end of dataset
            data_rows = dataset[self.cur_test_row_id: self.cur_test_row_id + batch_size]
            self.cur_test_row_id += batch_size
        return data_rows, False
    
    def head_reset(self):
        self.cur_train_row_id = 0
        self.cur_test_row_id = 0


def MSE_anom_batch_loss_fn(mse_loss_fn: MSELoss, output_ls: List[Tensor], target_ls: List[Tensor]) -> Tensor:
    assert len(output_ls) == len(target_ls), "batch size between output and target should be the same."
    total_loss = float(0)
    for i,output in enumerate(output_ls):
        assert output.shape == target_ls[i].shape, "reserved tokens hidden states should have the same shape."
        loss = mse_loss_fn(output, target_ls[i])
        total_loss += loss
    batch_size = i+1
    return total_loss, batch_size
        

def compressor_batch_loss_calc(compress_model: compress_Transformer, llm_model: gptFast.Transformer, data_rows: dict, loss_fn: MSELoss, device: str = "cuda") -> Tensor:
    with torch.device(device=device):
        padded_braced_tokens = data_rows['padded_braced_tokens'].to(device)
        braced_len = data_rows['braced_len'].to(device)
        padded_reserved_tokens = data_rows['padded_reserved_tokens'].to(device)
        eval_seq_len = data_rows['eval_seq_len'].to(device)
        extracted_hidden_states = data_rows['extracted_hidden_states']
        for i, hidden_state in enumerate(extracted_hidden_states):
            extracted_hidden_states[i] = hidden_state.to(device)
        
        compressor_output = compress_model(idx = padded_braced_tokens, seq_lens = braced_len, tok_level_pad_mask = True, drop_out_p=DROP_OUT_P)
        compressed_tokens = compressor_output[:, -1, :] * embedding_scale # choose the last token's out put as the compressed token
        with torch.no_grad():
            to_eval_output = llm_model(idx = padded_reserved_tokens, seq_lens = eval_seq_len, tok_level_pad_mask = True,
                                    compression_eval = True, compressed_tokens = compressed_tokens)
            batch_hidden_states = extract_single_compressed_hidden_states(to_eval_output, seq_len_lst=eval_seq_len, layer_idx=32, pad_mode='left', device=device)
        
        batch_loss, batch_size = MSE_anom_batch_loss_fn(loss_fn, batch_hidden_states, extracted_hidden_states)
        average_batch_loss = batch_loss/batch_size
        
        return average_batch_loss
        
     

def train_loop(compress_model: compress_Transformer, llm_model: gptFast.Transformer, tokenizer: gptFastTokenizer.TiktokenWrapper, dataloader: simple_Dataset_dataloader):
    compress_model.train()
    llm_model.eval()
    
    dataloader.head_reset()
    loss_fn = MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(compress_model.parameters(), lr=LR)
    
    total_loss = float(0)
    training_step = 0
    end_flag = False
    while not end_flag:
        data_rows, end_flag = dataloader.dataset_get_batch(BATCH_SIZE, split='train')
        if end_flag:
            print('End of train dataset')
            dataloader.head_reset()
            # return end_flag
        # calc loss and train
        average_batch_loss = compressor_batch_loss_calc(compress_model, llm_model, data_rows, loss_fn, device)
        wb_run.log({"[LOSS] train_loss": average_batch_loss.item()})
        # update accumulate loss
        total_loss += average_batch_loss
        # trainer step
        if training_step % update_freq == 0:
            total_loss = total_loss/update_freq
            print(f"Train loss: {total_loss.item()}")
            optimizer.zero_grad()
            total_loss.backward
            optimizer.step()
            total_loss = 0
        training_step += 1
        # eval
        if training_step % eval_freq == 1:
            data_rows, end_flag = dataloader.dataset_get_batch(BATCH_SIZE, split='test')
            if end_flag:
                print('End of test dataset')
                return end_flag
            with torch.no_grad():
                test_average_batch_loss = compressor_batch_loss_calc(compress_model, llm_model, data_rows, loss_fn)
                wb_run.log({"[LOSS] test_loss": test_average_batch_loss.item()})
                print(f"----- Test loss: {test_average_batch_loss.item()} -----")
        
    return end_flag

def main():
    model_args = gptFast.ModelArgs(**gptFast.transformer_configs["Llama-3-8B"], max_seq_length=MAX_seq_len, output_hidden_states=True, output_attentions=True)
    # load tokenizer, llama3 uses tiktoken
    tokenizer = gptFastTokenizer.TiktokenWrapper("/home/yuhao/work/code_repo/gpt-fast/tokenizer.model")
    llm_model = gptFast.Transformer.from_pretrained(model_args, "/home/yuhao/work/code_repo/gpt-fast/consolidated.00.pth", device)
    compressor_args = gptFast.ModelArgs(**transformer_configs["compressor"], max_seq_length=MAX_seq_len, output_hidden_states=False, output_attentions=False)
    compress_model = compress_Transformer.creat_compressor(compressor_args, device,
                                                         embedding_model_dict_path="consolidated.00.pth", 
                                                         model_dict_key='tok_embeddings.weight').to(device)
    
    dataset_path = Path("dataset/to_train_stage_5")
    compress_datasets = []
    for i, data_file_path in enumerate(dataset_path.glob("*.parquet*")):
        ds = Dataset.from_parquet(str(data_file_path))
        if i == 0: print(ds.column_names)
        compress_datasets.append(ds)
        if i >= datasets_num: break
    full_dataset = concatenate_datasets(compress_datasets)
    full_dataset.set_format(type='torch', columns=['padded_braced_tokens', 'padded_reserved_tokens', 'extracted_hidden_states', 'eval_seq_len', 'braced_len', 'ids'])
    full_dataset_dict = full_dataset.train_test_split(test_size=0.1)
    dataloader = simple_Dataset_dataloader(full_dataset_dict)
    
    train_loop(compress_model, llm_model, tokenizer, dataloader)
    # save trained model
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    torch.save(compress_model.state_dict(), os.path.join("src/compressor/trained_models", f"{current_time}_compress_model.pth"))
    
if __name__ == "__main__":
    main()