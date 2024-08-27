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
import torch.optim.lr_scheduler as lr_scheduler

BATCH_SIZE = 5
EPOCHS = 50
DATASET_LENGTH = 125000
lr_update_freq = DATASET_LENGTH//(BATCH_SIZE*EPOCHS)
update_freq = 1
eval_freq = 20
grad_check_freq = 30
LR = 1e-3 #5e-2
DROP_OUT_P = 0.0
datasets_num = 14
MAX_seq_len = 96
embedding_scale = 3e-4
natural_language_test_freq = 10*eval_freq 

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
        "origin_lr": LR,
        "lr_update_freq": lr_update_freq,
        "lr_scheduler": "CosineAnnealingLR",
        "drop_out_p": DROP_OUT_P,
        "embedding_scale": embedding_scale,
        "datasets_num": datasets_num,
        "max_seq_len": MAX_seq_len,
        "device": device,
    }
)
wb_table = wb.Table(columns=["[PRED] compress_pred", "[PRED] origin_pred"])

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


def MSE_anom_batch_loss_fn(mse_loss_fn: MSELoss, output_ls: List[Tensor], target_ls: List[Tensor]) -> Tuple[Tensor,int]:
    assert len(output_ls) == len(target_ls), "batch size between output and target should be the same."
    losses = []
    for i,output in enumerate(output_ls):
        assert output.shape == target_ls[i].shape, "reserved tokens hidden states should have the same shape."
        loss = mse_loss_fn(output, target_ls[i])
        losses.append(loss)
    batch_size = i+1
    b_loss = torch.stack(losses).mean()
    return b_loss, batch_size
        

def compressor_batch_loss_calc(compress_model: compress_Transformer, llm_model: gptFast.Transformer, data_rows: dict, loss_fn: MSELoss, device: str = "cuda", nl_lang_test: bool=False) -> Tensor:
    with torch.device(device=device):
        padded_braced_tokens = data_rows['padded_braced_tokens'].to(device)
        braced_len = data_rows['braced_len'].to(device)
        padded_reserved_tokens = data_rows['padded_reserved_tokens'].to(device)
        eval_seq_len = data_rows['eval_seq_len'].to(device)
        extracted_hidden_states = data_rows['extracted_hidden_states']
        for i, hidden_state in enumerate(extracted_hidden_states):
            extracted_hidden_states[i] = hidden_state.to(device, dtype=torch.bfloat16)
        
        compressor_output = compress_model(idx = padded_braced_tokens, seq_lens = braced_len, tok_level_pad_mask = True, drop_out_p=DROP_OUT_P)
        compressed_tokens = compressor_output[:, -1, :] * embedding_scale # choose the last token's out put as the compressed token
        # with torch.no_grad():
        to_eval_output = llm_model(idx = padded_reserved_tokens, seq_lens = eval_seq_len, tok_level_pad_mask = True,
                                compression_eval = True, compressed_tokens = compressed_tokens)
        batch_hidden_states = extract_single_compressed_hidden_states(to_eval_output, seq_len_lst=eval_seq_len, layer_idx=32, pad_mode='left', device=device)
        
        avg_batch_loss, batch_size = MSE_anom_batch_loss_fn(loss_fn, batch_hidden_states, extracted_hidden_states)
            
        # test natural language
        if nl_lang_test:
            return avg_batch_loss, torch.argmax(to_eval_output.logits, dim=2), data_rows['pred']
        
        return avg_batch_loss
        
     

def train_loop(compress_model: compress_Transformer, llm_model: gptFast.Transformer, tokenizer: gptFastTokenizer.TiktokenWrapper, dataloader: simple_Dataset_dataloader):
    # model setup
    compress_model.train()
    llm_model.eval()
    # dataloader setup
    dataloader.head_reset()
    # trainer setup
    loss_fn = MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(compress_model.parameters(), lr=LR)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_update_freq)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=1e-5)
    
    total_loss = torch.tensor(0.0, dtype=torch.bfloat16, device=device)
    training_step = 0
    end_flag = False
    while not end_flag:
        data_rows, end_flag = dataloader.dataset_get_batch(BATCH_SIZE, split='train')
        if end_flag:
            print('End of train dataset')
            # dataloader.head_reset()
            return end_flag
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
            total_loss.to(dtype=torch.bfloat16).backward()
            # grad check
            if training_step % grad_check_freq == 1:
                grad_dict = {}
                for name, param in compress_model.named_parameters():
                    if name == 'tok_embeddings.weight': continue
                    if param.grad is not None:
                        grad_dict[f"[Grad] {name}_grad_mean"] = param.grad.mean().item()
                        grad_dict[f"[Grad] {name}_grad_max"] = param.grad.max().item()
                        grad_dict[f"[Grad] {name}_grad_min"] = param.grad.min().item()
                    else: print(f"Alert!!!!! ===== {name} has no grad. ===== !!!!!Alert")
            # step
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
            if training_step % natural_language_test_freq == 1:
                # test natural language
                with torch.no_grad():
                    test_nl_loss, compress_pred, origin_pred = compressor_batch_loss_calc(compress_model, llm_model, data_rows, loss_fn, nl_lang_test=True)
                    compress_pred = compress_pred.detach().cpu().tolist()
                    origin_pred = origin_pred.detach().cpu().tolist()
                    for seq_id in range(len(origin_pred)):
                        l_cp_pred = tokenizer.decode(compress_pred[seq_id])
                        l_org_pred = tokenizer.decode(origin_pred[seq_id])
                        wb_table.add_data(*[l_cp_pred, l_org_pred])
                        print(f"==========================nl_test_{seq_id}===========================")
                        print(f"Origin pred: {l_org_pred} ---> Compress pred: {l_cp_pred}\n")
                wb.log({"pred_compare_table": wb_table})
        # lr update
        scheduler.step()
                
    return end_flag

def main():
    torch.autograd.set_detect_anomaly(True)
    
    model_args = gptFast.ModelArgs(**gptFast.transformer_configs["Llama-3-8B"], max_seq_length=MAX_seq_len, output_hidden_states=True, output_attentions=True)
    # load tokenizer, llama3 uses tiktoken
    tokenizer = gptFastTokenizer.TiktokenWrapper("/home/yuhao/work/code_repo/gpt-fast/tokenizer.model")
    llm_model = gptFast.Transformer.from_pretrained(model_args, "/home/yuhao/work/code_repo/gpt-fast/consolidated.00.pth", device)
    for param in llm_model.parameters():
        param.requires_grad = False
    compressor_args = gptFast.ModelArgs(**transformer_configs["compressor"], max_seq_length=MAX_seq_len, output_hidden_states=False, output_attentions=False)
    compress_model = compress_Transformer.creat_compressor(compressor_args, device,
                                                         embedding_model_dict_path="consolidated.00.pth", 
                                                         model_dict_key='tok_embeddings.weight').to(device, dtype=torch.bfloat16)
    
    dataset_path = Path("dataset/to_train_stage_5")
    compress_datasets = []
    for i, data_file_path in enumerate(dataset_path.glob("*.parquet*")):
        ds = Dataset.from_parquet(str(data_file_path))
        if i == 0: print(ds.column_names)
        compress_datasets.append(ds)
        if i >= datasets_num: break 
    full_dataset = concatenate_datasets(compress_datasets)
    full_dataset.set_format(type='torch', columns=['padded_braced_tokens', 'padded_reserved_tokens', 'extracted_hidden_states', 'eval_seq_len', 'braced_len', 'ids', 'pred'])
    full_dataset_dict = full_dataset.train_test_split(test_size=0.1)
    dataloader = simple_Dataset_dataloader(full_dataset_dict)
    
    train_loop(compress_model, llm_model, tokenizer, dataloader)
    # save trained model
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    torch.save(compress_model.state_dict(), os.path.join("src/compressor/trained_models", f"{current_time}_compress_model.pth"))
    
if __name__ == "__main__":
    main()
