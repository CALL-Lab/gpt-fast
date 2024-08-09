import src.gptfast.model as gptFast
import src.gptfast.tokenizer as gptFastTokenizer
import torch
import numpy as np


def extract_attn_map(outputs, layer_idx=0, compress_idx=None):
    '''
    compress_idx: [num_compressed_tokens]
    if not None, the attention map will be compressed to the specified index

    '''
    # get the attention map from the model
    attn = outputs.attentions[layer_idx]# shape is (batch_size, num_heads, seq_len, max_seq_len)
    bsz, num_heads, seq_len, max_seq_len = attn.shape
    idxs = torch.arange(seq_len)
    non_compress_idx = torch.tensor([i for i in idxs if i not in compress_idx])
    compress_attn = attn[:,:,compress_idx,:]
    attn_map = compress_attn[:,:,:,non_compress_idx]
    return attn_map

def simp_lst_pad(encoded_pad, max_seq_length, origin_seq, mode='left'): # mode='left' or 'right'
    if mode == 'left':
        return  encoded_pad * (max_seq_length - len(origin_seq)) + origin_seq
    elif mode == 'right':
        return origin_seq + encoded_pad * (max_seq_length - len(origin_seq))

# load llama3 model
device = "cuda"
model_args = gptFast.ModelArgs(**gptFast.transformer_configs["Llama-3-8B"], max_seq_length=32, output_hidden_states=True, output_attentions=True)
# load tokenizer, llama3 uses tiktoken
tokenizer = gptFastTokenizer.TiktokenWrapper("/home/yuhao/work/code_repo/gpt-fast/tokenizer.model")
model = gptFast.Transformer.from_pretrained(model_args, "/home/yuhao/work/code_repo/gpt-fast/consolidated.00.pth", device)
token_1 = tokenizer.encode("A quick brown fox")
# token padding test
token_2 = tokenizer.encode("A quick brown fox text")
token_lst = []
len_lst = []
token_lst.append(token_1)
len_lst.append(len(token_1))
token_lst.append(token_2)
len_lst.append(len(token_2))
# encoded_pad = tokenizer.encode("<|reserved_special_token_3|>")
encoded_pad = [0]
max_seq_length = 32
for seq_id in range(len(token_lst)):
    token_lst[seq_id] = simp_lst_pad(encoded_pad, max_seq_length, token_lst[seq_id])
tokens = np.array(token_lst)
print(tokens)
outputs = model.forward(torch.tensor(tokens).to(device), torch.tensor(np.array(len_lst)).to(device), tok_level_pad_mask=False) # [torch.arange(seq_len).to(device) for seq_len in len_lst]
compress_attn = extract_attn_map(outputs, layer_idx=0, compress_idx=torch.tensor([2,3]))
# print(outputs.hidden_states)
# print(outputs.attentions)
pred = torch.argmax(outputs.logits, dim=2)
# print(pred)
for seq_id in range(len(len_lst)):
    print(tokenizer.decode(list(pred[seq_id][-len_lst[seq_id]:])))
    
'''
An interesting thing to note:

When 'tok_level_pad_mask' switch is on, the padding token is absolutely masked out \
in token embedding token (to 0) and in attention map (to -inf). The output of the \
mode model is as follows (the padding token is clamped out):

['. check fox', '. look fox jumps']

When 'tok_level_pad_mask' is switched off, the padding token is only masked out \
in attention map (to -inf). The output of the mode model is as follows \
(the padding token is clamped out):

['. check fox', '. look fox jumps']
'''