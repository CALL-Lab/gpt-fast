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
tokenizer = gptFastTokenizer.TiktokenWrapper("tokenizer.model")
model = gptFast.Transformer.from_pretrained(model_args, "consolidated.00.pth", device)
token_1 = tokenizer.encode("A quick brown fox")
# token padding test
token_2 = tokenizer.encode("A quick brown fox text")
token_lst = []
token_lst.append(token_1)
token_lst.append(token_2)
# encoded_pad = tokenizer.encode("<|reserved_special_token_3|>")
encoded_pad = [0]
max_seq_length = 32
for seq_id in range(len(token_lst)):
    token_lst[seq_id] = simp_lst_pad(encoded_pad, max_seq_length, token_lst[seq_id])
tokens = np.array(token_lst)
print(tokens)
outputs = model(torch.tensor(tokens).to(device), [torch.arange(len(seq)).to(device) for seq in token_lst] )
compress_attn = extract_attn_map(outputs, layer_idx=0, compress_idx=torch.tensor([2,3]))
print(outputs.hidden_states)
print(outputs.attentions)
pred = torch.argmax(outputs.logits, dim=2)
print(pred)
outputs_txt = tokenizer.decode(list(pred[1]))
print(outputs)