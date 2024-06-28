import src.gptfast.model as gptFast
import src.gptfast.tokenizer as gptFastTokenizer
import torch


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


# load llama3 model
device = "cuda"
model_args = gptFast.ModelArgs(**gptFast.transformer_configs["Llama-3-8B"], max_seq_length=16, output_hidden_states=True, output_attentions=True)
model = gptFast.Transformer.from_pretrained(model_args, "llama-3-8b/original/consolidated.00.pth", device)
# load tokenizer, llama3 uses tiktoken
tokenizer = gptFastTokenizer.TiktokenWrapper("llama-3-8b/original/tokenizer.model")
tokens = tokenizer.encode("A quick brown fox")
print(tokens)
outputs = model(torch.tensor([tokens]).to(device), torch.arange(len(tokens)).to(device))
compress_attn = extract_attn_map(outputs, layer_idx=0, compress_idx=torch.tensor([2,3]))
print(outputs.hidden_states)
print(outputs.attentions)
pred = torch.argmax(outputs.logits, dim=2)
print(pred)
outputs = tokenizer.decode(list(pred[0]))
print(outputs)