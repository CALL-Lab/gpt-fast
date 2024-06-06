# gpt-fast

A simple and straight-forward transformer inference implement in PyTorch, forked from [gpt-fast](https://github.com/pytorch-labs/gpt-fast).

This fork uses a huggingface-like API, and the user have access to intermediate variables like attentions and hidden states.

## API

This library uses similar API with [transformers](https://github.com/huggingface/transformers).

### Initializing the model and tokenizer

You can create an instance of the transformer with `gptfast.model.Transformer(config)`, where `config` is a `gptFast.model.ModelArgs` object, which contains the hyperparameters of the transformer. Some predefined configurations are available in `gptfast/model/transformer_configs` in the format of Python dictionaries.

After loading or initializing the weights, you need to invoke the `post_init()` method of the transformer to initialize the positional embedding and causal mask.

You can also use the `from_pretrained(config, path, device)` method to load a pretrained model in one step.

This library supports `sentencepiece` and `tiktoken` tokenizers. A wrapper class `TokenizerInterface` is provided for a unified tokenizer interface. You can load a tokenizer with `gptfast.tokenizer.SentencePieceWrapper(path)` or `gptfast.tokenizer.TiktokenWrapper(path)` to load the tokenizer from a file.

### Inference

Use `TokenizerInterface.encode(text)` to encode a text into token ids and `TokenizerInterface.decode(text)` to decode token ids into a text.

Predict the next token with `model(input_ids, input_pos)` where `model` is an instance of Transformer, and `input_ids` is a sequence of token ids. The length of the input sequence should be less than or equal to the maximum length specified in the config.

The output of the model is a `TransformerOutput` object, which contains the logits of the next token, and optionally the hidden states of each layer (the outputs of each attention layers followed by the output of the normalization layer) and the attention coefficient of each attention layer. To make the model output the hidden states and attention coefficients, you need to specify `output_hidden_states=True` and `output_attentions=True` in the model config when initializing the model.

## A minimal example

```python
import gptfast.model as gptFast
import gptfast.tokenizer as gptFastTokenizer
import torch

# load llama3 model
device = "cuda"
model_args = gptFast.ModelArgs(**gptFast.transformer_configs["Llama-3-8B"], max_seq_length=16, output_hidden_states=True, output_attentions=True)
model = gptFast.Transformer.from_pretrained(model_args, "llama-3-8b/original/consolidated.00.pth", device)
# load tokenizer, llama3 uses tiktoken
tokenizer = gptFastTokenizer.TiktokenWrapper("llama-3-8b/original/tokenizer.model")

# inference
tokens = tokenizer.encode("A quick brown fox")
print(tokens)
outputs = model(torch.tensor([tokens]).to(device), torch.arange(len(tokens)).to(device))
print(outputs.hidden_states)
print(outputs.attentions)
pred = torch.argmax(outputs.logits, dim=2)
print(pred)
outputs = tokenizer.decode(list(pred[0]))
print(outputs)
```