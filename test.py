import torch
import random
import logging
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel

logging.basicConfig(level=logging.INFO)

path = "D:/ai/gpt-2/models/M"


def select_top_k(predictions, k=10):
    predicted_index = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]
    ).item()
    return predicted_index


tokenizer = GPT2Tokenizer.from_pretrained(path)

text = "A scientist made a new clever robot"
indexed_tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([indexed_tokens])

model = GPT2LMHeadModel.from_pretrained(path)

total_predicted_text = text
n = 500
for _ in range(n):
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    predicted_index = select_top_k(predictions, k=10)
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    total_predicted_text += tokenizer.decode(predicted_index)

    if "<|endoftext|>" in total_predicted_text:
        break

    indexed_tokens += [predicted_index]
    tokens_tensor = torch.tensor([indexed_tokens])

print(total_predicted_text)
