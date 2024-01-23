import torch
import random
import logging
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)

path = "D:/ai/gpt-2/models/M"


def select_top_k(predictions, k=10):
    predicted_index = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]
    ).item()
    return predicted_index


tokenizer = GPT2Tokenizer.from_pretrained(path)

with open("./test.txt", "r") as f:
    dataset = f.read()

indexed_text = tokenizer.encode(dataset)


dataset_cut = []
len_block = 512
for i in range(len(indexed_text) // len_block):
    dataset_cut.append(indexed_text[i * len_block : i * len_block + len_block])

dataset_tensor = torch.tensor(dataset_cut)


train_set = TensorDataset(dataset_tensor, dataset_tensor)
train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=False)

device = torch.device("cpu")

from torch import nn
from torch.autograd import Variable
import time

pre = time.time()

epoch = 50  # 循环学习 50 次

model = GPT2LMHeadModel.from_pretrained(path)
model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 定义优化器

for i in range(epoch):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)

        optimizer.zero_grad()

        loss, logits, _ = model(data, labels=target)

        total_loss += loss

        loss.backward()
        optimizer.step()

        if batch_idx == len(train_loader) - 1:
            # 在每个 Epoch 的最后输出一下结果
            print("average loss:", total_loss / len(train_loader))

print("训练时间：", time.time() - pre)
