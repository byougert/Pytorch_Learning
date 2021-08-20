# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/5/21 17:10

import torch
from torch import nn, optim
from NN_and_DL.RNN.data import load_data
from NN_and_DL.RNN.model import Net

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = Net().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

x, y, time_steps = load_data(50, 1)
x = x.to(device)
y = y.to(device)
h = torch.zeros(model.rnn.num_layers, 1, model.rnn.hidden_size)
h = h.to(device)


def train(epochs):
    global h
    for epoch in range(epochs):
        output, h = model(x, h)
        h = h.detach()

        loss = criterion(output, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 1000 == 0:
            print(f'Epoch: {epoch+1}   Loss: {loss.item()}')


def _main(model_dir):
    train(epochs=10000)
    torch.save((model, h), model_dir)


if __name__ == '__main__':
    _main(model_dir='model.pth')