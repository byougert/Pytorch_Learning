# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/5/21 15:30

import torch
from torch import nn


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.output_size = 1
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=20,
            num_layers=1,
            batch_first=True
        )
        self.linear = nn.Linear(self.rnn.hidden_size, self.output_size)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        out = out.view(-1, self.rnn.hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, h
