# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/13 16:38

from torch import nn


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatter = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatter(x)
        logits = self.linear_relu_stack(x)
        return logits
