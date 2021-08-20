# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/23 16:20

import torch
import torch.nn as nn
from pytorch.tutorial.Learning_PyTorch.NeuralNetworks.module import Net


def main():
    net = Net()
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)

    target = torch.randn(10)
    target = target.view(1, -1)

    criterion = nn.MSELoss()
    loss = criterion(out, target)
    print(loss)

    net.zero_grad()  # zeroes the gradient buffers of all parameters

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)


if __name__ == '__main__':
    main()