# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/17 16:01

import torch
import torch.optim as optim
from pytorch.tutorial.recipes.Defining_a_neural_network.Module import NeuralNetwork


def main():
    net = NeuralNetwork()
    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
    print("Model's state_dict:")
    for state in net.state_dict():
        print('Size:', net.state_dict()[state].size(), '\t', state)
    print()
    for state in optimizer.state_dict():
        print(state, '\t', optimizer.state_dict()[state])


if __name__ == '__main__':
    main()