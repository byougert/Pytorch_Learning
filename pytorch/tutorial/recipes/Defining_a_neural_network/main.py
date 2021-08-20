# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/17 15:01

import torch
from pytorch.tutorial.recipes.Defining_a_neural_network.Module import NeuralNetwork


def main():
    random_data = torch.randn((1, 1, 28, 28))
    my_nn = NeuralNetwork()
    result = my_nn(random_data)
    print(result)


if __name__ == '__main__':
    main()