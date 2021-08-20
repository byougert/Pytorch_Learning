# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/13 14:50

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_data(one_hot: bool = False, size=10):
    def one_hot_transform(y):
        return torch.zeros(size, dtype=torch.long).scatter_(0, torch.tensor(y), value=1)

    target_transform = one_hot_transform if one_hot else None

    training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=target_transform
    )
    testing_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=target_transform
    )
    return training_data, testing_data


if __name__ == '__main__':
    train_data, test_data = load_data(True)
    print(test_data[0])
