# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/8/12 12:25

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):

    def __init__(self, length, size):
        self.length = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        print("\tIn Model: input size", x.size(),
              "output size", output.size())
        return output


def __main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = 5
    output_size = 2
    batch_size = 30
    data_size = 100

    dataloader = DataLoader(dataset=RandomDataset(data_size, input_size), batch_size=batch_size, shuffle=True)

    model = Model(input_size, output_size)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    model = model.to(device)

    for batch in dataloader:
        batch = batch.to(device)
        output = model(


            batch)
        print("Outside: input size", batch.size(),
              "output_size", output.size())


if __name__ == '__main__':
    __main()