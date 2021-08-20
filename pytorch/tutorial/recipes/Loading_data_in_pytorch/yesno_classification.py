# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/17 14:14

from pytorch.tutorial.recipes.Loading_data_in_pytorch.data import load_data
from torch.utils.data import DataLoader


def main():
    yesno_data = load_data()
    data_loader = DataLoader(yesno_data, batch_size=1, shuffle=True)
    print(len(data_loader))


if __name__ == '__main__':
    main()