# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/26 19:52

from pathlib import Path

import requests
import torch
import pickle
import gzip
from matplotlib import pyplot
import numpy as np


def load_data():
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
    return x_train, y_train, x_valid, y_valid


def _main():
    load_data()


if __name__ == '__main__':
    _main()
