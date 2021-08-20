# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/5/21 20:41

import torch
import matplotlib.pyplot as plt
from NN_and_DL.RNN.data import load_data


def show(x, y, preds, time_steps):
    x = x.data.numpy().flatten()
    y = y.data.numpy()
    plt.scatter(time_steps[:-1], x.flatten(), s=90)
    plt.plot(time_steps[:-1], x.flatten())
    plt.scatter(time_steps[:-1], preds)
    plt.show()


def test(model, x, y, time_steps, h):
    predictions = []
    start = x[:, 0, :]
    h = torch.zeros(model.rnn.num_layers, 1, model.rnn.hidden_size)

    for _ in range(x.shape[1]):
        start = start.view(1, 1, 1)
        pred, h = model(start, h)
        start = pred
        predictions.append(pred.detach().numpy().flatten()[0])

    show(x, y, predictions, time_steps)


def _main(model_dir):
    x, y, time_steps = load_data(50, 0)
    model, h = torch.load(model_dir)
    model, h = model.cpu(), h.cpu()
    test(model, x, y, time_steps, h)


if __name__ == '__main__':
    _main(model_dir='model.pth')