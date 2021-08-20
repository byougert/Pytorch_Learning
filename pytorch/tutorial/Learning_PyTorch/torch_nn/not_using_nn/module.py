# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/26 20:09

import math
import torch

from pytorch.tutorial.Learning_PyTorch.torch_nn.not_using_nn.data import load_data


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb, weights, bias):
    return log_softmax(xb @ weights + bias)


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def _main():
    weights = torch.randn(784, 10) / math.sqrt(784)
    weights.requires_grad_()
    bias = torch.zeros(10, requires_grad=True)

    bs = 64  # batch size

    x_train, y_train, x_valid, y_valid = load_data()
    n, c = x_train.shape

    xb = x_train[0:bs]  # a mini-batch from x
    preds = model(xb, weights, bias)  # predictions
    print(preds[0], preds.shape)

    loss_func = nll
    yb = y_train[0:bs]
    print('Before:')
    print('loss:', loss_func(preds, yb))
    print('accuracy:', accuracy(preds, yb))

    lr = 0.5  # learning rate
    epochs = 4  # how many epochs to train for

    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            #         set_trace()
            start_i = i * bs
            end_i = start_i + bs
            xa = x_train[start_i:end_i]
            ya = y_train[start_i:end_i]
            pred = model(xa, weights, bias)
            loss = loss_func(pred, ya)

            loss.backward()
            with torch.no_grad():
                weights -= weights.grad * lr
                bias -= bias.grad * lr
                weights.grad.zero_()
                bias.grad.zero_()
    preds = model(xb, weights, bias)
    print('After:')
    print('loss:', loss_func(preds, yb))
    print('accuracy:', accuracy(preds, yb))


if __name__ == '__main__':
    _main()