# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/9 21:00

import torch
from torch import nn
from torch.utils.data import DataLoader

from log import logger
from pytorch.tutorial.basic.data import load_data
from pytorch.tutorial.basic.module import NeuralNetwork


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 200 == 0:
            loss, current = loss.item(), batch * len(X)
            logger.info(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0., 0.
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += pred.argmax(1).eq(y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    logger.info(f'Test Error: ----------Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n\n')


def main(model_path, batch_size=64, epochs=10):
    training_data, testing_data = load_data(one_hot=False)
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    testing_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

    model = NeuralNetwork().to(device)
    logger.info(str(model))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        logger.info(f'Epoch {epoch + 1} \n{"-" * 8}')
        train(training_loader, model, loss_fn, optimizer)
        test(testing_loader, model, loss_fn)

    logger.info('Done!')

    torch.save(model.state_dict(), model_path)
    logger.info(f'Saved PyTorch Model State to {model_path}')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using {device} device')
    main(model_path='model.pth')
