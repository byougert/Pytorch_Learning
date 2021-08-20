# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/23 18:21
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch.tutorial.Learning_PyTorch.Classifier.data import load_data
from pytorch.tutorial.Learning_PyTorch.Classifier.module import Net


def train(trainloader, optimizer, net, criterion, epochs: int):
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def main(data_dir, batch_size, model_path, epochs):
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    trainloader, testloader, classes = load_data(data_dir, batch_size)
    train(trainloader, optimizer, net, criterion, epochs)

    torch.save(net.state_dict(), model_path)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    main(data_dir='data', batch_size=4, model_path='cifar_net.pth', epochs=3)
