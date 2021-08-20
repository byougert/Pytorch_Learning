# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/23 17:42

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_data(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def __main(data_dir, batch_size):
    trainloader, testloader, classes = load_data(data_dir, batch_size)
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(len(trainloader.dataset))

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))


if __name__ == '__main__':
    __main(data_dir='data', batch_size=4)
