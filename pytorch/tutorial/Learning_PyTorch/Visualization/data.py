# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/27 19:23

import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST


def load_data(batch_size, data_dir='./data'):
    # transforms
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    # datasets
    trainset = FashionMNIST(data_dir,
                            download=True,
                            train=True,
                            transform=transform)
    testset = FashionMNIST(data_dir,
                           download=True,
                           train=False,
                           transform=transform)

    # dataloaders
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=0)

    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # constant for classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    return trainloader, testloader, classes


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()
