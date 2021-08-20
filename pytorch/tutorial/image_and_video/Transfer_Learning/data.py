# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/5/7 15:00

import numpy as np
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def load_data(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    root = Path(data_dir)
    img_datasets = {x: datasets.ImageFolder(root / x, data_transforms[x]) for x in ('train', 'val')}
    dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(img_datasets[x], batch_size=4, shuffle=True) for x in ('train', 'val')}
    class_names = img_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names


def imshow(inp, plt, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated
    # plt.show()
