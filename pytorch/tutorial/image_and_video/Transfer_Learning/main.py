# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/5/7 16:10

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import models

from pytorch.tutorial.image_and_video.Transfer_Learning.model import train
from pytorch.tutorial.image_and_video.Transfer_Learning.data import load_data


def main(model_dir):
    dataloaders, dataset_sizes, class_names = load_data('hymenoptera_data')
    model_ft = models.resnet18(pretrained=True)
    if fixed:
        for param in model_ft.parameters():
            param.requires_grad = False
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, device, num_epochs=25)
    torch.save(model_ft, model_dir)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')
    fixed = True
    main('model_fixed.pth' if fixed else 'model.pth')