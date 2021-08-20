# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/5/7 15:01

import torch
from pytorch.tutorial.image_and_video.Transfer_Learning.data import load_data
from pytorch.tutorial.image_and_video.Transfer_Learning.model import visualize_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloaders, dataset_sizes, class_names = load_data('hymenoptera_data')
model = torch.load('model.pth')
visualize_model(model, dataloaders, class_names, device, num_images=6)