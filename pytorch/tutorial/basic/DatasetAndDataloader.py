# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/11 12:17
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from pytorch.tutorial.basic.QuickStart import load_data

training_data, testing_data = load_data()

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
training_loader = DataLoader(training_data, batch_size=64)
print(len(training_loader.dataset))
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
