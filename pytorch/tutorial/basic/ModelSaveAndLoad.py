# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/11 15:30
import random

import torch

from pytorch.tutorial.basic.QuickStart import NeuralNetwork, load_data

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def main(model_path):
    training_data, testing_data = load_data()

    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        for i in random.sample(range(len(testing_data)), 10):
            X, y = testing_data[i][0], testing_data[i][1]
            pred = model(X)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Sample: {i}, Predicted: "{predicted}", Actual: "{actual}"')


if __name__ == '__main__':
    main(model_path='model.pth')
