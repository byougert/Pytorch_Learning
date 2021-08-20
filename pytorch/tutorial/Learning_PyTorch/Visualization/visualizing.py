# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/27 19:42

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from pytorch.tutorial.Learning_PyTorch.Visualization.data import load_data, matplotlib_imshow
from pytorch.tutorial.Learning_PyTorch.Visualization.model import Net, fit, test

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# get some random training images
trainloader, testloader, classes = load_data(batch_size=4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


def graph():
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    # writer.add_image('four_fashion_mnist_images', img_grid)

    writer.add_graph(Net(), images)
    writer.close()


def embedding():
    # helper function
    def select_n_random(data, labels, n=100):
        '''
        Selects n random datapoints and their corresponding labels from a dataset
        '''
        assert len(data) == len(labels)

        perm = torch.randperm(len(data))
        return data[perm][:n], labels[perm][:n]

    # select random images and their target indices
    images, labels = select_n_random(trainloader.dataset.data, trainloader.dataset.targets)

    # get the class labels for each image
    class_labels = [classes[lab] for lab in labels]

    # log embeddings
    features = images.view(-1, 28 * 28)
    writer.add_embedding(features,
                         metadata=class_labels,
                         label_img=images.unsqueeze(1))
    writer.close()


def tracking(model_dir):
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    fit(2, trainloader, net, optimizer, criterion, writer, classes, device)
    torch.save(net, model_dir)


def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()


def pr_curve(model_dir):
    net = torch.load(model_dir)
    test_probs, test_label = test(testloader, net, device)

    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_label)


if __name__ == '__main__':
    pr_curve(model_dir='model.pth')
