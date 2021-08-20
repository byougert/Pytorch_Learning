# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/9 15:58


def load_data(data_dir, label_dir):
    r"""
    Read and load dataset and label

    :param data_dir: str
        Path of the data file

    :param label_dir: str
        Path of the label file

    :return: (list, list)
        A list of the dataset. Every element in the dataset is a tuple,
        indicating a vector.
        A list of the label.
    """
    with open(data_dir, 'r') as data_fp, open(label_dir, 'r') as label_fp:
        data = [eval(line) for line in data_fp.readlines() if line]
        label = [eval(line) for line in label_fp.readlines() if line]
    return data, label
