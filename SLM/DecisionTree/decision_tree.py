# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/12 18:16

import numpy as np

from SLM.DecisionTree.data import load_data


def entropy(p):
    return sum(map(lambda x: -x * np.log2(x) if x else 0, p))


def condition_entropy(data, features):
    d = len(data)
    cond_entropy_list = []
    for feature in features[1:-1]:
        feature_group = data.groupby(feature)
        feature_group_size = feature_group.size()
        feature_prob = feature_group_size / d

        cond_entropy = 0.
        for name, group in feature_group:
            cond_entropy += cal_entropy(group.iloc[:, -1], feature_group_size[name]) * feature_prob[name]
        cond_entropy_list.append(cond_entropy)
    return np.array(cond_entropy_list)


def cal_entropy(label, d):
    return entropy(label.groupby(label).size() / d)


def mutual_information(data, feature):
    H_D = cal_entropy(data.iloc[:, -1], len(data))
    H_Di = condition_entropy(data, feature)
    mutual_infor = (H_D - H_Di).tolist()
    print(mutual_infor.index(max(mutual_infor)))


def main():
    training_data, features = load_data(data_path=r'data/dataset.csv', encoding='gbk', header=0)
    mutual_information(training_data, features)


if __name__ == '__main__':
    main()
