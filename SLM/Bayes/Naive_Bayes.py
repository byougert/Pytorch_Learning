# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/9 15:52

from collections import Counter

from SLM.Bayes.load import load_data


def prior_probability(label: list):
    counter = Counter(label)


def main():
    data, label = load_data(r'dataset/data.txt', r'dataset/label.txt')
    print(data)
    print(label)


if __name__ == '__main__':
    main()
