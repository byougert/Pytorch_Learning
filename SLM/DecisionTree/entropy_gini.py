# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/12 16:24

import matplotlib.pyplot as plt
import numpy as np


def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2(1 - p)


def gini(p):
    return 1 - p ** 2 - (1 - p) ** 2


def main():
    p = np.linspace(0, 1, 200)
    y_entropy = entropy(p)
    y_gini = gini(p)

    plt.figure(figsize=(8, 8))
    plt.xlim(0, 1)
    plt.ylim(-0.2, 1.1)
    l1, = plt.plot(p, y_entropy, c='#00ffff', linewidth=2)
    l2, = plt.plot(p, y_gini, c='#ff0099', linewidth=2)
    ax = plt.gca()
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    ax.xaxis.set_ticks_position('bottom')  # 设置bottom为x轴
    ax.yaxis.set_ticks_position('left')  # 设置left为x轴
    ax.spines['bottom'].set_position(('data', 0))  # 这个位置的括号要注意
    ax.spines['left'].set_position(('data', 0))
    plt.legend(handles=[l1, l2], labels=['Entropy', 'Gini'], loc='lower right')

    plt.show()


if __name__ == '__main__':
    main()
