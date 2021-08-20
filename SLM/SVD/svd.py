# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/5/12 18:09

import numpy as np


def cal_svd(A):
    M, N = A.shape
    W = np.dot(A.T, A)
    print('W:\n', W)
    value, vector = np.linalg.eig(W)
    V = vector.T


def _main():
    A = np.array([[1, 1], [2, 2], [0, 0]])
    print('Origin:\n', A)
    cal_svd(A)


if __name__ == '__main__':
    _main()