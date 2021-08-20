# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/28 16:12

import numpy as np


def forward(A, B, PI, O):
    """
    Given HMM(hidden Markov model), λ = (A, B, PI) and observation sequence: O,
    calculate the probability: P(O|λ) with forward algorithm.

    @see: backward(A, B, PI, O)

    :param A:
        probability matrix of state transfer
    :param B:
        probability matrix of state to observation
    :param PI:
        probability distribution of initial state
    :param O:
        observation sequence

    :return:
        alpha, P(O|λ)
    """
    N, N = A.shape
    T, = O.shape
    alpha = np.zeros(shape=(T, N))
    for i in range(N):
        alpha[0][i] = PI[i] * B[i][O[0]]
    for t in range(T-1):
        for i in range(N):
            alpha[t+1][i] = sum([alpha[t][j] * A[j][i] for j in range(N)]) * B[i][O[t+1]]
    return alpha, sum(alpha[-1])


def backward(A, B, PI, O):
    """
    Given HMM(hidden Markov model), λ = (A, B, PI) and observation sequence: O
    calculate the probability: P(O|λ) with backward algorithm.

    :param A:
        probability matrix of state transfer
    :param B:
        probability matrix of state to observation
    :param PI:
        probability distribution of initial state
    :param O:
        observation sequence

    :return: 
        beta, P(O|λ)
    """
    N, N = A.shape
    T, = O.shape
    beta = np.zeros(shape=(T, N))
    beta[T-1] = 1
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t][i] = sum([A[i][j] * B[j][O[t+1]] * beta[t+1][j] for j in range(N)])
    return beta, sum([PI[i] * B[i][O[0]] * beta[0][i] for i in range(N)])


