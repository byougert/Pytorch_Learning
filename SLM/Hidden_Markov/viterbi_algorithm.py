# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/28 18:12

import numpy as np


def optimal_sequence(A, B, PI, O):
    """
    Given HMM(hidden Markov model), λ = (A, B, PI) and observation sequence: O,
    calculate optimal state sequence.

    :param A:
        probability matrix of state transfer
    :param B:
        probability matrix of state to observation
    :param PI:
        probability distribution of initial state
    :param O:
        observation sequence

    :return:
        optimal state sequence
    """
    N, N = A.shape
    T, = O.shape
    delta = np.zeros(shape=(T, N))
    psi = np.zeros(shape=(T, N))
    for i in range(N):
        delta[0][i] = PI[i] * B[i][O[0]]
    for t in range(1, T):
        for i in range(N):
            delta[t][i] = max(delta[t-1][j] * A[j][i] for j in range(N)) * B[i][O[t]]
            psi[t][i] = np.argmax([delta[t-1][j] * A[j][i] for j in range(N)])
    seq = np.zeros(T, dtype=np.int)
    seq[T-1] = np.argmax([delta[T-1][i] for i in range(N)])
    for t in range(T-2, -1, -1):
        seq[t] = psi[t+1][seq[t+1]]
    return delta, psi, seq
