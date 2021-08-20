# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/28 18:25

import numpy as np
from SLM.Hidden_Markov.probability_calculate import backward, forward
from SLM.Hidden_Markov.viterbi_algorithm import optimal_sequence


def _main():
    A = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])
    PI = np.array([0.2, 0.4, 0.4])
    O = np.array([0, 1, 0, 1])
    alpha, for_prob = forward(A, B, PI, O)
    beta, back_prob = backward(A, B, PI, O)
    print(f'Forward: P(O|λ) = {for_prob: .8f}')
    print(f'Backward: P(O|λ) = {back_prob: .8f}')

    delta, psi, seq = optimal_sequence(A, B, PI, O)
    print(f'Optimal sequence: {seq}')


if __name__ == "__main__":
    _main()
