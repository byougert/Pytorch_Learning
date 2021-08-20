# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/5/21 16:44

import torch
import numpy as np


def load_data(num_timestamp, start):
    time_steps = np.linspace(start, start+10, num_timestamp)
    data = np.sin(time_steps).reshape(num_timestamp, 1)
    x = torch.tensor(data[:-1, :]).float().view(1, num_timestamp-1, 1)
    y = torch.tensor(data[1:, :]).float().view(1, num_timestamp-1, 1)
    return x, y, time_steps
