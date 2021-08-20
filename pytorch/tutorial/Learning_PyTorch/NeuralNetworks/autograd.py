# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/4/23 15:29

import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
q = 3 * a ** 2 - b ** 2
print('Tensor Q:', q)
q.backward(gradient=torch.tensor([2., 1.]))
print(a.grad)
print(b.grad)