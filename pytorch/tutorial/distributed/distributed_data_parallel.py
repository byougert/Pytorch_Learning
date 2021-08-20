# -*- coding:utf-8 -*-
# @Author : Byougert
# @Time : 2021/8/12 12:15

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed import Backend
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def main_work(rank, world_size):
    dist.init_process_group(Backend.NCCL, rank=rank, world_size=world_size)
    model = nn.Linear(10, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()


def __main():
    world_size = 2
    mp.spawn(main_work, args=(world_size,), nprocs=world_size)


if __name__ == '__main__':
    __main()