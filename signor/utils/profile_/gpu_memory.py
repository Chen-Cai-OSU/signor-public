# Created at 2021-03-10
# Summary: profile gpu memory usage. https://github.com/Stonesjtu/pytorch_memlab

import torch
from pytorch_memlab import profile, set_target_gpu

from signor.monitor.time import tf



@tf
@profile
def func():
    net1 = torch.nn.Linear(1024, 1024).cuda(0)
    set_target_gpu(1)
    net2 = torch.nn.Linear(1024, 1024).cuda(1)
    set_target_gpu(0)
    net3 = torch.nn.Linear(1024, 1024).cuda(0)

func()