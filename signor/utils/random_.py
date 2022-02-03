import torch
import numpy as np
import random

from signor.monitor.probe import summary


def random_mask(size, ratio=0.5):
    mask = torch.FloatTensor(*size).uniform_() > ratio
    return mask

def random_binary():
    pass

def fix_seed(seed=42):
    " this is the first version for graph coarse project. But it seems that sometimes it will cause error for other projects. "
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def fix_seed2(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    np.random.seed(42)
    summary(np.random.random(), 'random')
    exit()

    size = (10, 1)
    print(random_mask(*size, ratio=0.5))

