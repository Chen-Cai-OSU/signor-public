# Created at 5/13/21
# Summary: profile python memory

import torch
from memory_profiler import profile


def my_profile(on=False):
    if on:
        return profile
    else:
        return lambda x: x

@my_profile(on=False)
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    del a
    a = 1
    return a

@my_profile(on=False)
def my_func2():
    n = 1000
    x = torch.rand(n, n)
    return x@x

if __name__ == '__main__':
    my_func()
