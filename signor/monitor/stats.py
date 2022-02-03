import numpy as np
import torch
from signor.format.format import pf
from functools import partial
from signor.utils.np import tonp

def stats(x, precision = 2, verbose = True, var_name='None'):
    """
    print the stats of a (np.array, list, pt.Tensor)

    :param x:
    :param precision:
    :param verbose:
    :return:
    """
    if isinstance(x, torch.Tensor): x = tonp(x)
    assert isinstance(x, (list, np.ndarray)), 'stats only take list or numpy array'

    ave_ = np.mean(x)
    median_ = np.median(x)
    max_ = np.max(x)
    min_ = np.min(x)
    std_ = np.std(x)
    pf_ = partial(pf, precision=precision)

    if verbose:
        ave_, min_, max_, median_, std_ = list(map(pf_, [ave_, min_, max_, median_, std_]))
        line = '{:>25}: {:>5}(mean) {:>5}(min) {:>5}(max) {:>5}(median) {:>5}(std)'.format(var_name, ave_, min_, max_, median_, std_)
        print(line)
    return list(map(pf_, [ave_, min_, max_, median_, std_]))

if __name__ == '__main__':
    # line_new = "{:>20}  {:>25} {:>15}".format(10, 20, 30)
    # print(line_new)
    # exit()
    # x = np.random.random((100, 3))
    # print(stats(x))

    x = torch.rand((100, 3))
    print(x.__class__())
    stats(x, var_name='x')
    stats(x, var_name='abc'*3)
