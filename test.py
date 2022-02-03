# Created by Chen at 2/2/22

from signor.utils.random_ import fix_seed
fix_seed()
print('test fix_seed')

import os
from signor.ioio.dir import cur_dir
print(eval(cur_dir()))
print('test cur_dir')

from signor.format.format import red
print(red('test red'))

from signor.monitor.probe import summary
import numpy as np
x = np.random.random((10, 10))
summary(x, 'x')

from signor.monitor.time import timefunc
from time import sleep

@timefunc
def test_timefund():
    sleep(2)

from signor.viz.matrix import matshow
matrix = np.random.random((10, 10))
matshow(matrix, var_name='test matrix')

from signor.configs.util import subset_dict, dict2name
d = {'a': 1, 'b': 2}
print(dict2name(d))

from signor.utils.np import tonp
import torch
x = np.random.random((3,3))
x = torch.tensor(x)
print(tonp(x))


